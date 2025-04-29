import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import argparse
import wandb

from Gaspard_model.models.models import ShallowNetEncoder, MLPEncoder
from Gaspard_model.models.models_paper import shallownet, deepnet, eegnet, tsconv, mlpnet


def parse_args():
    parser = argparse.ArgumentParser()
    home=os.environ["HOME"]
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--save_dir",type=str,default=f"{home}/EEG2Video/Gaspard_model/checkpoints/comparison")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["ShallowNetEncoder", "MLPEncoder", "shallownet", "deepnet", "eegnet", "tsconv", "mlpnet"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--input_type", type=str, choices=["raw", "features"], required=True)
    parser.add_argument("--no_early_stop", action="store_true", help="Disable early stopping")
    return parser.parse_args()


class EEGDataset(Dataset):
    def __init__(self, data, labels, blocks):
        self.data = data
        self.labels = labels
        self.blocks = blocks

    def __getitem__(self, idx):
        x = self.data[idx]
        if x.ndim == 2 and x.shape[1] > 1:
            # Normalize per channel (for raw EEG)
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        else:
            # Normalize whole feature vector
            x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.tensor(x, dtype=torch.float32), self.labels[idx], self.blocks[idx]

    def __len__(self):
        return len(self.labels)


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Training encoder on", device, flush=True)
    os.makedirs(args.save_dir, exist_ok=True)

    data = np.load(args.npz_path)
    x = data["eeg"] if args.input_type == "raw" else data["de"] if "de" in data else data["psd"]
    y = data["labels"]
    b = data["blocks"]

    dataset = EEGDataset(x, y, b)

    fold = 4
    val_block = (fold - 1) % 7
    train_idx = [i for i in range(len(dataset)) if dataset.blocks[i] not in [fold, val_block]]
    val_idx = [i for i in range(len(dataset)) if dataset.blocks[i] == val_block]
    test_idx = [i for i in range(len(dataset)) if dataset.blocks[i] == fold]

    dl_train = DataLoader(Subset(dataset, train_idx), batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(Subset(dataset, val_idx), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    dl_test = DataLoader(Subset(dataset, test_idx), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Encoder setup
    C = x.shape[1] if args.input_type == "raw" else 62
    T = x.shape[2] if args.input_type == "raw" else 5
    input_dim = C * T if args.input_type == "features" else None

    if args.model_name == "ShallowNetEncoder":
        encoder = ShallowNetEncoder(C, T).to(device)
        feat_dim = encoder(torch.randn(1, C, T).to(device)).shape[1]
    elif args.model_name == "MLPEncoder":
        encoder = MLPEncoder(input_dim=input_dim).to(device)
        feat_dim = encoder(torch.randn(2, input_dim).to(device)).shape[1]
    elif args.model_name in ["shallownet", "deepnet", "eegnet", "tsconv"]:
        ModelClass = {
            "shallownet": shallownet,
            "deepnet": deepnet,
            "eegnet": eegnet,
            "tsconv": tsconv
        }[args.model_name]

        # Spécial : deepnet / eegnet exigent T ≥ 200
        T_model = max(T, 200) if args.model_name in ["deepnet", "eegnet"] else T
        encoder = ModelClass(out_dim=128, C=C, T=T_model).to(device)

        # ⚠️ Temporarily neutralize output layer to calculate flattened dimension
        if hasattr(encoder, "out"):
            encoder.out = nn.Identity()

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, C, T_model, device=device)
            dummy_output = encoder(dummy_input)
            feat_dim = dummy_output.view(1, -1).shape[1]

        # 🔁 Restore output with correct size
        encoder.out = nn.Linear(feat_dim, 128).to(device)
        feat_dim = 128

        
    elif args.model_name == "mlpnet":
        encoder = mlpnet(out_dim=128, input_dim=input_dim).to(device)
        feat_dim = 128

    classifier = nn.Linear(feat_dim, 40).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        wandb.init(project="eeg2video-model-comparison", name=f"{args.model_name}-fold{fold}", config=vars(args))

    best_val_acc = 0
    patience = 25
    wait = 0

    for epoch in range(1, args.n_epochs + 1):
        encoder.train(); classifier.train()
        total_loss = total_correct = total_samples = 0

        for xb, yb, _ in tqdm(dl_train, desc=f"[Fold {fold}] Epoch {epoch}/{args.n_epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            if args.input_type == "raw" and args.model_name in ["shallownet", "deepnet", "eegnet", "tsconv"]:
                xb = xb.unsqueeze(1)
                if args.model_name in ["deepnet", "eegnet"] and xb.shape[-1] < 200:
                    pad = 200 - xb.shape[-1]
                    xb = torch.nn.functional.pad(xb, (0, pad))
                if args.model_name in ["deepnet", "eegnet"] and xb.shape[-1] < 200:
                    pad = 200 - xb.shape[-1]
                    xb = torch.nn.functional.pad(xb, (0, pad))  # pad on time axis
            elif args.input_type == "features":
                xb = xb.view(xb.size(0), -1)  # (B, 310)

            optimizer.zero_grad()
            feats = encoder(xb)
            logits = classifier(feats)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_samples += yb.size(0)

        train_acc = total_correct / total_samples

        # Validation
        encoder.eval(); classifier.eval(); val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb, _ in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                if args.input_type == "raw" and args.model_name in ["shallownet", "deepnet", "eegnet", "tsconv"]:
                    xb = xb.unsqueeze(1)
                if args.model_name in ["deepnet", "eegnet"] and xb.shape[-1] < 200:
                    pad = 200 - xb.shape[-1]
                    xb = torch.nn.functional.pad(xb, (0, pad))
                elif args.input_type == "features":
                    xb = xb.view(xb.size(0), -1)
                pred = classifier(encoder(xb)).argmax(1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.size(0)
        val_acc = val_correct / val_total

        print(f"[Fold {fold}] Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train Loss: {total_loss / len(dl_train):.4f}")

        if args.use_wandb:
            wandb.log({"train/acc": train_acc, "val/acc": val_acc, "train/loss": total_loss / len(dl_train), "epoch": epoch})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if not args.no_early_stop and wait >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    # Final test
    encoder.eval(); classifier.eval(); test_correct = test_total = 0
    with torch.no_grad():
        for xb, yb, _ in dl_test:
            xb, yb = xb.to(device), yb.to(device)
            if args.input_type == "raw" and args.model_name in ["shallownet", "deepnet", "eegnet", "tsconv"]:
                xb = xb.unsqueeze(1)
                if args.model_name in ["deepnet", "eegnet"] and xb.shape[-1] < 200:
                    pad = 200 - xb.shape[-1]
                    xb = torch.nn.functional.pad(xb, (0, pad))
            elif args.input_type == "features":
                xb = xb.view(xb.size(0), -1)
            pred = classifier(encoder(xb)).argmax(1)
            test_correct += (pred == yb).sum().item()
            test_total += yb.size(0)
    test_acc = test_correct / test_total
    print(f"📊 Fold {fold} | Test Accuracy: {test_acc:.4f}")

    if args.use_wandb:
        wandb.log({"test/acc": test_acc})
        wandb.finish()


if __name__ == "__main__":
    train()
