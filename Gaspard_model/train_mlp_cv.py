import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import wandb
from Gaspard_model.models.models import MLPEncoder  # Doit utiliser la version avec BatchNorm et Dropout amélioré

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default=os.path.join(os.environ["HOME"], "EEG2Video/data/DE_500ms_sw/sub1_features.npz"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.environ["HOME"], "EEG2Video/Gaspard_model/checkpoints/cv_mlp_DE"))
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--empty_save_dir", action="store_true", help="Clear save directory before training")

    return parser.parse_args()

class EEGFeatureDataset(Dataset):
    def __init__(self, features, labels, blocks):
        self.features = features
        self.labels = labels
        self.blocks = blocks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x, self.labels[idx], self.blocks[idx]

def train_mlp_cv(args):
    os.makedirs(args.save_dir, exist_ok=True)
    if getattr(args, "empty_save_dir", False):
        for f in os.listdir(args.save_dir):
            os.remove(os.path.join(args.save_dir, f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GLOBAL] Training on {device}", flush=True)

    data = np.load(args.npz_path)
    features = torch.tensor(data["de"], dtype=torch.float32) if "DE" in args.npz_path else torch.tensor(data["psd"], dtype=torch.float32) # ou "psd"
    labels = torch.tensor(data["labels"], dtype=torch.long)
    blocks = torch.tensor(data["blocks"], dtype=torch.long)

    dataset = EEGFeatureDataset(features, labels, blocks)
    input_dim = features.shape[1] * features.shape[2]

    for fold in range(7):
        print(f"\n🔁 Fold {fold}")

        if args.use_wandb:
            run_id_path = os.path.join(args.save_dir, f"wandb_runid_fold{fold}.txt")
            if os.path.exists(run_id_path):
                with open(run_id_path, "r") as f:
                    run_id = f.read().strip()
                resume = "must"
            else:
                run_id = wandb.util.generate_id()
                with open(run_id_path, "w") as f:
                    f.write(run_id)
                resume = None
            wandb.init(project="eeg2video-mlp-DE-v1", name=f"cv_local_mlp_fold{fold}", config=vars(args), id=run_id, resume=resume)

        val_block = (fold - 1) % 7
        train_idx = [i for i in range(len(dataset)) if dataset.blocks[i] not in [fold, val_block]]
        val_idx = [i for i in range(len(dataset)) if dataset.blocks[i] == val_block]
        test_idx = [i for i in range(len(dataset)) if dataset.blocks[i] == fold]

        dl_train = DataLoader(Subset(dataset, train_idx), batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
        dl_val = DataLoader(Subset(dataset, val_idx), batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        dl_test = DataLoader(Subset(dataset, test_idx), batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

        model = MLPEncoder(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        wait = 0
        patience = 25
        start_epoch = 1

        for epoch in range(start_epoch, args.n_epochs + 1):
            model.train()
            total_loss = total_correct = total_samples = 0

            for xb, yb, _ in tqdm(dl_train, desc=f"[Fold {fold}] Epoch {epoch}/{args.n_epochs}"):
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.view(xb.size(0), -1)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_samples += yb.size(0)

            train_loss = total_loss / len(dl_train)
            train_acc = total_correct / total_samples

            # Validation
            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for xb, yb, _ in dl_val:
                    xb, yb = xb.to(device), yb.to(device)
                    xb = xb.view(xb.size(0), -1)
                    logits = model(xb)
                    loss = criterion(logits, yb)

                    val_loss += loss.item()
                    val_correct += (logits.argmax(1) == yb).sum().item()
                    val_total += yb.size(0)

            val_loss /= len(dl_val)
            val_acc = val_correct / val_total

            print(f"[Fold {fold}] Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if args.use_wandb:
                wandb.log({"train/loss": train_loss, "train/acc": train_acc, "val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                wait = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_fold{fold}.pt"))
            else:
                wait += 1
                if not args.no_early_stop and wait >= patience:
                    print(f"⏹️ Early stopping triggered at epoch {epoch}")
                    break

            if epoch % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"fold{fold}_epoch{epoch}.pt"))

        # Final test accuracy
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f"best_fold{fold}.pt")))
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for xb, yb, _ in dl_test:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.view(xb.size(0), -1)
                preds = model(xb).argmax(1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(yb.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        print(f"📊 Fold {fold} | Test Accuracy: {test_acc:.4f}")
        if args.use_wandb:
            wandb.log({"test/acc": test_acc})
            wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train_mlp_cv(args)
