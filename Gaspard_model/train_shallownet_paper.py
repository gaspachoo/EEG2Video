import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os
import time
import argparse
from models.encoders import ShallowNetEncoder
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str,default=os.path.join(os.environ["HOME"], "EEG2Video/dataset/EEG_500ms_sw/sub1_segmented.npz"),help="Path to .npz file")
    parser.add_argument("--save_dir", type=str,default=os.path.join(os.environ["HOME"], "EEG2Video/Gaspard_model/checkpoints/cv_shallownet"),help="Directory to save checkpoints")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=20, help="Epoch frequency for saving checkpoints")
    parser.add_argument("--no_early_stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--use_wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the encoder")
    return parser.parse_args()

class EEGBlockDataset(Dataset):
    def __init__(self, eeg, labels, blocks):
        self.eeg = eeg
        self.labels = labels
        self.blocks = blocks

    def __getitem__(self, idx):
        x = self.eeg[idx]
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        return x, self.labels[idx], self.blocks[idx]

    def __len__(self):
        return len(self.labels)

def train_encoder(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[GLOBAL] Training encoder on", device, flush=True)

    data = np.load(args.npz_path)
    eeg = torch.tensor(data["eeg"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    blocks = torch.tensor(data["blocks"], dtype=torch.long)

    full_dataset = EEGBlockDataset(eeg, labels, blocks)

    for fold in range(7):
        print(f"\n🔁 Fold {fold}")

        if args.use_wandb:
            # Generate or load a unique run_id per fold
            run_id_path = os.path.join(args.save_dir, f"wandb_runid_fold{fold}.txt")
            if os.path.exists(run_id_path):
                with open(run_id_path, "r") as f:
                    run_id = f.read().strip()
                resume = "must"
                print(f"🔁 Resume wandb run {run_id}")
            else:
                run_id = wandb.util.generate_id()
                with open(run_id_path, "w") as f:
                    f.write(run_id)
                resume = None
                print(f"🆕 New wandb run created : {run_id}")

            wandb.init(
                project="eeg2video-shallownet-v1",
                name=f"cv_global_shallow_fold{fold}",
                config=vars(args),
                id=run_id,
                resume=resume
            )


        val_block = (fold - 1) % 7
        train_indices = [i for i in range(len(full_dataset)) if full_dataset.blocks[i] not in [fold, val_block]]
        val_indices = [i for i in range(len(full_dataset)) if full_dataset.blocks[i] == val_block]
        test_indices = [i for i in range(len(full_dataset)) if full_dataset.blocks[i] == fold]

        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=256,
            shuffle=True,
            num_workers=6,
            pin_memory=True
            )

        val_loader = DataLoader(
            Subset(full_dataset, val_indices),
            batch_size=256,
            shuffle=True,
            num_workers=6,
            pin_memory=True
            )

        segment_len = eeg.shape[-1]  # typiquement 100 pour 0.5s, 200 pour 1s
        encoder = ShallowNetEncoder(62, segment_len).to(device)

        with torch.no_grad():
            dummy_input = torch.randn(1, 62, segment_len).to(device)
            feat_dim = encoder(dummy_input).shape[1]

        classifier = nn.Linear(feat_dim, 40).to(device)

        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience, wait = 70, 0

        checkpoint_path = os.path.join(args.save_dir, f"best_fold{fold}.pth")
        start_epoch = 1

        if os.path.exists(checkpoint_path):
            print(f"🔁 Resume from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(checkpoint["encoder"])
            classifier.load_state_dict(checkpoint["classifier"])
            start_epoch = checkpoint.get("epoch", 1) + 1  # Reprendre à l'epoch suivante

        
        for epoch in range(start_epoch, args.n_epochs + 1):
            encoder.train(); classifier.train()
            total_loss = total_correct = total_samples = 0

            for xb, yb, _ in tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch}/{args.n_epochs}"):
                xb, yb = xb.to(device), yb.to(device)

                scaler = torch.cuda.amp.GradScaler()
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    feats = encoder(xb)
                    logits = classifier(feats)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


                total_loss += loss.item()
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_samples += yb.size(0)
                
            train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_samples

            encoder.eval(); classifier.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for xb, yb, _ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    feats = encoder(xb)
                    logits = classifier(feats)
                    loss = criterion(logits, yb)

                    val_loss += loss.item()
                    val_correct += (logits.argmax(1) == yb).sum().item()
                    val_total += yb.size(0)

            val_acc = val_correct / val_total
            val_loss /= len(val_loader)

            print(f"[Fold {fold}] Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if args.use_wandb:
                wandb.log({
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "epoch": epoch
                })

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                wait = 0
                torch.save({
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch
                }, os.path.join(args.save_dir, f"best_fold{fold}.pth"))
            else:
                wait += 1
                if not args.no_early_stop and wait >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch} for fold {fold}")
                    break

            if epoch % args.save_every == 0:
                torch.save({
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch
                }, os.path.join(args.save_dir, f"fold{fold}_epoch{epoch}.pth"))

        if args.use_wandb:
            wandb.finish()



if __name__ == "__main__":
    args = parse_args()
    train_encoder(args)
