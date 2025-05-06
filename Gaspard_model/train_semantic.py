import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import wandb
from transformers import CLIPTokenizer

class SemanticPredictor(nn.Module):
    def __init__(self, input_dim=310, output_dim=77 * 768):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class EEGTextDataset(Dataset):
    def __init__(self, eeg_file, text_file):
        eeg = np.load(eeg_file)  # (7, 40, 5, 62, 5)
        text = np.load(text_file)  # (7, 40, 5, 77, 768)

        eeg = eeg[..., 0, :]  # Take only the first 1-second window → (7, 40, 5, 62, 5) → (7, 40, 5, 62)
        eeg = eeg.reshape(-1, 62, 5).mean(axis=2)  # (1400, 62)
        self.eeg = eeg.reshape(len(eeg), -1)  # (1400, 310)
        self.text = text[..., 0, :, :]  # (7, 40, 5, 77, 768) → (7, 40, 5, 77, 768) if needed, placeholder for proper selection
        self.text = self.text.reshape(len(eeg), 77 * 768)  # (1400, 59016)

        scaler = StandardScaler()
        self.eeg = scaler.fit_transform(self.eeg).astype(np.float32)
        self.text = self.text.astype(np.float32)

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]

def train():
    import argparse
    parser = argparse.ArgumentParser()
    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/EEG2Video"
    parser.add_argument('--eeg_file', type=str, default=f"{root}/data/EEG_embeddings/sub3.npy")
    parser.add_argument('--text_file', type=str, default=f"{root}/data/Text_embeddings/block0_embeddings.npy")
    parser.add_argument('--save_path', type=str, default=f"{root}/Gaspard_model/checkpoints/semantic_predictor.pth")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    dataset = EEGTextDataset(args.eeg_file, args.text_file)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = SemanticPredictor().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if args.use_wandb:
        wandb.init(project="eeg2video-semantic", config=vars(args))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                val_loss += criterion(out, y).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    train()
