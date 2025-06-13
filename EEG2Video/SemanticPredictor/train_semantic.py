import os, sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import wandb
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from EEG2Video.SemanticPredictor.models.clip import CLIP


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EEGTextDataset(Dataset):
    def __init__(self, eeg_file, text_dir):
        # Load EEG features for a single subject
        eeg = np.load(eeg_file)
        # Expect shape (blocks, concepts, trials, channels, windows)
        if eeg.ndim == 5:
            b, c, t, ch, w = eeg.shape
            # reshape to (samples, channels, windows)
            eeg_flat = eeg.reshape(b * c * t, ch, w)
        elif eeg.ndim == 3:
            b, ch, w = eeg.shape
            eeg_flat = eeg.reshape(b, ch, w)
        else:
            raise ValueError(f"Unexpected EEG array dims: {eeg.shape}")
        # flatten channels × windows into 310-dimensional vectors
        eeg_vecs = eeg_flat.reshape(eeg_flat.shape[0], -1)  # (samples, 310)

        # Load all CLIP text embeddings (.pt) by block
        text_list = []
        for fname in sorted(os.listdir(text_dir)):
            if fname.endswith('.pt'):
                tensor = torch.load(os.path.join(text_dir, fname))  # (200, seq_len, dim)
                emb = tensor.reshape(tensor.shape[0], -1).cpu().numpy()  # (200, 77*768)
                text_list.append(emb)
        text_all = np.vstack(text_list)  # (blocks*200, 77*768)

        # Align lengths
        n_eeg, n_text = eeg_vecs.shape[0], text_all.shape[0]
        if n_eeg != n_text:
            min_n = min(n_eeg, n_text)
            print(f"Warning: EEG samples={n_eeg} vs TEXT samples={n_text}. Trimming to {min_n}.")
            eeg_vecs = eeg_vecs[:min_n]
            text_all = text_all[:min_n]

        # Standardize EEG features
        self.scaler = StandardScaler().fit(eeg_vecs.astype(np.float32))
        eeg_std = self.scaler.transform(eeg_vecs.astype(np.float32))

        self.eeg = torch.from_numpy(eeg_std)
        self.text = torch.from_numpy(text_all.astype(np.float32))

    def __len__(self):
        return self.eeg.size(0)

    def __getitem__(self, idx):
        return self.eeg[idx], self.text[idx]
def train():
    seed_everything(114514)
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--eeg_file', type=str, default="./data/Preprocessing/DE_1per2s/sub1.npy")
    parser.add_argument('--text_dir', type=str, default="./data/SemanticPredictor/Text_embeddings")
    parser.add_argument('--save_path', type=str, default="./EEG2Video/checkpoints/semantic")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.init(project="eeg2video-semantic", name = 'semantic predictor', config=vars(args))

    dataset = EEGTextDataset(args.eeg_file, args.text_dir)
    # save fitted scaler for inference
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(dataset.scaler, f)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = CLIP().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine annealing scheduler over epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*(len(train_loader)+len(val_loader)))
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for eeg_vec, text_emb in train_loader:
            eeg_vec = eeg_vec.cuda()
            text_emb = text_emb.cuda()
            optimizer.zero_grad()
            pred = model(eeg_vec)
            loss = criterion(pred, text_emb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for eeg_vec, text_emb in val_loader:
                eeg_vec = eeg_vec.cuda()
                text_emb = text_emb.cuda()
                val_loss += criterion(model(eeg_vec), text_emb).item()
        val_loss /= len(val_loader)

        # Logging
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0]
            })
        
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'eeg2text_clip.pt'))
    print(f"Model saved to {args.save_path}/eeg2text_clip.pt")

if __name__ == '__main__':
    train()