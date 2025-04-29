import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from Gaspard_model.models.models import Seq2SeqTransformer

class EEGVideoLatentDataset(Dataset):
    def __init__(self, eeg_dir, latent_dir, split='train'):
        """
        Args:
            eeg_dir: Directory containing EEG .npy files
            latent_dir: Directory containing VAE latent .npy files
            split: 'train' or 'test'
        """
        self.eeg_files = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith('.npy')])
        self.latent_files = sorted([os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith('.npy')])

        # Simple split based on naming (adapt if needed)
        if split == 'train':
            self.eeg_files = self.eeg_files[:-1]
            self.latent_files = self.latent_files[:-1]
        elif split == 'test':
            self.eeg_files = self.eeg_files[-1:]
            self.latent_files = self.latent_files[-1:]
        
        assert len(self.eeg_files) == len(self.latent_files), "Mismatch EEG/latent files!"

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        eeg = np.load(self.eeg_files[idx])  # Shape (7, 512)
        latent = np.load(self.latent_files[idx])  # Shape (6, 256)
        
        eeg = torch.tensor(eeg, dtype=torch.float32)
        latent = torch.tensor(latent, dtype=torch.float32)
        
        return eeg, latent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_dir", type=str, required=True, help="Path to EEG embeddings directory")
    parser.add_argument("--latent_dir", type=str, required=True, help="Path to VAE latents directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/seq2seq", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()

def train_seq2seq(args):
    if args.use_wandb:
        wandb.init(project="eeg2video-seq2seq", config=vars(args))

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and Dataloader
    train_dataset = EEGVideoLatentDataset(args.eeg_dir, args.latent_dir, split='train')
    test_dataset = EEGVideoLatentDataset(args.eeg_dir, args.latent_dir, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = Seq2SeqTransformer().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0

        for eeg, latents in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - Train"):
            eeg, latents = eeg.to(device), latents.to(device)

            optimizer.zero_grad()
            output = model(eeg)  # (batch_size, 6, 256)
            loss = criterion(output, latents)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

        # Test phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for eeg, latents in tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} - Test"):
                eeg, latents = eeg.to(device), latents.to(device)

                output = model(eeg)
                loss = criterion(output, latents)
                running_test_loss += loss.item()

        avg_test_loss = running_test_loss / len(test_loader)
        print(f"Epoch {epoch}: Test Loss = {avg_test_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                "train/loss": avg_train_loss,
                "test/loss": avg_test_loss,
                "epoch": epoch
            })

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"seq2seq_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    args = parse_args()
    train_seq2seq(args)
