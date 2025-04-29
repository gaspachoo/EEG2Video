import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from diffusers.models.vae import AutoencoderKL
import torchvision.transforms as T
import wandb
from tqdm import tqdm
import imageio
import argparse
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/Video_Gif", help="Path to GIF dataset.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/vae/", help="Directory to save models.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()

class EEGGIFDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        for block in sorted(os.listdir(root_dir)):
            block_path = os.path.join(root_dir, block)
            if not os.path.isdir(block_path):
                continue
            for gif in sorted(os.listdir(block_path)):
                self.samples.append(os.path.join(block_path, gif))
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gif_path = self.samples[idx]
        frames = imageio.mimread(gif_path)
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames

def compute_loss(vae, frames, beta=0.25):
    posterior = vae.encode(frames).latent_dist
    z = posterior.sample()
    reconstructions = vae.decode(z).sample
    recon_loss = nn.functional.mse_loss(reconstructions, frames, reduction="mean")
    kl_loss = posterior.kl().mean()
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss

def train_vae(args):
    if args.use_wandb:
        wandb.init(project="eeg2video-vae", config=vars(args))

    dataset = EEGGIFDataset(args.data_dir)
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    beta = 0.25

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Training phase
        vae.train()
        running_train_loss = 0.0
        running_train_recon_loss = 0.0
        running_train_kl_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - Train"):
            batch = batch.to(device)
            frames = batch.view(-1, 3, 288, 512)

            optimizer.zero_grad()
            loss, recon_loss, kl_loss = compute_loss(vae, frames, beta)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_recon_loss += recon_loss.item()
            running_train_kl_loss += kl_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_recon_loss = running_train_recon_loss / len(train_loader)
        avg_train_kl_loss = running_train_kl_loss / len(train_loader)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

        # Validation phase
        vae.eval()
        running_val_loss = 0.0
        running_val_recon_loss = 0.0
        running_val_kl_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} - Validation"):
                batch = batch.to(device)
                frames = batch.view(-1, 3, 288, 512)
                loss, recon_loss, kl_loss = compute_loss(vae, frames, beta)

                running_val_loss += loss.item()
                running_val_recon_loss += recon_loss.item()
                running_val_kl_loss += kl_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_recon_loss = running_val_recon_loss / len(val_loader)
        avg_val_kl_loss = running_val_kl_loss / len(val_loader)

        print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                "train/loss": avg_train_loss,
                "train/recon_loss": avg_train_recon_loss,
                "train/kl_loss": avg_train_kl_loss,
                "val/loss": avg_val_loss,
                "val/recon_loss": avg_val_recon_loss,
                "val/kl_loss": avg_val_kl_loss,
                "epoch": epoch
            })

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"vae_epoch{epoch}.pth")
            torch.save(vae.state_dict(), ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")

    # Final testing phase
    vae.eval()
    running_test_loss = 0.0
    running_test_recon_loss = 0.0
    running_test_kl_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            frames = batch.view(-1, 3, 288, 512)
            loss, recon_loss, kl_loss = compute_loss(vae, frames, beta)

            running_test_loss += loss.item()
            running_test_recon_loss += recon_loss.item()
            running_test_kl_loss += kl_loss.item()

    avg_test_loss = running_test_loss / len(test_loader)
    avg_test_recon_loss = running_test_recon_loss / len(test_loader)
    avg_test_kl_loss = running_test_kl_loss / len(test_loader)

    print(f"✅ Final Test Loss = {avg_test_loss:.4f}")

    if args.use_wandb:
        wandb.log({
            "test/loss": avg_test_loss,
            "test/recon_loss": avg_test_recon_loss,
            "test/kl_loss": avg_test_kl_loss,
            "epoch": args.epochs + 1  # différencier sur wandb
        })

if __name__ == "__main__":
    train_vae(parse_args())
