#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import AutoencoderKL, PNDMScheduler
from models.unet import UNet3DConditionModel
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
import wandb

class EEGVideoDataset(Dataset):
    def __init__(self, zhat_dir: str, sem_dir: str):
        z_blocks = []
        for fname in sorted(os.listdir(zhat_dir)):
            if fname.endswith('.npy'):
                z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        self.z_hat = np.concatenate(z_blocks, axis=0)  # (N, F, C, H, W)

        e_blocks = []
        for fname in sorted(os.listdir(sem_dir)):
            if fname.endswith('.npy'):
                e_blocks.append(np.load(os.path.join(sem_dir, fname)))
        self.e_t = np.concatenate(e_blocks, axis=0)     # (N, L, D)

        assert self.z_hat.shape[0] == self.e_t.shape[0], \
            f"Mismatched samples: {self.z_hat.shape[0]} vs {self.e_t.shape[0]}"

    def __len__(self):
        return self.z_hat.shape[0]

    def __getitem__(self, idx):
        z0 = torch.tensor(self.z_hat[idx]).float()  # (F, C, H, W)
        et = torch.tensor(self.e_t[idx]).float()     # (L, D)
        return z0, et

class VideoDiffusionTrainer:
    def __init__(self, args):
        # 1) Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) Data
        ds = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        n_train = int(0.8 * len(ds))
        n_val   = len(ds) - n_train
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

        # 3) Pretrained VAE & UNet
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae"
        ).to(self.device)
        
        root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            f"{root}/EEG2Video/EEG2Video_New/Generation/stable-diffusion-v1-4",
            subfolder='unet'
        ).to(self.device)

        # 4) Scheduler
        self.scheduler = PNDMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # 5) Semantic projection
        sem_dim   = ds.e_t.shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_sem = nn.Linear(sem_dim, cross_dim).to(self.device)

        # 6) Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.unet.parameters()) + list(self.proj_sem.parameters()),
            lr=args.lr
        )

        # 7) WandB
        if args.use_wandb:
            wandb.init(project="eeg2video-diffusion", config=vars(args))

        self.args = args

    def train_epoch(self, epoch):
        self.unet.train()
        running_loss = 0.0
        for z0, sem in tqdm(self.train_loader, desc=f"Epoch {epoch} – train"):
            # Move & reshape
            z0 = z0.to(self.device).permute(0,2,1,3,4)   # (B,C,F,H,W)
            sem = sem.to(self.device)                    # (B,L,D)
            sem_proj = self.proj_sem(sem)                # (B,L,cross_dim)
            sem_proj = sem_proj.unsqueeze(1)   
            
            # Noise & timesteps
            noise = torch.randn_like(z0)
            t = torch.randint(0, self.scheduler.num_train_timesteps, (z0.size(0),), device=self.device)
            z_t = self.scheduler.add_noise(z0, noise, t)

            # Forward + loss
            self.optimizer.zero_grad()
            with autocast():
                pred = self.unet(z_t, t, encoder_hidden_states=sem_proj).sample
                loss = F.mse_loss(pred, noise)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            torch.cuda.empty_cache()

        return running_loss / len(self.train_loader)

    def val_epoch(self):
        self.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z0, sem in tqdm(self.val_loader, desc="Validation"):
                z0 = z0.to(self.device).permute(0,2,1,3,4)
                sem_proj = self.proj_sem(sem.to(self.device))
                noise = torch.randn_like(z0)
                t = torch.randint(0, self.scheduler.num_train_timesteps, (z0.size(0),), device=self.device)
                z_t = self.scheduler.add_noise(z0, noise, t)
                pred = self.unet(z_t, t, encoder_hidden_states=sem_proj).sample
                val_loss += F.mse_loss(pred, noise).item()
        return val_loss / len(self.val_loader)

    def save_checkpoint(self, epoch):
        out = os.path.join(self.args.root, "checkpoints")
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"unet_epoch{epoch}.pt")
        self.unet.save_pretrained(path)
        print(f"Checkpoint → {path}")

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            tr_loss = self.train_epoch(epoch)
            vl_loss = self.val_epoch()
            print(f"Epoch {epoch}: train={tr_loss:.4f}, val={vl_loss:.4f}")
            torch.cuda.empty_cache()
            if epoch % self.args.save_every == 0:
                self.save_checkpoint(epoch)

def parse_args():
    parser = argparse.ArgumentParser(description="3D Video Diffusion Training")
    root = os.path.join(os.environ.get("HOME", ""), "EEG2Video")
    parser.add_argument("--zhat_dir",   type=str, default=os.path.join(root, "data/Predicted_latents"))
    parser.add_argument("--sem_dir",    type=str, default=os.path.join(root, "data/Semantic_embeddings"))
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--root",       type=str, default=root)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    trainer = VideoDiffusionTrainer(args)
    trainer.train()
