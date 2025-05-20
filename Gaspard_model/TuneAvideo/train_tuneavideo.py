import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from torch.cuda.amp import autocast
import wandb
from tqdm import tqdm
import argparse

os.environ["PYTORCH_CUDA_ALLOC_conf"] = "max_split_size_mb:24"

class EEGVideoDataset(Dataset):
    def __init__(self, zhat_dir: str, sem_dir: str):
        z_blocks = []
        for fname in sorted(os.listdir(zhat_dir)):
            if not fname.endswith('.npy'): continue
            z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        self.z_hat = np.concatenate(z_blocks, axis=0)

        e_blocks = []
        for fname in sorted(os.listdir(sem_dir)):
            if not fname.endswith('.npy'): continue
            e_blocks.append(np.load(os.path.join(sem_dir, fname)))
        self.e_t = np.concatenate(e_blocks, axis=0)

        assert self.z_hat.shape[0] == self.e_t.shape[0], \
            f"Got {self.z_hat.shape[0]} ẑ vs {self.e_t.shape[0]} eₜ samples"

    def __len__(self):
        return self.z_hat.shape[0]

    def __getitem__(self, idx):
        z0 = torch.tensor(self.z_hat[idx]).float()
        et = torch.tensor(self.e_t[idx]).float()
        return z0, et

class TuneAVideoTrainer:
    def __init__(self, args):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data
        dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        self.val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

        # pretrained components
        root = args.root
        self.vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            f"{root}/EEG2Video/EEG2Video_New/Generation/stable-diffusion-v1-4",
            subfolder='unet'
        ).to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

        # --- freeze logic ---
        # Freeze entire VAE
        self.vae.requires_grad_(False)
        # Freeze entire UNet, then unfreeze only the designated attention sub-modules
        self.unet.requires_grad_(False)
        trainable_modules = ("attn1.to_q", "attn2.to_q", "attn_temp")
        for name, module in self.unet.named_modules():
            if any(name.endswith(m) for m in trainable_modules):
                for p in module.parameters():
                    p.requires_grad = True
        # ---------------------

        # pipeline
        self.pipeline = TuneAVideoPipeline(
            vae=self.vae, tokenizer=self.tokenizer,
            unet=self.unet, scheduler=self.scheduler
        )
        self.pipeline.unet.train()

        # projection
        sem_dim   = dataset[0][1].shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_eeg = nn.Linear(sem_dim, cross_dim).to(self.device)

        # optimizer & scheduler
        # Only parameters with requires_grad=True (i.e. the three UNet submodules + proj_eeg) are optimized
        self.optimizer = torch.optim.Adam(
            list(self.pipeline.unet.parameters()) + list(self.proj_eeg.parameters()),
            lr=args.lr
        )
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # wandb
        if args.use_wandb:
            wandb.init(project="eeg2video-tuneavideo", config=vars(args))

        self.args = args

    def _train_epoch(self, epoch):
        self.pipeline.unet.train()
        total_loss = 0.0
        for z0, et in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} - train"):
            z0 = z0.to(self.device).permute(0,2,1,3,4)
            et = et.to(self.device).unsqueeze(1)
            et = self.proj_eeg(et)

            noise = torch.randn_like(z0)
            timesteps = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
            z_t = self.scheduler.add_noise(z0, noise, timesteps)

            self.optimizer.zero_grad()
            out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et)
            loss = F.mse_loss(out.sample, noise)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            #torch.cuda.empty_cache()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.pipeline.unet.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast():
            for z0, et in tqdm(self.val_loader, desc="Validation"):
                z0 = z0.to(self.device).permute(0,2,1,3,4)
                et_proj = self.proj_eeg(et.to(self.device).unsqueeze(1))
                noise = torch.randn_like(z0)
                timesteps = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
                z_t = self.scheduler.add_noise(z0, noise, timesteps)
                try:
                    out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et_proj)
                except RuntimeError as e:
                    if 'CUBLAS_STATUS_EXECUTION_FAILED' in str(e):
                        out = self.pipeline.unet(z_t.float(), timesteps, encoder_hidden_states=et_proj.float())
                    else:
                        raise
                val_loss += F.mse_loss(out.sample, noise).item()
        return val_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch):
        ckpt_dir = os.path.join(self.args.root, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f'tuneavideo_unet_epoch{epoch}.pt')
        self.pipeline.unet.save_pretrained(path)
        print(f"Saved checkpoint at {path}")

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch()
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            #torch.cuda.empty_cache()
            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Tune-A-Video with EEG')
    root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
    parser.add_argument('--zhat_dir',   type=str,   default=f"{root}/data/Predicted_latents")
    parser.add_argument('--sem_dir',    type=str,   default=f"{root}/data/Semantic_embeddings")
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--use_wandb',  action='store_true')
    parser.add_argument('--save_every', type=int,   default=10)
    parser.add_argument('--root',       type=str,   default=root)
    return parser.parse_args()

if __name__ == '__main__':
    trainer = TuneAVideoTrainer(parse_args())
    trainer.train()
