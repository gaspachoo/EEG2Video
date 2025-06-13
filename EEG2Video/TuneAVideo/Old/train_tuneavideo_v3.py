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
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import argparse

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
            f"Mismatch: {self.z_hat.shape[0]} ẑ vs {self.e_t.shape[0]} eₜ samples"

    def __len__(self): return self.z_hat.shape[0]

    def __getitem__(self, idx):
        z0 = torch.tensor(self.z_hat[idx]).float()
        et = torch.tensor(self.e_t[idx]).float()
        return z0, et

class TuneAVideoTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CUDNN tuning
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # DataLoader settings
        dl_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        if args.prefetch_factor is not None and args.num_workers > 0:
            dl_kwargs['prefetch_factor'] = args.prefetch_factor

        dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        train_size = int(0.8 * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
        self.train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
        self.val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

        # Pretrained components
        self.vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            "./EEG2Video/EEG2Video_New/Generation/stable-diffusion-v1-4",
            subfolder='unet'
        ).to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

        # Optional: channels_last layout
        if args.use_channels_last:
            self.unet.to(memory_format=torch.channels_last)

        # Freeze VAE and UNet except key attention parts
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        trainable = ("attn1.to_q", "attn2.to_q", "attn_temp")
        for name, module in self.unet.named_modules():
            if any(name.endswith(m) for m in trainable):
                for p in module.parameters(): p.requires_grad = True

        # Optional xFormers and gradient checkpointing
        if args.use_xformers:
            self.unet.enable_xformers_memory_efficient_attention()
        if args.use_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # Pipeline setup
        self.pipeline = TuneAVideoPipeline(
            vae=self.vae, tokenizer=self.tokenizer,
            unet=self.unet, scheduler=self.scheduler
        )
        self.pipeline.unet.train()

        # Projection layer
        sem_dim = dataset[0][1].shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_eeg = nn.Linear(sem_dim, cross_dim).to(self.device)

        # Optimizer
        params = list(self.pipeline.unet.parameters()) + list(self.proj_eeg.parameters())
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            self.optimizer = bnb.AdamW8bit(params, lr=args.lr)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=args.lr)

        # Mixed precision
        self.use_amp = args.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # Optional: Torch compile
        if args.use_torch_compile:
            self.pipeline.unet = torch.compile(self.pipeline.unet)
            self.proj_eeg = torch.compile(self.proj_eeg)

        # Scheduler timesteps
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # W&B
        if args.use_wandb:
            wandb.init(project="eeg2video-tuneavideo", config=vars(args))

    def _step(self, z0, et):
        # Move and format tensors
        z0 = z0.to(self.device)
        if self.args.use_channels_last:
            z0 = z0.to(memory_format=torch.channels_last)
        z0 = z0.permute(0,2,1,3,4)
        et = et.to(self.device).unsqueeze(1)
        et = self.proj_eeg(et)

        noise = torch.randn_like(z0)
        timesteps = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
        z_t = self.scheduler.add_noise(z0, noise, timesteps)
        return z_t, noise, timesteps

    def _train_epoch(self, epoch):
        self.pipeline.unet.train()
        total_loss = 0.0
        acc_steps = self.args.gradient_accumulation_steps
        for i, (z0, et) in enumerate(tqdm(self.train_loader, desc=f"Train {epoch}/{self.args.epochs}")):
            z_t, noise, timesteps = self._step(z0, et)
            print(">>> z_t.shape:", z_t.shape)  # doit être [B, C, T, H, W]
            print(">>> timesteps.max():", timesteps.max().item(), "of", len(self.scheduler.timesteps))

            # Forward/backward
            if self.use_amp:
                with autocast(): out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=self.proj_eeg(et.to(self.device).unsqueeze(1)))
                loss = F.mse_loss(out.sample, noise) / acc_steps
                self.scaler.scale(loss).backward()
            else:
                out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=self.proj_eeg(et.to(self.device).unsqueeze(1)))
                loss = F.mse_loss(out.sample, noise) / acc_steps
                loss.backward()

            # Step
            if (i+1) % acc_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True if self.args.use_set_to_none else False)

            total_loss += loss.item() * acc_steps
            if self.args.use_empty_cache: torch.cuda.empty_cache()
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.pipeline.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z0, et in tqdm(self.val_loader, desc="Validate"):
                z_t, noise, timesteps = self._step(z0, et)
                if self.use_amp:
                    with autocast(): out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=self.proj_eeg(et.to(self.device).unsqueeze(1)))
                else:
                    out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=self.proj_eeg(et.to(self.device).unsqueeze(1)))
                val_loss += F.mse_loss(out.sample, noise).item()
        return val_loss / len(self.val_loader)

    def _save(self, epoch):
        ckpt_dir = ("EEG2Video/checkpoints/TuneAVideo")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.pipeline.unet.save_pretrained(os.path.join(ckpt_dir, f'unet_ep{epoch}.pt'))

    def train(self):
        for epoch in range(1, self.args.epochs+1):
            tl = self._train_epoch(epoch)
            vl = self._validate_epoch()
            print(f"Epoch {epoch}: train={tl:.4f}, val={vl:.4f}")
            if epoch % self.args.save_every == 0: self._save(epoch)


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument('--zhat_dir', type=str, default="./data/Predicted_latents")
    p.add_argument('--sem_dir', type=str, default="./data/Semantic_embeddings")
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--use_wandb', action='store_true')

    # Performance flags
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--use_8bit_adam', action='store_true')
    p.add_argument('--use_xformers', action='store_true')
    p.add_argument('--use_gradient_checkpointing', action='store_true')
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--use_set_to_none', action='store_true')
    p.add_argument('--cudnn_benchmark', action='store_true')
    p.add_argument('--prefetch_factor', type=int, default=None)
    p.add_argument('--use_empty_cache', action='store_true')
    p.add_argument('--use_channels_last', action='store_true')
    p.add_argument('--use_torch_compile', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainer = TuneAVideoTrainer(args)
    trainer.train()
