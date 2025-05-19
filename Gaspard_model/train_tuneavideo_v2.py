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
            if not fname.endswith('.npy'):
                continue
            z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        self.z_hat = np.concatenate(z_blocks, axis=0)

        e_blocks = []
        for fname in sorted(os.listdir(sem_dir)):
            if not fname.endswith('.npy'):
                continue
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

        # CUDNN tuning
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # data
        dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        dl_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': args.pin_memory
        }
        self.train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
        self.val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

        # pretrained components
        root = args.root
        self.vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            f"{root}/EEG2Video/EEG2Video_New/Generation/stable-diffusion-v1-4",
            subfolder='unet'
        ).to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

        # freeze logic
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        trainable_modules = ("attn1.to_q", "attn2.to_q", "attn_temp")
        for name, module in self.unet.named_modules():
            if any(name.endswith(m) for m in trainable_modules):
                for p in module.parameters():
                    p.requires_grad = True

        # optional performance tweaks
        if args.use_xformers:
            self.unet.enable_xformers_memory_efficient_attention()
        if args.use_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

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
        opt_params = list(self.pipeline.unet.parameters()) + list(self.proj_eeg.parameters())
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.AdamW8bit(opt_params, lr=args.lr)
            except ImportError:
                raise ImportError("bitsandbytes is required for 8-bit Adam. Install via `pip install bitsandbytes`.")
        else:
            self.optimizer = torch.optim.Adam(opt_params, lr=args.lr)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # mixed precision
        self.use_amp = args.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # wandb
        if args.use_wandb:
            wandb.init(project="eeg2video-tuneavideo", config=vars(args))

        self.args = args

    def _train_epoch(self, epoch):
        self.pipeline.unet.train()
        total_loss = 0.0
        acc_steps = self.args.gradient_accumulation_steps
        for i, (z0, et) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} - train")):
            z0 = z0.to(self.device).permute(0,2,1,3,4)
            et = et.to(self.device).unsqueeze(1)
            et = self.proj_eeg(et)

            noise = torch.randn_like(z0)
            timesteps = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
            z_t = self.scheduler.add_noise(z0, noise, timesteps)

            if self.use_amp:
                with autocast():
                    out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et)
                    loss = F.mse_loss(out.sample, noise) / acc_steps
                self.scaler.scale(loss).backward()
            else:
                out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et)
                loss = F.mse_loss(out.sample, noise) / acc_steps
                loss.backward()

            if (i + 1) % acc_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.args.use_set_to_none:
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.optimizer.zero_grad()

            total_loss += loss.item() * acc_steps
            torch.cuda.empty_cache()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.pipeline.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z0, et in tqdm(self.val_loader, desc="Validation"):
                z0 = z0.to(self.device).permute(0,2,1,3,4)
                et_proj = self.proj_eeg(et.to(self.device).unsqueeze(1))
                noise = torch.randn_like(z0)
                timesteps = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
                z_t = self.scheduler.add_noise(z0, noise, timesteps)
                if self.use_amp:
                    with autocast():
                        out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et_proj)
                else:
                    out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et_proj)
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
            torch.cuda.empty_cache()
            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tune-A-Video with EEG (with optional optimizations)')
    root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
    parser.add_argument('--zhat_dir',   type=str,   default=f"{root}/data/Predicted_latents")
    parser.add_argument('--sem_dir',    type=str,   default=f"{root}/data/Semantic_embeddings")
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--use_wandb',  action='store_true')
    parser.add_argument('--save_every', type=int,   default=10)
    parser.add_argument('--root',       type=str,   default=root)

    # Optional performance flags
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--use_8bit_adam', action='store_true', help='Use 8-bit Adam optimizer (bitsandbytes)')
    parser.add_argument('--use_xformers', action='store_true', help='Enable xFormers memory-efficient attention')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help='Enable gradient checkpointing in UNet')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Steps for gradient accumulation')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--pin_memory', action='store_true', help='Use pin_memory for DataLoader')
    parser.add_argument('--use_set_to_none', action='store_true', help='Zero grad with set_to_none')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='Enable torch.backends.cudnn.benchmark')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainer = TuneAVideoTrainer(args)
    trainer.train()
