import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from tqdm import tqdm
import pynvml
import wandb
from fvcore.nn import FlopCountAnalysis

########
# Multiparameter training script for TuneAVideo with DDP
########

# ---------------------------
# Dataset
# ---------------------------
class EEGVideoDataset(Dataset):
    """Simple npy‑folder dataset holding zhat and semantic embeddings."""

    def __init__(self, zhat_dir: str, sem_dir: str):
        z_blocks, e_blocks = [], []
        for fname in sorted(os.listdir(zhat_dir)):
            if fname.endswith('.npy'):
                z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        for fname in sorted(os.listdir(sem_dir)):
            if fname.endswith('.npy'):
                e_blocks.append(np.load(os.path.join(sem_dir, fname)))

        self.z_hat = np.concatenate(z_blocks, axis=0)
        self.e_t = np.concatenate(e_blocks, axis=0)
        assert self.z_hat.shape[0] == self.e_t.shape[0], (
            f"Mismatch z_hat {self.z_hat.shape} vs e_t {self.e_t.shape}")

    def __len__(self):
        return self.z_hat.shape[0]

    def __getitem__(self, idx):
        z0 = torch.from_numpy(self.z_hat[idx]).float()
        et = torch.from_numpy(self.e_t[idx]).float()
        return z0, et


# ---------------------------
# Trainer
# ---------------------------
class TuneAVideoTrainerDDP:
    def __init__(self, args, rank: int, local_rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        # ---------- Data ----------
        ds = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        n_train = int(0.8 * len(ds))
        n_val = len(ds) - n_train
        train_ds, val_ds = random_split(ds, [n_train, n_val])

        self.train_sampler = DistributedSampler(train_ds, world_size, rank)
        self.val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False)

        dl_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        if args.prefetch_factor is not None and args.num_workers > 0:
            dl_kwargs['prefetch_factor'] = args.prefetch_factor

        self.train_loader = DataLoader(train_ds, sampler=self.train_sampler, **dl_kwargs)
        self.val_loader = DataLoader(val_ds, sampler=self.val_sampler, **dl_kwargs)

        # ---------- Models ----------
        self.vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to('cpu').eval()
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            './stable-diffusion-v1-4', subfolder='unet').to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')

        # freeze
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        trainable = ("attn1.to_q", "attn2.to_q", "attn_temp")
        for n, m in self.unet.named_modules():
            if any(n.endswith(t) for t in trainable):
                for p in m.parameters():
                    p.requires_grad_(True)

        # pipeline
        self.pipeline = TuneAVideoPipeline(vae=self.vae, tokenizer=self.tokenizer, unet=self.unet, scheduler=self.scheduler)
        self.pipeline.unet.train()

        sem_dim = ds[0][1].shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_eeg = nn.Linear(sem_dim, cross_dim).to(self.device)

        # wrap in DDP
        self.pipeline.unet = DDP(self.pipeline.unet, device_ids=[local_rank], output_device=local_rank,
                                 gradient_as_bucket_view=True)
        self.proj_eeg = DDP(self.proj_eeg, device_ids=[local_rank], output_device=local_rank)

        params = list(self.pipeline.unet.parameters()) + list(self.proj_eeg.parameters())
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            self.optimizer = bnb.AdamW8bit(params, lr=args.lr)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=args.lr)

        self.use_amp = args.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None
        self.accum_steps = args.gradient_accumulation_steps
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # perf flags
        if args.use_channels_last:
            self.pipeline.unet.to(memory_format=torch.channels_last)

        if args.use_xformers:
            self.pipeline.unet.module.enable_xformers_memory_efficient_attention()
        if args.use_gradient_checkpointing:
            self.pipeline.unet.module.enable_gradient_checkpointing()
        if args.use_torch_compile:
            self.pipeline.unet = torch.compile(self.pipeline.unet)
            self.proj_eeg = torch.compile(self.proj_eeg)

        # nvml for gpu metrics
        pynvml.nvmlInit()
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

        # wandb
        if args.use_wandb and rank == 0:
            wandb.init(project='eeg2video-tuneavideo-new', config=vars(args))
            wandb.define_metric('gpu/used_GB')
            wandb.define_metric('gpu/clock_MHz')
            wandb.define_metric('model/flops')

    # ----------------------- utility -----------------------
    def _format_batch(self, z0, et):
        z0 = z0.to(self.device).permute(0, 2, 1, 3, 4).contiguous()
        et = et.to(self.device).unsqueeze(1)
        eh = self.proj_eeg(et).contiguous()
        return z0, eh

    def _step_ddpm_inputs(self, z0):
        noise = torch.randn_like(z0)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (z0.size(0),), device=self.device).long()
        z_t = self.scheduler.add_noise(z0, noise, timesteps)
        return z_t, noise, timesteps

    # ----------------------- train / val -----------------------
    def _train_epoch(self, epoch):
        self.pipeline.unet.train()
        self.train_sampler.set_epoch(epoch)
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=self.args.use_set_to_none)

        for i, (z0, et) in enumerate(tqdm(self.train_loader, desc=f"Rank {self.rank} | Train {epoch}", position = self.rank, leave=True)):
            z0, eh = self._format_batch(z0, et)
            z_t, noise, timesteps = self._step_ddpm_inputs(z0)

            with autocast(enabled=self.use_amp):
                out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=eh)
                loss = F.mse_loss(out.sample, noise) / self.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=self.args.use_set_to_none)

            total_loss += loss.item() * self.accum_steps
        if self.args.use_empty_cache:
                torch.cuda.empty_cache()
        return total_loss / len(self.train_loader)
    

    def _validate_epoch(self):
        self.pipeline.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z0, et in tqdm(self.val_loader, desc=f"Rank {self.rank} | Val", position = self.rank, leave=True):
                z0, eh = self._format_batch(z0, et)
                z_t, noise, timesteps = self._step_ddpm_inputs(z0)
                with autocast(enabled=self.use_amp):
                    out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=eh)
                    loss = F.mse_loss(out.sample, noise)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    # ----------------------- save -----------------------
    def _save_ckpt(self, epoch):
        if self.rank != 0:
            return
        ckpt_dir = self.args.ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        unet_to_save = self.pipeline.unet.module if isinstance(self.pipeline.unet, DDP) else self.pipeline.unet
        unet_to_save.save_pretrained(os.path.join(ckpt_dir, f'unet_ep{epoch}.pt'))

    # ----------------------- epoch loop -----------------------
    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            torch.cuda.reset_peak_memory_stats(self.device)

            tr_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch()

            if self.rank == 0:
                print(f"Epoch {epoch}: train={tr_loss:.4f}, val={val_loss:.4f}")
                
                if self.args.use_wandb:

                    # gpu stats
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
                    max_alloc = torch.cuda.max_memory_reserved(device=self.device) / 1024 ** 3
                    
                    peak = torch.cuda.max_memory_reserved(self.device) / 1024**3
                    current = torch.cuda.memory_allocated(self.device) / 1024**3
                    
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': tr_loss,
                        'val_loss': val_loss,
                        'gpu/clock_MHz': gpu_clock,
                        'gpu/max_allocated_GB': max_alloc,
                        'gpu/pk_GB': peak,
                        'gpu/cur_GB': current
                        })
                    

            if epoch % self.args.save_every == 0:
                self._save_ckpt(epoch)
                
            

# ---------------------------
# DDP launcher
# ---------------------------

def ddp_launch(rank: int, args):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    local_rank = rank  # in single‑node multi‑gpu setup

    # set master addr/port if not already defined
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12355')

    init_process_group(backend='nccl', rank=rank, world_size=int(os.environ['WORLD_SIZE']))

    trainer = TuneAVideoTrainerDDP(args, rank, local_rank, int(os.environ['WORLD_SIZE']))
    trainer.train()

    destroy_process_group()


# ---------------------------
# Argparse
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--zhat_dir', type=str, default='./data/Predicted_latents')
    p.add_argument('--sem_dir', type=str, default='./data/Semantic_embeddings')
    p.add_argument('--ckpt_dir', type=str, default='./EEG2Video/checkpoints/TuneAVideo')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save_every', type=int, default=1)
    p.add_argument('--use_wandb', action='store_true')

    # performance flags (same as v4)
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--use_8bit_adam', action='store_true')
    p.add_argument('--use_xformers', action='store_true')
    p.add_argument('--use_gradient_checkpointing', action='store_true')
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--use_set_to_none', action='store_true')
    p.add_argument('--prefetch_factor', type=int, default=None)
    p.add_argument('--use_empty_cache', action='store_true')
    p.add_argument('--use_channels_last', action='store_true')
    p.add_argument('--use_torch_compile', action='store_true')

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError('DDP version requires at least 2 GPUs.')
    mp.spawn(ddp_launch, args=(args,), nprocs=world_size, join=True)
