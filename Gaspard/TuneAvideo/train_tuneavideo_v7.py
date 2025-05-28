import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

import numpy as np
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from tqdm import tqdm
import pynvml
import wandb

############################################################
#                      Dataset                             #
############################################################

class EEGVideoDataset(Dataset):
    """Numpy‑folder dataset holding (ẑ, semantic‑embedding) pairs."""

    def __init__(self, zhat_dir: str, sem_dir: str):
        z_blocks, e_blocks = [], []
        for fname in sorted(os.listdir(zhat_dir)):
            if fname.endswith(".npy"):
                z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        for fname in sorted(os.listdir(sem_dir)):
            if fname.endswith(".npy"):
                e_blocks.append(np.load(os.path.join(sem_dir, fname)))

        self.z_hat = np.concatenate(z_blocks, axis=0)
        self.e_t   = np.concatenate(e_blocks, axis=0)
        assert self.z_hat.shape[0] == self.e_t.shape[0], (
            f"Mismatch z_hat {self.z_hat.shape} vs e_t {self.e_t.shape}")

    def __len__(self):
        return self.z_hat.shape[0]

    def __getitem__(self, idx):
        z0 = torch.from_numpy(self.z_hat[idx]).float()
        et = torch.from_numpy(self.e_t[idx]).float()
        return z0, et

############################################################
#                       Trainer                            #
############################################################

class TuneAVideoTrainerDDP:
    """Distributed trainer for Tune‑a‑Video fine‑tuning on EEG guidance."""

    def __init__(self, args, rank: int, local_rank: int, world_size: int):
        self.args        = args
        self.rank        = rank
        self.local_rank  = local_rank
        self.world_size  = world_size
        self.device      = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        # -------------- Data -------------- #
        ds      = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        n_train = int(0.8 * len(ds))
        n_val   = len(ds) - n_train

        # Split ONCE deterministically on every rank to keep identical indices
        g = torch.Generator().manual_seed(args.split_seed)
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

        self.train_sampler = DistributedSampler(train_ds, world_size, rank)
        self.val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)

        dl_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        if args.prefetch_factor is not None and args.num_workers > 0:
            dl_kwargs["prefetch_factor"] = args.prefetch_factor

        self.train_loader = DataLoader(train_ds, sampler=self.train_sampler, **dl_kwargs)
        self.val_loader   = DataLoader(val_ds,   sampler=self.val_sampler,   **dl_kwargs)

        # -------------- Models -------------- #
        self.vae       = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cpu").eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.unet      = UNet3DConditionModel.from_pretrained_2d(
            args.base_model_path, subfolder="unet").to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        
        
        # Freeze everything except a few q‑projection layers
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        trainable_suffixes = ("attn1.to_q", "attn2.to_q", "attn_temp")
        for n, m in self.unet.named_modules():
            if any(n.endswith(suf) for suf in trainable_suffixes):
                for p in m.parameters():
                    p.requires_grad_(True)

        # Wrap in pipeline then DDP
        self.pipeline = TuneAVideoPipeline(vae=self.vae, tokenizer=self.tokenizer,
                                           unet=self.unet, scheduler=self.scheduler)
        self.pipeline.unet.train()

        sem_dim   = ds[0][1].shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_eeg = nn.Linear(sem_dim, cross_dim).to(self.device)

        self.pipeline.unet = DDP(self.pipeline.unet, device_ids=[local_rank], output_device=local_rank,
                                 gradient_as_bucket_view=True)
        self.proj_eeg      = DDP(self.proj_eeg,      device_ids=[local_rank], output_device=local_rank)

        # -------------- Optim / AMP -------------- #
        params = list(self.pipeline.unet.parameters()) + list(self.proj_eeg.parameters())
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            self.optimizer = bnb.AdamW8bit(params, lr=args.lr)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=args.lr)

        self.use_amp   = args.mixed_precision
        self.scaler    = GradScaler() if self.use_amp else None
        self.acc_steps = args.gradient_accumulation_steps
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.lr_scheduler = self._build_lr_scheduler()

        # Perf‑flags
        if args.use_channels_last:
            self.pipeline.unet.to(memory_format=torch.channels_last)
        if args.use_xformers:
            self.pipeline.unet.module.enable_xformers_memory_efficient_attention()
        if args.use_gradient_checkpointing:
            self.pipeline.unet.module.enable_gradient_checkpointing()
        if args.use_torch_compile:
            self.pipeline.unet = torch.compile(self.pipeline.unet)
            self.proj_eeg      = torch.compile(self.proj_eeg)

        ################ GPU metrics + W&B ################
        pynvml.nvmlInit()
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

        if args.use_wandb and rank == 0:
            wandb.init(project="eeg2video-tuneavideo", config=vars(args))

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------

    def _format_batch(self, z0, et):
        z0 = z0.to(self.device).permute(0, 2, 1, 3, 4).contiguous()  # (B,4,6,36,64)
        eh = self.proj_eeg(et.to(self.device).unsqueeze(1)).contiguous()
        return z0, eh

    def _step_ddpm_inputs(self, z0):
        noise     = torch.randn_like(z0)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps,
                                  (z0.size(0),), device=self.device).long()
        z_t = self.scheduler.add_noise(z0, noise, timesteps)
        return z_t, noise, timesteps
    
    def _build_lr_scheduler(self):
        a = self.args
        if a.lr_sched == 'none':
            return None
        if a.lr_sched == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=a.t_max, eta_min=a.lr_min)
        if a.lr_sched == 'step':
            return StepLR(self.optimizer, step_size=a.step_size, gamma=a.gamma)
        if a.lr_sched == 'plateau':
            return ReduceLROnPlateau(self.optimizer, factor=a.gamma, patience=a.patience, min_lr=a.lr_min)
        raise ValueError(a.lr_sched)

    # ---------------------------------------------------------------------
    # Training / Validation
    # ---------------------------------------------------------------------

    def _train_epoch(self, epoch):
        self.pipeline.unet.train()
        self.train_sampler.set_epoch(epoch)

        running = 0.0
        self.optimizer.zero_grad(set_to_none=self.args.use_set_to_none)
        

        for i, (z0, et) in enumerate(tqdm(self.train_loader, desc=f"Rank {self.rank} | Train {epoch}", position=self.rank)):
            z0, eh = self._format_batch(z0, et)
            z_t, noise, timesteps = self._step_ddpm_inputs(z0)

            with autocast(enabled=self.use_amp):
                out  = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=eh)
                loss = F.mse_loss(out.sample, noise) / self.acc_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.acc_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=self.args.use_set_to_none)

            running += loss.item() * self.acc_steps

        # sync mean train loss for nicer logging
        train_loss = torch.tensor(running / len(self.train_loader), device=self.device)
        train_loss = self._sync_mean(train_loss)
        return train_loss.item()

    def _validate_epoch(self):
        self.pipeline.unet.eval()
        total_loss   = torch.zeros(1, device=self.device)
        total_samples = torch.zeros(1, device=self.device)

        with torch.no_grad():
            for z0, et in tqdm(self.val_loader, desc=f"Rank {self.rank} | Val", position=self.rank):
                z0, eh = self._format_batch(z0, et)
                z_t, noise, timesteps = self._step_ddpm_inputs(z0)
                with autocast(enabled=self.use_amp):
                    out  = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=eh)
                    loss = F.mse_loss(out.sample, noise)
                bs = z0.size(0)
                total_loss   += loss * bs
                total_samples += bs

        # Aggregate across all ranks
        self._sync_sum(total_loss)
        self._sync_sum(total_samples)
        return (total_loss / total_samples).item()

    # ------------------------------------------------------------------
    # Utility sync helpers
    # ------------------------------------------------------------------

    def _sync_sum(self, tensor):
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _sync_mean(self, tensor):
        self._sync_sum(tensor)
        if dist.is_initialized():
            tensor /= self.world_size
        return tensor

    # ------------------------------------------------------------------
    # Checkpointing & Epoch loop
    # ------------------------------------------------------------------

    def _save_ckpt(self, epoch):
        if self.rank != 0:
            return
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        unet_to_save = self.pipeline.unet.module if isinstance(self.pipeline.unet, DDP) else self.pipeline.unet
        unet_to_save.save_pretrained(os.path.join(self.args.ckpt_dir, f"unet_ep{epoch}.pt"))

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            torch.cuda.reset_peak_memory_stats(self.device)
            tr_loss  = self._train_epoch(epoch)
            val_loss = self._validate_epoch()

            # LR scheduler step
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
                
            if self.rank == 0:
                print(f"Epoch {epoch}: train={tr_loss:.4f}, val={val_loss:.4f}")
                if self.args.use_wandb:
                    mem_info  = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, pynvml.NVML_CLOCK_GRAPHICS)
                    peak_mem  = torch.cuda.max_memory_reserved(self.device) / 1024 ** 3
                    cur_mem   = torch.cuda.memory_allocated(self.device) / 1024 ** 3

                    wandb.log({
                        "epoch": epoch,
                        "train_loss": tr_loss,
                        "val_loss": val_loss,
                        "gpu/clock_MHz": gpu_clock,
                        "gpu/peak_GB": peak_mem,
                        "gpu/cur_GB": cur_mem,
                    })
            if epoch % self.args.save_every == 0:
                self._save_ckpt(epoch)

############################################################
#                   DDP launch helper                      #
############################################################

def ddp_launch(rank: int, args):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
    local_rank = rank

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")

    init_process_group(backend="nccl", rank=rank, world_size=int(os.environ["WORLD_SIZE"]))
    trainer = TuneAVideoTrainerDDP(args, rank, local_rank, int(os.environ["WORLD_SIZE"]))
    trainer.train()
    destroy_process_group()

############################################################
#                      CLI args                            #
############################################################

def parse_args():
    p = argparse.ArgumentParser()
    
    # Paths & data
    p.add_argument("--zhat_dir", type=str, default="./data/Predicted_latents")
    p.add_argument("--sem_dir", type=str, default="./data/Semantic_embeddings")
    p.add_argument("--base_model_path", type=str, default="./Gaspard/stable-diffusion-v1-4",
                   help="Path to the 2D SD‑1.4 checkpoint that was temporally inflated.")

    # Training hyper‑params
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--split_seed", type=int, default=42,
                   help="Seed used for deterministic train/val split across ranks.")

    # Scheduler
    p.add_argument('--lr_sched', type=str, choices=['none', 'cosine', 'step', 'plateau'], default='cosine')
    p.add_argument('--t_max', type=int, default=50)
    p.add_argument('--lr_min', type=float, default=1e-6)
    p.add_argument('--step_size', type=int, default=10)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--patience', type=int, default=3)

    # Perf / misc flags
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--use_xformers", action="store_true")
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_set_to_none", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--use_empty_cache", action="store_true")
    p.add_argument("--use_channels_last", action="store_true")
    p.add_argument("--use_torch_compile", action="store_true")
    
    # Checkpointing & logging
    p.add_argument("--use_wandb", action="store_true") 
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints/TuneAVideo")
    p.add_argument("--save_every", type=int, default=1)
    return p.parse_args()

############################################################
#                        Main                              #
############################################################

if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError("DDP version requires >=2 GPUs.")
    mp.spawn(ddp_launch, args=(args,), nprocs=n_gpus, join=True)
