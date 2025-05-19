import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from torch.cuda.amp import autocast
from tqdm import tqdm
import wandb

# -----------------------------------------------------------------------------
# NumPy ≥ 2.0 compatibility helpers
# -----------------------------------------------------------------------------

def _to_float32_array(x: np.ndarray | np.number | list) -> np.ndarray:
    """Return *x* as an ndarray of dtype float32.

    This helper avoids the NumPy‑2 scalar issue (torch cannot create tensors
    directly from NumPy scalar objects such as ``np.float32``).
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:  # scalar → make it 1‑D to satisfy torch.from_numpy
        arr = arr.reshape(1)
    return arr

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class EEGVideoDataset(Dataset):
    """(ẑ, eₜ) pairs stored as *.npy* blocks on disk."""

    def __init__(self, zhat_dir: str, sem_dir: str):
        z_blocks, e_blocks = [], []

        for fname in sorted(os.listdir(zhat_dir)):
            if fname.endswith(".npy"):
                z_blocks.append(np.load(os.path.join(zhat_dir, fname), mmap_mode=None))
        for fname in sorted(os.listdir(sem_dir)):
            if fname.endswith(".npy"):
                e_blocks.append(np.load(os.path.join(sem_dir, fname), mmap_mode=None))

        if not z_blocks or not e_blocks:
            raise RuntimeError("No .npy files found in provided directories.")

        self.z_hat = np.concatenate(z_blocks, axis=0).astype(np.float32, copy=False)
        self.e_t   = np.concatenate(e_blocks, axis=0).astype(np.float32, copy=False)

        if self.z_hat.shape[0] != self.e_t.shape[0]:
            raise ValueError(
                f"Mismatched sample counts: {self.z_hat.shape[0]} ẑ vs "
                f"{self.e_t.shape[0]} eₜ samples"
            )

    def __len__(self) -> int:  # type: ignore[override]
        return self.z_hat.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        z0_np = _to_float32_array(self.z_hat[idx])
        et_np = _to_float32_array(self.e_t[idx])
        # ``torch.from_numpy`` shares memory → no extra copy. Call .clone() to avoid
        # unexpected inplace ops downstream.
        return torch.from_numpy(z0_np).clone(), torch.from_numpy(et_np).clone()

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class TuneAVideoTrainer:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --------------------------- Data -----------------------------------
        dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        train_size = int(0.8 * len(dataset))
        val_size   = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

        # ------------------------ Pre‑trained parts -------------------------
        root = args.root
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae"
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            f"{root}/EEG2Video/EEG2Video_New/Generation/stable-diffusion-v1-4",
            subfolder="unet",
        ).to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )

        # ------------------------------ Pipeline ---------------------------
        self.pipeline = TuneAVideoPipeline(
            vae=self.vae,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
        )
        self.pipeline.unet.train()

        # ---------------------------- Projection ---------------------------
        sem_dim   = dataset[0][1].shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_eeg = nn.Linear(sem_dim, cross_dim).to(self.device)

        # -------------------------- Optimizer ------------------------------
        self.optimizer = torch.optim.Adam(
            list(self.pipeline.unet.parameters()) + list(self.proj_eeg.parameters()),
            lr=args.lr,
        )
        # Diffusers scheduler keeps internal timesteps → set once.
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # ---------------------------- wandb -------------------------------
        if args.use_wandb:
            wandb.init(project="eeg2video-tuneavideo", config=vars(args))

        self.args = args

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.pipeline.unet.train()
        running_loss = 0.0
        for z0, et in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} – train"):
            z0 = z0.to(self.device).permute(0, 2, 1, 3, 4).contiguous()
            et = self.proj_eeg(et.to(self.device).unsqueeze(1))

            noise = torch.randn_like(z0)
            timesteps = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
            z_t = self.scheduler.add_noise(z0, noise, timesteps)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et)
            loss = F.mse_loss(out.sample, noise)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            torch.cuda.empty_cache()
        return running_loss / len(self.train_loader)

    def _validate_epoch(self) -> float:
        self.pipeline.unet.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast():
            for z0, et in tqdm(self.val_loader, desc="Validation"):
                z0 = z0.to(self.device).permute(0, 2, 1, 3, 4).contiguous()
                et = self.proj_eeg(et.to(self.device).unsqueeze(1))
                noise = torch.randn_like(z0)
                timesteps = torch.randint(0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device)
                z_t = self.scheduler.add_noise(z0, noise, timesteps)

                # Some CUDA/blas versions fall back to fp32 – retry in fp32 if needed.
                try:
                    out = self.pipeline.unet(z_t, timesteps, encoder_hidden_states=et)
                except RuntimeError as e:
                    if "CUBLAS_STATUS_EXECUTION_FAILED" in str(e):
                        out = self.pipeline.unet(z_t.float(), timesteps, encoder_hidden_states=et.float())
                    else:
                        raise
                val_loss += F.mse_loss(out.sample, noise).item()
        return val_loss / len(self.val_loader)

    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt_dir = os.path.join(self.args.root, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"tuneavideo_unet_epoch{epoch}.pt")
        self.pipeline.unet.save_pretrained(path)
        print(f"[Checkpoint] Saved UNet weights to {path}")

    # ------------------------------------------------------------------

    def train(self) -> None:
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss   = self._validate_epoch()
            print(f"Epoch {epoch:>3} │ train={train_loss:.4f} │ val={val_loss:.4f}")
            torch.cuda.empty_cache()
            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    home = os.environ.get("HOME", os.environ.get("USERPROFILE", ""))
    root_default = os.path.join(home, "EEG2Video")

    p = argparse.ArgumentParser(description="Train Tune-A-Video on EEG embeddings")
    p.add_argument("--zhat_dir",   type=str,   default=f"{root_default}/data/Predicted_latents")
    p.add_argument("--sem_dir",    type=str,   default=f"{root_default}/data/Semantic_embeddings")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--save_every", type=int,   default=10, help="Save checkpoint every N epochs")
    p.add_argument("--use_wandb",  action="store_true")
    p.add_argument("--root",       type=str,   default=root_default)
    return p.parse_args()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    trainer = TuneAVideoTrainer(args)
    trainer.train()
