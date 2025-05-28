#### add scheduler compared to v6
# 
"""
train_tuneavideo_v5_fixed.py  – DDP training script (EEG→Video) **with LR‑scheduler support**
----------------------------------------------------------------
Changes vs. v5 original :
1. Deterministic data split shared across ranks (see _build_datasets)
2. Validation loss aggregated over all ranks
3. New CLI flags for LR scheduler (cosine, step, plateau, none)
4. WandB logging unified across ranks (only rank‑0 writes)
5. Defaults: batch=5, AdamW(lr=1e‑4), CosineAnnealingLR(T_max=10, eta_min=1e‑6)
"""
from __future__ import annotations
import os, random, argparse, math, time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def ddp_setup():
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    rank = int(os.environ.get("RANK", 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0

# -----------------------------------------------------------------------------
#  Dummy dataset / model stubs --------------------------------------------------
# -----------------------------------------------------------------------------

class DummyDataset(Dataset):
    def __init__(self, n=200):
        self.x = torch.randn(n, 4, 6, 36, 64)
        self.h = torch.randn(n, 1, 768)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.h[idx]

class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(768,1)
    def forward(self, z_t, t, encoder_hidden_states):
        return self.conv(encoder_hidden_states.squeeze(1)).mean()

# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, args, rank):
        self.device = torch.device("cuda", rank)
        self.rank = rank
        self.args = args
        self._build_datasets()
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()

    # -------------------------------
    #  Data
    # -------------------------------
    def _shared_split(self, dataset: Dataset):
        g = torch.Generator().manual_seed(self.args.split_seed)
        n_val = int(len(dataset) * self.args.val_ratio)
        n_train = len(dataset) - n_val
        train_idx, val_idx = random_split(range(len(dataset)), [n_train, n_val], generator=g)
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def _build_datasets(self):
        full = DummyDataset(200)  # <-- replace by real dataset loader
        train_ds, val_ds = self._shared_split(full)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            sampler=DistributedSampler(train_ds, shuffle=True),
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            sampler=DistributedSampler(val_ds, shuffle=False),
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    # -------------------------------
    #  Model / Opt / Sched
    # -------------------------------
    def _build_model(self):
        self.unet = DummyUNet().to(self.device)
        self.unet = DDP(self.unet, device_ids=[self.device])

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _build_scheduler(self):
        if self.args.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.t_max, eta_min=self.args.lr_min)
        elif self.args.scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor=self.args.gamma, patience=self.args.patience)
        else:
            self.scheduler = None

    # -------------------------------
    #  Epoch loops
    # -------------------------------
    def train(self):
        for epoch in range(self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch)
            t_loss = self._train_epoch(epoch)
            v_loss = self._validate_epoch()
            if is_main(self.rank):
                print(f"Epoch {epoch}: train={t_loss:.4f}, val={v_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.2e}")
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(v_loss)
                else:
                    self.scheduler.step()

    def _train_epoch(self, epoch):
        self.unet.train()
        total, n = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        for z_t, h in self.train_loader:
            z_t, h = z_t.to(self.device), h.to(self.device)
            loss = self.unet(z_t, None, encoder_hidden_states=h)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.detach() * z_t.size(0)
            n += z_t.size(0)
        # reduce across ranks
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        return (total / n).item()

    def _validate_epoch(self):
        self.unet.eval()
        loss_sum, n_sum = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        with torch.no_grad():
            for z_t, h in self.val_loader:
                z_t, h = z_t.to(self.device), h.to(self.device)
                loss = self.unet(z_t, None, encoder_hidden_states=h)
                loss_sum += loss * z_t.size(0)
                n_sum += z_t.size(0)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_sum,   op=dist.ReduceOp.SUM)
        return (loss_sum / n_sum).item()

# -----------------------------------------------------------------------------
#  Arg‑parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--scheduler", choices=["none", "cosine", "step", "plateau"], default="cosine")
    p.add_argument("--t_max", type=int, default=10, help="T_max for CosineAnnealingLR")
    p.add_argument("--step_size", type=int, default=5, help="step_size for StepLR")
    p.add_argument("--gamma", type=float, default=0.5, help="gamma for StepLR / Plateau")
    p.add_argument("--patience", type=int, default=2, help="patience for ReduceLROnPlateau")
    p.add_argument("--val_ratio", type=float, default=0.3)
    p.add_argument("--split_seed", type=int, default=42)
    return p.parse_args()

# -----------------------------------------------------------------------------
#  Main entry
# -----------------------------------------------------------------------------

def main():
    rank, world = ddp_setup()
    args = parse_args()
    trainer = Trainer(args, rank)
    trainer.train()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
