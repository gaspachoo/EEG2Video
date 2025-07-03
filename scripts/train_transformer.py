#!/usr/bin/env python3
"""Training script for the Transformer (P1).
The VAE and diffusion models remain frozen during this stage."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from transformer.model import Seq2SeqTransformer
from diffusers import DiffusionPipeline
from encoders.video_vae.extract_latents import load_vae


class PairDataset(Dataset):
    """Dataset loading EEG/video latent pairs stored as ``.npz`` files."""

    def __init__(self, root: str | Path):
        self.files = sorted(Path(root).rglob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No npz files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        src = torch.from_numpy(data["eeg_latent"]).float()
        tgt = torch.from_numpy(data["video_latent"]).float()
        return src, tgt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer with frozen VAE and diffusion modules")
    p.add_argument("data", type=Path, help="folder with paired latents")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save", type=Path, default=Path("transformer.pt"))
    p.add_argument("--freeze_vae", action="store_true")
    p.add_argument("--freeze_diffuser", action="store_true")
    p.add_argument("--vae_weights", type=str, default="encoders/video_vae/vae.pt")
    p.add_argument("--diffusion_weights", type=str, default="open-sora/open_sora")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Seq2SeqTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    vae = load_vae(args.vae_weights, device)
    diffuser = DiffusionPipeline.from_pretrained(args.diffusion_weights, torch_dtype=torch.float16).to(device)

    if args.freeze_vae:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
    if args.freeze_diffuser:
        diffuser.to(device)
        diffuser.eval()
        for p in diffuser.parameters():
            p.requires_grad = False

    print("Starting P1: Transformer training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_gt = tgt[:, 1:]

            out = model(src, tgt_in)
            mse = F.mse_loss(out, tgt_gt)
            cos = 1 - F.cosine_similarity(out.view(out.size(0), -1), tgt_gt.view(tgt_gt.size(0), -1)).mean()
            loss = mse + cos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch}: {total/len(loader):.4f}")

    torch.save(model.state_dict(), args.save)
    print("P1 finished")


if __name__ == "__main__":
    main()
