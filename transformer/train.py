import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .model import Seq2SeqTransformer


class LatentDataset(Dataset):
    """Dataset loading pairs of latent sequences."""

    def __init__(self, path: Path):
        data = torch.load(path)
        self.src = data["src"]
        self.tgt = data["tgt"]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path, help="torch file with 'src' and 'tgt'")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LatentDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

    if args.save:
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    main()
