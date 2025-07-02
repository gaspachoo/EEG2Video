import argparse
from pathlib import Path

import torch

from .model import Seq2SeqTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path, help="trained model checkpoint")
    p.add_argument("src", type=Path, help="tensor with source latents")
    p.add_argument("--steps", type=int, default=16, help="number of latents to generate")
    p.add_argument("--out", type=Path, default=Path("generated.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    src = torch.load(args.src)
    if src.dim() == 2:
        src = src.unsqueeze(0)
    src = src.to(device)

    tgt = torch.zeros(src.size(0), 1, model.d_model, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(args.steps):
            out = model(src, tgt)
            next_latent = out[:, -1:].detach()
            generated.append(next_latent.cpu())
            tgt = torch.cat([tgt, next_latent], dim=1)

    gen_seq = torch.cat(generated, dim=1)
    torch.save(gen_seq, args.out)


if __name__ == "__main__":
    main()
