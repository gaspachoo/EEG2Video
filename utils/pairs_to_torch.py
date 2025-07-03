"""Gather latent pair archives into a single torch file."""

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm



def load_all_pairs(pair_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load every ``npz`` in ``pair_dir`` and concatenate their contents."""
    eeg_list = []
    vid_list = []
    pair_files = []
    for root, _, files in os.walk(pair_dir):
        for name in files:
            if name.endswith(".npz"):
                pair_files.append(os.path.join(root, name))

    for path in tqdm(pair_files, desc="Loading pairs"):
        with np.load(path) as data:
            eeg_list.append(torch.from_numpy(data["eeg_latent"]))
            vid_list.append(torch.from_numpy(data["video_latent"]))

    src = torch.cat(eeg_list, dim=0)
    tgt = torch.cat(vid_list, dim=0)
    return src, tgt


def save_torch_dataset(pair_dir: str, out_path: str) -> None:
    """Save stacked latents from ``pair_dir`` into ``out_path``."""
    src, tgt = load_all_pairs(pair_dir)
    torch.save({"src": src.float(), "tgt": tgt.float()}, out_path)



def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Convert latent pair npz files to a torch archive")
    p.add_argument("--pair_dir", default="./data/latent_pairs", help="directory containing npz pairs")
    p.add_argument("--out", default="./data/pairs.pt", help="output torch file")
    args = p.parse_args()

    save_torch_dataset(args.pair_dir, args.out)


if __name__ == "__main__":
    main()
