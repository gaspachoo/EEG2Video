#!/usr/bin/env python3
"""Generate EEG latents with GLMNet and split them by block."""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EEGtoVideo.GLMNet.inference_glmnet import inf_glmnet, OCCIPITAL_IDX
from EEGtoVideo.GLMNet.modules.utils_glmnet import GLMNet, load_scaler, load_raw_stats
from utils import stack_eeg_windows


def generate_latents(raw_dir: str, ckpt_dir: str, output_dir: str, prefix: str, device: str = "cpu") -> None:
    """Run GLMNet inference and write one latent file per clip."""
    os.makedirs(output_dir, exist_ok=True)

    scaler = load_scaler(os.path.join(ckpt_dir, "scaler.pkl"))
    stats = load_raw_stats(os.path.join(ckpt_dir, "raw_stats.npz"))
    model_path = os.path.join(ckpt_dir, "glmnet_best.pt")

    for fname in os.listdir(raw_dir):
        if not fname.endswith(".npy") or not fname.startswith(prefix):
            continue
        subject = os.path.splitext(fname)[0]
        raw_windows = np.load(os.path.join(raw_dir, fname))

        time_len = raw_windows.shape[-1]
        num_channels = raw_windows.shape[-2]
        model = GLMNet.load_from_checkpoint(
            model_path,
            OCCIPITAL_IDX,
            C=num_channels,
            T=time_len,
            device=device,
        )

        embeddings = inf_glmnet(model, scaler, raw_windows, stats, device)
        dim = embeddings.shape[-1]
        embeddings = embeddings.reshape(7, 40, 5, 7, dim)

        for block in range(7):
            for concept in range(40):
                for rep in range(5):
                    windows = embeddings[block, concept, rep]
                    stacked = stack_eeg_windows(windows, 0)
                    index = 5 * concept + rep
                    out_dir = Path(output_dir) / subject / str(block)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / f"{index}.npy", stacked.astype(np.float32))
        print(f"Saved latents for {subject}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate EEG latents for build_pairs")
    p.add_argument("--raw_dir", default="./data/Preprocessing/Segmented_500ms_sw", help="pre-windowed EEG directory")
    p.add_argument("--subject_prefix", default="sub3", help="process subjects matching this prefix")
    p.add_argument("--checkpoint_path", help="directory with GLMNet checkpoint")
    p.add_argument("--output_dir", default="./data/eeg_segments", help="where to store EEG latents")
    p.add_argument("--device", default="cpu", help="computation device")
    args = p.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = f"./EEGtoVideo/checkpoints/glmnet/{args.subject_prefix}_label_cluster"

    generate_latents(args.raw_dir, args.checkpoint_path, args.output_dir, args.subject_prefix, args.device)


if __name__ == "__main__":
    main()

