#!/usr/bin/env python3
"""Generate EEG latents with GLMNet and split them by block."""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EEGtoVideo.GLMNet.inference_glmnet import inf_glmnet, OCCIPITAL_IDX
from EEGtoVideo.GLMNet.modules.utils_glmnet import GLMNet, load_scaler, load_raw_stats
from utils import stack_eeg_windows


def generate_latents(raw_dir: str, ckpt_dir: str, output_dir: str, prefix_list: list[str], device: str = "cpu") -> None:
    """Run GLMNet inference and write one latent file per clip."""
    os.makedirs(output_dir, exist_ok=True)

    scaler = load_scaler(os.path.join(ckpt_dir, "scaler.pkl"))
    stats = load_raw_stats(os.path.join(ckpt_dir, "raw_stats.npz"))
    model_path = os.path.join(ckpt_dir, "glmnet_best.pt")

    for prefix in tqdm(prefix_list):
        fname = f"{prefix}.npy"
        if not os.path.exists(os.path.join(raw_dir, fname)):
            raise FileNotFoundError(f"Raw EEG file {fname} not found in {raw_dir}")
        
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

        for block in tqdm(range(7), desc=f"Processing {subject}"):
            for concept in range(40):
                for rep in range(5):
                    windows = embeddings[block, concept, rep]
                    stacked = stack_eeg_windows(windows, 0)
                    index = 5 * concept + rep
                    out_dir = Path(f"{output_dir}/{subject}/Block{block}")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / f"{index}.npy", stacked.astype(np.float32))
        tqdm.write(f"Saved latents for {subject}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate EEG latents for build_pairs")
    p.add_argument("--raw_dir", default="./data/Preprocessing/Segmented_500ms_sw", help="pre-windowed EEG directory")
    p.add_argument("--subject_prefix_list", default= [f"sub{i}" for i in range(1,21)], help="process subjects matching this prefix")
    p.add_argument("--checkpoint_path", default=f"./EEGtoVideo/checkpoints/glmnet/sub3_label_cluster", help="directory with GLMNet checkpoint")
    p.add_argument("--output_dir", default="./data/gaspardnew/eeg_segments", help="where to store EEG latents")
    p.add_argument("--device", default="cpu", help="computation device")
    args = p.parse_args()

    generate_latents(args.raw_dir, args.checkpoint_path, args.output_dir, args.subject_prefix_list, args.device)


if __name__ == "__main__":
    main()

