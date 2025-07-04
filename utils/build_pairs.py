"""Generate aligned EEG/video latent pairs."""

import os
import sys
import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root) if project_root not in sys.path else None

from utils.align import load_aligned_latents


def build_pairs(eeg_dir: str, video_dir: str, output_dir: str) -> None:
    """Walk over latent directories and store aligned pairs.

    Parameters
    ----------
    eeg_dir : str
        Directory containing EEG latent ``.npy`` files. Each subject has its
        own folder ``subX`` with files organised as ``subX/Block{block}/index.npy``.
        Indices range from ``1`` to ``200`` following
        ``index = 5 * concept + repetition``.
    video_dir : str
        Directory with the corresponding video latents (``.npy`` or ``.npz``).
        The hierarchy mirrors the EEG data but without the subject prefix,
        i.e. ``Block{block}/index.npy``.
    output_dir : str
        Where the paired ``.npz`` files will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    subjects = [d for d in os.listdir(eeg_dir) if os.path.isdir(os.path.join(eeg_dir, d))]
    for sub in tqdm(subjects, desc="Processing subjects"):
        for block in os.listdir(os.path.join(eeg_dir, sub)):
            eeg_block = os.path.join(eeg_dir, sub, block)
            if not os.path.isdir(eeg_block):
                continue
            for idx in range(1, 201):
                eeg_path = os.path.join(eeg_block, f"{idx}.npy")
                if not os.path.exists(eeg_path):
                    continue
                video_path = os.path.join(video_dir, block, f"{idx}.npz")
                if not os.path.exists(video_path):
                    print(f"Video latent missing at path {video_path}, skipping")
                    continue

                rel_path = os.path.relpath(eeg_path, eeg_dir)
                out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".npz")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                eeg_latent, video_latent = load_aligned_latents(eeg_path, video_path)

                np.savez_compressed(
                    out_path,
                    eeg_latent=eeg_latent.astype(np.float32),
                    video_latent=video_latent.astype(np.float32),
                )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build EEG-video latent pairs")
    parser.add_argument("--eeg_dir", default="./data/gaspardnew/eeg_segments", help="directory with EEG latents")
    parser.add_argument("--video_dir", default="./data/gaspardnew/video_latents", help="directory with video latents")
    parser.add_argument("--output_dir", default="./data/gaspardnew/latent_pairs", help="where to save paired latents")
    args = parser.parse_args()

    build_pairs(args.eeg_dir, args.video_dir, args.output_dir)


if __name__ == "__main__":
    main()
