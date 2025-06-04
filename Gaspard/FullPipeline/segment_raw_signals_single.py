"""
segment_raw_signals_single.py
--------------------------------
Utility functions to extract **one** 2‑second raw EEG segment from the SEED‑DV dataset
without creating any intermediate files.  The logic reproduces exactly the slicing
strategy of the original `segment_raw_signals_200Hz.py`, but scoped to a single
(subject, block, concept, repetition) tuple so that it can be imported and used
inside a streaming inference pipeline.

Example
-------
>>> from segment_raw_signals_single import extract_2s_segment
>>> seg = extract_2s_segment(subject=3, block=1, concept=12, repetition=4)
>>> seg.shape  # (channels, timepoints)
(62, 400)

CLI
---
The module also exposes a small command‑line interface so you can test it quickly:

$ python segment_raw_signals_single.py --subject 3 --block 1 --concept 12 --rep 4 

This will save the segment as `seg_sub3_b1_c12_r4.npy` in the current folder.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Final

__all__ = ["extract_2s_segment"]

FS: Final[int] = 200  # sampling frequency (Hz)
_BASELINE_SEC: Final[int] = 3  # seconds of baseline before each concept
_REPS_PER_CONCEPT: Final[int] = 5  # number of video repetitions per concept
_CONCEPTS_PER_BLOCK: Final[int] = 40


def _validate_indices(subject: int, block: int, concept: int, repetition: int) -> None:
    if subject < 1:
        raise ValueError("`subject` must be >= 1 (files are named sub{subject}.npy)")
    if not 0 <= block <= 6:
        raise ValueError("`block` must be in [0, 6]")
    if not 0 <= concept < _CONCEPTS_PER_BLOCK:
        raise ValueError("`concept` must be in [0, 39]")
    if not 0 <= repetition < _REPS_PER_CONCEPT:
        raise ValueError("`repetition` must be in [0, 4]")


def extract_2s_segment(
    *,
    subject: int,
    block: int,
    concept: int,
    repetition: int,
    eeg_root: str = "./data/EEG",
    fs: int = FS,
) -> np.ndarray:
    """Return one raw 2‑second EEG segment (62 × 400).

    Parameters
    ----------
    subject : int
        Subject identifier *N* (expects file `subN.npy` in *eeg_root*).
    block : int
        Video block index in ``0 … 6``.
    concept : int
        Concept index in ``0 … 39``.
    repetition : int
        Repetition index in ``0 … 4`` (five clips per concept).
    eeg_root : str, default "./data/EEG"
        Folder containing the original continuous EEG `.npy` files.
    fs : int, default 200
        Sampling frequency (Hz).  Change only if you down‑sampled differently.

    Returns
    -------
    np.ndarray, shape (62, 2 × fs)
        The requested EEG segment.
    """

    _validate_indices(subject, block, concept, repetition)

    path = os.path.join(eeg_root, f"sub{subject}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"EEG file not found: {path}")

    # Continuous data: shape (7, 62, T)
    data = np.load(path, mmap_mode="r")
    block_data = data[block]  # (62, T)

    baseline_len = _BASELINE_SEC * fs          # 3 s * 200 Hz = 600 samples
    video_len = 2 * fs                         # 2 s * 200 Hz = 400 samples
    concept_stride = baseline_len + _REPS_PER_CONCEPT * video_len  # 600 + 5×400 = 2600 samples

    # Compute absolute start index for the desired segment inside the block timeline
    start = concept * concept_stride           # skip preceding concepts
    start += baseline_len                      # skip baseline of our concept
    start += repetition * video_len            # choose repetition *within* concept

    end = start + video_len
    segment = block_data[:, start:end]         # shape (62, 400)

    if segment.shape[1] != video_len:
        raise RuntimeError("Segment length mismatch – check indices and data integrity.")
    seg_full = np.asarray(segment)[None, None, None, ...]
    print(seg_full.shape, seg_full.dtype)  # Debugging output
    return np.asarray(seg_full)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a single 2‑second EEG segment.")
    parser.add_argument("--subject", type=int, required=True, help="Subject number (1‑20)")
    parser.add_argument("--block", type=int, required=True, help="Block index 0‑6")
    parser.add_argument("--concept", type=int, required=True, help="Concept index 0‑39")
    parser.add_argument("--rep", type=int, required=True, help="Repetition index 0‑4")
    parser.add_argument("--eeg_root", type=str, default="./data/EEG", help="Path to EEG folder")

    args = parser.parse_args()

    seg = extract_2s_segment(
        subject=args.subject,
        block=args.block,
        concept=args.concept,
        repetition=args.rep,
        eeg_root=args.eeg_root,
    )

    out_name = f"seg_sub{args.subject}_b{args.block}_c{args.concept}_r{args.rep}.npy"
    np.save(out_name, seg)
    print("Saved", out_name, "->", seg.shape)
