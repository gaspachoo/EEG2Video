"""Utilities to segment SEED-DV EEG recordings."""

import os
import numpy as np
from tqdm import tqdm

__all__ = ["extract_2s_segment", "segment_all_files"]

FS = 200
_BASELINE_SEC = 3
_REPS_PER_CONCEPT = 5
_CONCEPTS_PER_BLOCK = 40


def extract_2s_segment(
    *,
    block,
    concept,
    repetition,
    subject=None,
    eeg_root="./data/EEG",
    fs=FS,
    data=None,
):
    """Return one raw 2-second EEG segment (62 × 2*fs).

    Parameters
    ----------
    block, concept, repetition : int
        Indices of the desired segment within a block.
    subject : int, optional
        Subject identifier (1-indexed). Required if ``data`` is ``None``.
    eeg_root : str
        Folder containing the ``sub*.npy`` files.
    fs : int
        Sampling rate (Hz).
    data : np.ndarray, optional
        Pre-loaded EEG recording with shape ``(7, 62, -1)``.
    """

    if data is None:
        if subject is None or subject < 1:
            raise ValueError("`subject` must be >= 1 when no `data` is provided")
        path = os.path.join(eeg_root, f"sub{subject}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = np.load(path, mmap_mode="r")

    if not 0 <= block <= 6:
        raise ValueError("`block` must be in [0, 6]")
    if not 0 <= concept < _CONCEPTS_PER_BLOCK:
        raise ValueError("`concept` must be in [0, 39]")
    if not 0 <= repetition < _REPS_PER_CONCEPT:
        raise ValueError("`repetition` must be in [0, 4]")

    block_data = data[block]

    baseline_len = _BASELINE_SEC * fs
    video_len = 2 * fs
    concept_stride = baseline_len + _REPS_PER_CONCEPT * video_len

    start = concept * concept_stride
    start += baseline_len
    start += repetition * video_len
    end = start + video_len

    segment = block_data[:, start:end]
    if segment.shape[1] != video_len:
        raise RuntimeError("Segment length mismatch")
    return segment


def segment_all_files(
    eeg_root="./data/EEG",
    output_dir="./data/Preprocessing/Segmented_Rawf_200Hz_2s",
    fs=FS,
):
    """Segment all EEG files into ``(7, 40, 5, 62, 2*fs)`` arrays."""
    os.makedirs(output_dir, exist_ok=True)

    sub_list = [f for f in os.listdir(eeg_root) if f.endswith(".npy")]
    for subname in sub_list:
        subject = int(os.path.splitext(subname)[0].replace("sub", ""))
        data = np.load(os.path.join(eeg_root, subname))

        segs = np.empty(
            (
                7,
                _CONCEPTS_PER_BLOCK,
                _REPS_PER_CONCEPT,
                62,
                2 * fs,
            ),
            dtype=data.dtype,
        )

        for blk in range(7):
            for cpt in tqdm(range(_CONCEPTS_PER_BLOCK), leave=False, desc=f"sub{subject} blk{blk}"):
                for rep in range(_REPS_PER_CONCEPT):
                    segs[blk, cpt, rep] = extract_2s_segment(
                        block=blk,
                        concept=cpt,
                        repetition=rep,
                        subject=subject,
                        eeg_root=eeg_root,
                        fs=fs,
                        data=data,
                    )

        np.save(os.path.join(output_dir, subname), segs)


if __name__ == "__main__":
    segment_all_files()
