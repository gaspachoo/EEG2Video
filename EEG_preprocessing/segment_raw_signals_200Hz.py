"""Utilities to segment SEED-DV EEG recordings."""

import os
import numpy as np
from tqdm import tqdm

__all__ = ["extract_2s_segment", "segment_all_files"]

FS = 200
_BASELINE_SEC = 3
_REPS_PER_CONCEPT = 5
_CONCEPTS_PER_BLOCK = 40


def extract_2s_segment(*, subject, block, concept, repetition, eeg_root="./data/EEG", fs=FS):
    """Return one raw 2-second EEG segment (62 × 2*fs)."""
    if subject < 1:
        raise ValueError("`subject` must be >= 1")
    if not 0 <= block <= 6:
        raise ValueError("`block` must be in [0, 6]")
    if not 0 <= concept < _CONCEPTS_PER_BLOCK:
        raise ValueError("`concept` must be in [0, 39]")
    if not 0 <= repetition < _REPS_PER_CONCEPT:
        raise ValueError("`repetition` must be in [0, 4]")

    path = os.path.join(eeg_root, f"sub{subject}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = np.load(path, mmap_mode="r")
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
    """Segment all EEG files into (7, 40, 5, 62, 2*fs) arrays."""
    os.makedirs(output_dir, exist_ok=True)
    sub_list = [f for f in os.listdir(eeg_root) if f.endswith(".npy")]
    for subname in sub_list:
        npydata = np.load(os.path.join(eeg_root, subname))
        save_data = np.empty((0, 40, 5, 62, 2 * fs))
        for block_id in range(7):
            print("block:", block_id)
            now_data = npydata[block_id]
            l = 0
            block_data = np.empty((0, 5, 62, 2 * fs))
            for class_id in tqdm(range(40)):
                l += (3 * fs)
                class_data = np.empty((0, 62, 2 * fs))
                for i in range(5):
                    class_data = np.concatenate(
                        (class_data, now_data[:, l:l + 2 * fs].reshape(1, 62, 2 * fs))
                    )
                    l += (2 * fs)
                block_data = np.concatenate((block_data, class_data.reshape(1, 5, 62, 2 * fs)))
            save_data = np.concatenate((save_data, block_data.reshape(1, 40, 5, 62, 2 * fs)))
        np.save(os.path.join(output_dir, subname), save_data)


if __name__ == "__main__":
    segment_all_files()
