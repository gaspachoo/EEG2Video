import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    """Simple dataset returning EEG embeddings and corresponding video latents."""

    def __init__(self, eeg_array: np.ndarray, video_array: np.ndarray):
        if eeg_array.shape[0] != video_array.shape[0]:
            raise ValueError("EEG and video arrays must have the same first dimension")
        self.eeg = torch.tensor(eeg_array, dtype=torch.float32)
        self.video = torch.tensor(video_array, dtype=torch.float32)

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, idx: int):
        return self.eeg[idx], self.video[idx]
