import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGVideoDataset(Dataset):
    """
    Dataset to load raw EEG + video latent pairs for EEG2Video Seq2Seq training.
    Assumes EEG files are stored as a numpy array (7, 40, 5, 2, 62, 200) and latents as (6, 40, 5, 4, 6, 36, 64)
    Returns EEG: (1, 62, 100) and video latent: (6, 4, 36, 64) per 1-second window.
    """
    def __init__(self, eeg_npy_path, latent_npy_path):
        self.eeg_data = np.load(eeg_npy_path)      # shape (7, 40, 5, 2, 62, 200)
        self.latent_data = np.load(latent_npy_path)  # shape (6, 40, 5, 4, 6, 36, 64)

        self.pairs = []
        for block in range(6):  # 6 blocks for training
            for concept in range(40):
                for rep in range(5):
                    for w in range(2):
                        self.pairs.append((block, concept, rep, w))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        b, c, r, w = self.pairs[idx]
        eeg = self.eeg_data[b, c, r, w]  # (62, 200)
        z0 = self.latent_data[b, c, r, :, :, :, :]  # (6, 4, 36, 64)
        eeg = torch.tensor(eeg, dtype=torch.float32)  # (62, 200)
        z0 = torch.tensor(z0, dtype=torch.float32)    # (6, 4, 36, 64)
        return eeg.unsqueeze(0), z0  # (1, 62, 200), (6, 4, 36, 64)
