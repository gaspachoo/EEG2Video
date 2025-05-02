import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EEG2VideoDataset(Dataset):
    """
    Dataset to load EEG-video latent pairs for EEG2Video Seq2Seq training.
    Assumes files are named like sub3_b{block}_c{concept}_r{rep}_w{w}.npz
    and stored in a directory. Each (block, concept, rep) has 2 windows.
    We group 7 EEG embeddings of shape (1, 512) into (7, 512) per sample.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_map = self._group_files()

    def _group_files(self):
        """Group files by (block, concept, rep) triplets."""
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith(".npz")]
        triplet_map = {}
        for f in all_files:
            parts = f.split("_")
            key = "_".join(parts[1:4])  # b{block}_c{concept}_r{rep}
            triplet_map.setdefault(key, []).append(f)
        return [v for v in triplet_map.values() if len(v) == 2]  # keep only complete pairs

    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, idx):
        paths = sorted(self.file_map[idx])  # ensure w0 then w1
        eegs = []
        for f in paths:
            data = np.load(os.path.join(self.data_dir, f))
            eegs.append(data['eeg'].squeeze(0))  # (512,)
            if 'z0' in locals():
                pass  # z0 already loaded from w0
            else:
                z0 = data['z0']  # (6, 9216)
        eeg_seq = np.stack(eegs, axis=0)  # (2, 512)
        return torch.tensor(eeg_seq, dtype=torch.float32), torch.tensor(z0, dtype=torch.float32)
