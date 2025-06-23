import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    """EEG/video paired dataset."""

    def __init__(self, eeg: torch.Tensor, video: torch.Tensor) -> None:
        super().__init__()
        self.eeg = eeg
        self.video = video

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, item: int):
        return self.eeg[item], self.video[item]
