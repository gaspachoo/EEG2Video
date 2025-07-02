import numpy as np


def load_aligned_latents(eeg_path: str, video_path: str):
    """Load EEG and video latents and align them by length.

    Parameters
    ----------
    eeg_path : str
        Path to the EEG latent .npy file.
    video_path : str
        Path to the video latent .npy file.

    Returns
    -------
    tuple of np.ndarray
        Tuple `(eeg_latent, video_latent)` trimmed to the same number of samples.
    """
    eeg_latent = np.load(eeg_path)
    video_latent = np.load(video_path)

    n = min(len(eeg_latent), len(video_latent))
    return eeg_latent[:n], video_latent[:n]

