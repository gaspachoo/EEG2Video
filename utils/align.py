import numpy as np

__all__ = [
    "stack_eeg_windows",
    "load_aligned_latents",
]


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

def stack_eeg_windows(eeg_windows: np.ndarray, start: int, *, windows_per_clip: int = 7, video_latents: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Stack 6 consecutive EEG windows and optionally return the matching video latent.

    Parameters
    ----------
    eeg_windows : np.ndarray
        Array of shape ``(N, dim)`` containing EEG embeddings for all windows.
    start : int
        Starting index of the sequence within ``eeg_windows``.
    windows_per_clip : int, optional
        Number of windows in each video clip. Defaults to ``7`` which
        corresponds to 500 ms windows extracted from 2-second clips.
    video_latents : np.ndarray | None, optional
        Array of shape ``(M, dim)`` with one latent per clip. If provided,
        the latent of the clip containing ``start`` is returned alongside
        the stacked windows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | np.ndarray
        Stacked windows of shape ``(6, dim)``. If ``video_latents`` is not
        ``None``, the corresponding latent is also returned.

    Raises
    ------
    ValueError
        If the requested sequence extends beyond array bounds or spans
        multiple clips.
    """
    n = 6
    end = start + n
    if end > len(eeg_windows):
        raise ValueError("Sequence exceeds available windows")

    clip_idx = start // windows_per_clip
    if end - 1 >= (clip_idx + 1) * windows_per_clip:
        raise ValueError("Sequence crosses clip boundary")

    stacked = eeg_windows[start:end]
    if video_latents is None:
        return stacked
    if clip_idx >= len(video_latents):
        raise ValueError("No matching video latent for the selected clip")
    return stacked, video_latents[clip_idx]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((14, 256))  # two clips of seven windows
    vid = rng.standard_normal((2, 256))
    x, y = stack_eeg_windows(eeg, 1, video_latents=vid)
    print("EEG stack shape:", x.shape)
    print("Latent shape:", y.shape)
    assert x.shape == (6, 256)

