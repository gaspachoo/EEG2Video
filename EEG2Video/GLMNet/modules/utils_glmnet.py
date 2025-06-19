import torch
import torch.nn as nn
from EEG2Video.GLMNet.modules.models_paper import shallownet, mlpnet
from sklearn.preprocessing import StandardScaler
import numpy as np

class GLMNet(nn.Module):
    """ShallowNet (raw) + MLP (freq) → concat → FC."""

    def __init__(self, occipital_idx, T:int, out_dim: int = 40, emb_dim: int = 512):
        """Construct the GLMNet model.

        Parameters
        ----------
        occipital_idx : iterable
            Indexes of occipital channels used for the local branch.
        out_dim : int
            Dimension of the classification output.
        emb_dim : int
            Dimension of the intermediate embeddings (each branch outputs
            ``emb_dim`` features).
        T : int
            Number of temporal samples of the raw EEG. This value can vary
            depending on the dataset.
        """
        super().__init__()
        self.occipital_idx = list(occipital_idx)

        # Global branch processing raw EEG
        self.raw_global = shallownet(emb_dim, 62, T)
        # Local branch processing spectral features
        self.freq_local = mlpnet(emb_dim, len(self.occipital_idx) * 5)

        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, out_dim),
        )

    def forward(self, x_raw, x_feat, return_features: bool = False):
        """Forward pass of the network.

        Parameters
        ----------
        x_raw : torch.Tensor
            Raw EEG of shape ``(B, 1, 62, T)``.
        x_feat : torch.Tensor
            Spectral features of shape ``(B, 62, 5)``.
        return_features : bool, optional
            If ``True`` returns the concatenated features before the final
            projection layer. Defaults to ``False``.
        """

        g_raw = self.raw_global(x_raw)
        l_freq = self.freq_local(x_feat[:, self.occipital_idx, :])
        features = torch.cat([g_raw, l_freq], dim=1)

        if return_features:
            return features

        return self.fc(features)

def standard_scale_features(X, scaler=None, return_scaler=False):
    """Scale features with ``StandardScaler``.

    Parameters
    ----------
    X : np.ndarray
        Array of shape ``(N, ...)`` to scale.
    scaler : sklearn.preprocessing.StandardScaler or None
        If ``None`` a new scaler is fitted on ``X``. Otherwise ``X`` is
        transformed using the provided scaler.
    return_scaler : bool, optional
        Whether to return the fitted scaler.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as ``X``.
    sklearn.preprocessing.StandardScaler, optional
        Returned only if ``return_scaler`` is ``True``.
    """

    orig_shape = X.shape[1:]
    X_2d = X.reshape(len(X), -1)

    if scaler is None:
        scaler = StandardScaler().fit(X_2d)

    X_scaled = scaler.transform(X_2d).reshape((len(X),) + orig_shape)

    if return_scaler:
        return X_scaled, scaler
    return X_scaled


def compute_raw_stats(X: np.ndarray):
    """Compute per-channel mean and std from training data."""
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2)) + 1e-6
    return mean, std


def normalize_raw(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize raw EEG with provided statistics."""
    return (X - mean[None, :, None]) / std[None, :, None]
