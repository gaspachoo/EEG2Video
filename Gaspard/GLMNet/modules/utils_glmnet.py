import torch
import torch.nn as nn
from Gaspard.GLMNet.modules.models_paper import shallownet, mlpnet
from sklearn.preprocessing import StandardScaler 

class GLMNet(nn.Module):
    """ShallowNet (raw) + MLP (freq) → concat → FC"""
    def __init__(self, occipital_idx, out_dim: int = 40, emb_dim: int = 256): ### Use required embedding dim/2 here
        super().__init__()
        self.occipital_idx = list(occipital_idx)
        
        self.raw_global  = shallownet(emb_dim, 62, 200)  # (B,1,62,200)
        self.freq_local  = mlpnet(emb_dim, len(self.occipital_idx) * 5)  # (B,1,len(occipital*5)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim), nn.GELU(), nn.Linear(emb_dim, out_dim)
        )

    def forward(self, x_raw, x_feat):
        g_raw  = self.raw_global(x_raw)  # (B,emb)
        l_freq = self.freq_local(x_feat[:, self.occipital_idx, :])
        return self.fc(torch.cat([g_raw, l_freq], dim=1))

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