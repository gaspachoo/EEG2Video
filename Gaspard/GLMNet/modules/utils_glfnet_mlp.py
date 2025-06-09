import torch
import torch.nn as nn
from Gaspard.GLMNet.modules.models_paper import mlpnet
from Gaspard.GLMNet.modules.utils_glmnet import standard_scale_features

class GLFNetMLP(nn.Module):
    """MLP-based EEG classifier using global and occipital branches."""

    def __init__(self, occipital_idx, out_dim: int = 40, emb_dim: int = 64):
        super().__init__()
        self.occipital_idx = list(occipital_idx)
        self.globalnet = mlpnet(emb_dim, 62 * 5)
        self.localnet = mlpnet(emb_dim, len(self.occipital_idx) * 5)
        self.fc = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x_feat, return_features: bool = False):
        global_feat = self.globalnet(x_feat)
        local_feat = self.localnet(x_feat[:, self.occipital_idx, :])
        features = torch.cat([global_feat, local_feat], dim=1)
        if return_features:
            return features
        return self.fc(features)
