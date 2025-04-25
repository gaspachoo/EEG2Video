import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(310, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 10000),
            # nn.BatchNorm1d(50000),
            nn.ReLU(),
            # nn.Linear(10000, 10000),
            # nn.ReLU(),
            nn.Linear(10000, 77 * 768)
        )

    def forward(self, eeg):
        eeg_embeddings = self.mlp(eeg)
          # shape: (batch_size)
        return eeg_embeddings
    
#GLMNet EEG encoder with global + local branches
class GLMNetEncoder(nn.Module):
    def __init__(self, input_type='frequency', time_len=128, local_indices=None):
        """
        input_type: 'frequency' (e.g. DE/PSD) or 'raw' (EEG signals)
        time_len: number of time points (if using raw)
        """
        super().__init__()
        self.input_type = input_type
        self.local_indices = local_indices

        if input_type == 'raw':
            self.global_encoder = MLPEncoder(in_ch=62, time_len=time_len)
            local_ch = len(local_indices) if local_indices else 62
            self.local_encoder = ShallowNetEncoder(in_ch=local_ch, time_len=time_len)
            self.global_dim = self.global_encoder.flattened_dim
            self.local_dim = self.local_encoder.flattened_dim

        elif input_type == 'frequency':
            in_dim = 62 * 5  # e.g. 5 bands per channel
            self.global_encoder = nn.Sequential(
                nn.Linear(in_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU()
            )
            local_dim = len(local_indices) * 5 if local_indices else in_dim
            self.local_encoder = nn.Sequential(
                nn.Linear(local_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU()
            )
            self.global_dim = 2048
            self.local_dim = 2048

        else:
            raise ValueError("input_type must be 'raw' or 'frequency'")

        # Final projection to (77, 768)
        self.projection = nn.Linear(self.global_dim + self.local_dim, 77 * 768)
            
    def forward(self, x):
        if self.input_type == 'frequency':
            # x: (B, 62*5)
            global_feat = self.global_encoder(x)

            if self.local_indices:
                x_channels = x.view(x.shape[0], 62, -1)  # (B, 62, 5)
                x_local = x_channels[:, self.local_indices, :].reshape(x.shape[0], -1)
            else:
                x_local = x

            local_feat = self.local_encoder(x_local)

        else:  # raw input
            # x: (B, 62, T)
            global_feat = self.global_encoder(x)

            if self.local_indices:
                x_local = x[:, self.local_indices, :]
            else:
                x_local = x

            local_feat = self.local_encoder(x_local)

        combined = torch.cat([global_feat, local_feat], dim=1)
        out = self.projection(combined)
        return out.view(-1, 77, 768)
    
    def freeze_parts(self, freeze_global=True, freeze_local=False, freeze_projection=False):
        for param in self.global_encoder.parameters():
            param.requires_grad = not freeze_global
        for param in self.local_encoder.parameters():
            param.requires_grad = not freeze_local
        for param in self.projection.parameters():
            param.requires_grad = not freeze_projection



class MLPEncoder(nn.Module):
    def __init__(self, input_dim=62*5, num_classes=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class ShallowNetEncoder(nn.Module):
    def __init__(self, in_ch=62, time_len=200, dropout=0.6):
        super().__init__()
        self.net = nn.Sequential(
            # 1. Temporal convolution (like original ShallowNet)
            nn.Conv2d(1, 32, kernel_size=(1, 13), padding=(0, 6)),
            nn.BatchNorm2d(32),
            nn.ELU(),

            # 2. Spatial convolution across channels
            nn.Conv2d(32, 64, kernel_size=(in_ch, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),

            # 3. Temporal compression
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2)),

            # 4. Dropout
            nn.Dropout(dropout),

            nn.Flatten()
        )

    def forward(self, x):  # x: (B, C, T)
        x = x.unsqueeze(1)  # → (B, 1, C, T)
        return self.net(x)


class MLPEncoder_feat(nn.Module):
    """
    Remove last layer
    """
    def __init__(self, input_dim=62*5, feat_dim=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        self.output_dim = feat_dim  # 128

    def forward(self, x):
        return self.feature_extractor(x.view(x.size(0), -1))