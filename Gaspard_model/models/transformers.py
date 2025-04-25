import torch
import torch.nn as nn

class EEG2VideoTransformer(nn.Module):
    def __init__(self, embed_dim=768, seq_len=77, nhead=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))  # Learned positional embeddings

    def forward(self, eeg_embeddings):
        """
        Input: eeg_embeddings: (N, 77, 768)
        Output: predicted latent features: (N, 77, 768)
        """
        x = eeg_embeddings + self.pos_embed  # Add position encoding
        x = self.transformer(x)
        return x  # Output used as latent prediction \hat{z0}
