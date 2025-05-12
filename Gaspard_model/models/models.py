import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#GLMNet EEG encoder with global + local branches
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
    
    
class Seq2SeqTransformer_beta(nn.Module):
    def __init__(self, 
                 num_encoder_layers=2, 
                 num_decoder_layers=4, 
                 emb_size=512, 
                 nhead=8, 
                 src_seq_len=7, 
                 tgt_seq_len=6, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        super().__init__()
        
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.emb_size = emb_size
        
        # Positional encoding
        self.src_pos_emb = nn.Parameter(torch.randn(1, src_seq_len, emb_size))
        self.tgt_pos_emb = nn.Parameter(torch.randn(1, tgt_seq_len, emb_size))
        
        # Transformer core
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important : input (batch, seq, feature)
        )

        # Projection head
        self.output_proj = nn.Linear(emb_size, 256)  # from emb_size (512) to latent_dim (256)

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, 7, 512)
        Returns:
            out: Tensor of shape (batch_size, 6, 256)
        """
        batch_size = src.size(0)
        
        # Add position encoding to src
        src = src + self.src_pos_emb  # (batch_size, 7, 512)
        
        # Prepare target (tgt) tokens as zeros initially
        tgt = torch.zeros(batch_size, self.tgt_seq_len, self.emb_size, device=src.device)
        tgt = tgt + self.tgt_pos_emb  # (batch_size, 6, 512)

        # Transformer forward
        memory = self.transformer.encoder(src)  # Encode EEG
        output = self.transformer.decoder(tgt, memory)  # Decode into video latents
        
        # Project output to latent space
        out = self.output_proj(output)  # (batch_size, 6, 256)
        
        return out

class GLMNetFeatureExtractor(nn.Module):
    def __init__(self, g_enc, l_enc, output_dim=512):
        super().__init__()
        self.g = g_enc
        self.l = l_enc
        self.project = nn.Linear(g_enc.flattened_dim + l_enc.output_dim, output_dim)

    def forward(self, xr, xf):
        eeg_feat = self.g(xr)
        de_feat = self.l(xf.view(xf.size(0), -1))
        concat = torch.cat([eeg_feat, de_feat], dim=1)  # (batch, 4096)
        return self.project(concat)  # (batch, 512)

    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, eeg_dim=512, latent_dim=256, num_eeg_tokens=7, num_video_tokens=6):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.latent_dim = latent_dim

        self.pos_enc = PositionalEncoding(eeg_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=eeg_dim, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)

        self.latent_query = nn.Parameter(torch.randn(num_video_tokens, latent_dim))  # (6, 256)
        self.eeg_to_latent = nn.Linear(eeg_dim, latent_dim)

    def forward(self, eeg):  # eeg: (B, 7, 512)
        eeg = self.pos_enc(eeg)                    # Add position
        eeg = eeg.transpose(0, 1)                  # (7, B, 512)
        memory = self.encoder(eeg)                 # (7, B, 512)
        memory = self.eeg_to_latent(memory)        # (7, B, 256)

        tgt = self.latent_query.unsqueeze(1).expand(-1, eeg.size(1), -1)  # (6, B, 256)
        output = self.decoder(tgt, memory)        # (6, B, 256)
        return output.transpose(0, 1)             # (B, 6, 256)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)