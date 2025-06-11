import torch
import torch.nn as nn
from .my_autoregressive_transformer import PositionalEncoding

class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # projetions
        self.input_proj  = nn.Linear(512, d_model)    # EEG embedding dim -> d_model
        self.output_proj = nn.Linear(9216, d_model)   # Video latent dim -> d_model
        self.out_linear  = nn.Linear(d_model, 9216)   # d_model -> video latent dim

        # positional encodings
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src, tgt):
        # src: (B, 7, 512), tgt: (B, 6, 9216)
        src = self.input_proj(src)   # (B,7,d_model)
        tgt = self.output_proj(tgt)  # (B,6,d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        # causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        memory = self.transformer_encoder(src)  # (B,7,d_model)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # (B,6,d_model)
        return self.out_linear(output)  # (B,6,9216)

    def forward_autoreg(self, src, seq_len=6):
        """Autoregressive inference without teacher forcing.

        Parameters
        ----------
        src : Tensor
            EEG embeddings of shape ``(B, 7, 512)``.
        seq_len : int, optional
            Number of latent frames to generate (default: 6).

        Returns
        -------
        Tensor
            Predicted latent sequence of shape ``(B, seq_len, 9216)``.
        """

        batch_size = src.size(0)

        # encode source sequence
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)  # (B,7,d_model)

        # first decoder token is zeros
        ys = torch.zeros(batch_size, 1, self.output_proj.out_features, device=src.device)
        preds = []

        for _ in range(seq_len):
            tgt = self.pos_decoder(ys)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(src.device)
            out = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            last = out[:, -1:, :]
            pred = self.out_linear(last)
            preds.append(pred)

            next_in = self.output_proj(pred)
            ys = torch.cat((ys, next_in), dim=1)

        return torch.cat(preds, dim=1)
