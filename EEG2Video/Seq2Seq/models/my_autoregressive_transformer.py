import torch
import torch.nn as nn

class MyEEGNetEmbedding(nn.Module):
    """EEG feature extractor used before the transformer."""

    def __init__(self, d_model: int = 128, C: int = 62, T: int = 100,
                 F1: int = 16, D: int = 4, F2: int = 16, cross_subject: bool = False) -> None:
        super().__init__()
        self.drop_out = 0.25 if cross_subject else 0.5

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, F1, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )

        self.embedding = nn.Linear(48, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    """Injects positional information into token embeddings."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class myTransformer(nn.Module):
    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.img_embedding = nn.Linear(4 * 36 * 64, d_model)
        self.eeg_embedding = MyEEGNetEmbedding(d_model=d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=4,
        )

        self.positional_encoding = PositionalEncoding(d_model, dropout=0.0)
        self.txtpredictor = nn.Linear(d_model, 13)
        self.predictor = nn.Linear(d_model, 4 * 36 * 64)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        src : Tensor
            EEG input of shape ``(batch, 7, 62, 100)`` which becomes ``(batch, 7, d_model)`` after embedding.
        tgt : Tensor
            Video latent sequence of shape ``(batch, 7, 4, 36, 64)`` where the first frame is padding.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Text logits and latent predictions of shape ``(batch, 7, 4, 36, 64)``.
            Only the last 6 frames are used for the loss.
        """
        src = self.eeg_embedding(src.view(src.size(0) * src.size(1), 1, 62, 100))
        src = src.view(tgt.size(0), 7, -1)

        tgt = tgt.view(tgt.size(0), tgt.size(1), -1)
        tgt = self.img_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        memory = self.transformer_encoder(src)

        new_tgt = torch.zeros(tgt.size(0), 1, tgt.size(2), device=tgt.device)
        for i in range(6):
            out = self.transformer_decoder(new_tgt, memory, tgt_mask=tgt_mask[: i + 1, : i + 1])
            new_tgt = torch.cat((new_tgt, out[:, -1:, :]), dim=1)

        memory = memory.mean(dim=1)
        preds = self.predictor(new_tgt).view(new_tgt.size(0), new_tgt.size(1), 4, 36, 64)
        return self.txtpredictor(memory), preds
