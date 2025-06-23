import torch
import torch.nn as nn

from EEG2Video.GLMNet.modules.models_paper import shallownet, mlpnet

class MyEEGNetEmbedding(nn.Module):
    """EEGNet feature extractor used before the transformer."""

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


class ShallowNetEmbedding(nn.Module):
    """Wraps the shallownet model used in GLMNet."""

    def __init__(self, d_model: int = 128, C: int = 62, T: int = 100,
                 weights_path: str | None = None) -> None:
        super().__init__()
        self.model = shallownet(out_dim=d_model, C=C, T=T)
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class MLPNetEmbedding(nn.Module):
    """Wraps the mlp model used in GLMNet."""

    def __init__(self, d_model: int = 128, C: int = 62, occipital_idx: list = list(range(50, 62)),
                 weights_path: str | None = None) -> None:
        super().__init__()
        self.occipital_idx = occipital_idx
        self.model = mlpnet(out_dim=d_model, input_dim=len(occipital_idx) * 5)
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.occipital_idx, :]
        return self.model(x)



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
    def __init__(self, d_model: int = 512, eeg_encoder: str = "eegnet",
                 encoder_ckpt: str | None = None, C : int | None = None, T : int | None = None) -> None:
        super().__init__()
        self.img_embedding = nn.Linear(4 * 36 * 64, d_model)
        self.T = T
        self.C = C
        if eeg_encoder == "shallownet":
            assert type(C) == int and type(T) == int
            self.eeg_embedding = ShallowNetEmbedding(
                d_model=d_model,
                C=self.C,
                T=self.T,
                weights_path=encoder_ckpt,
            )
        elif eeg_encoder == 'mlpnet':
            assert type(self.C) == int and type(self.T) == int
            assert self.T==5, "You should use a features file for eeg"
            self.eeg_embedding = MLPNetEmbedding(
                d_model=d_model,
                C=self.C,
                weights_path=encoder_ckpt,
            )
            
        elif eeg_encoder == "eegnet":
            self.eeg_embedding = MyEEGNetEmbedding(d_model=d_model)
        else:
            raise ValueError("This encoder has not been implemented yet, check your typing, choices are ['shallownet','eegnet']")

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

        Raises
        ------
        ValueError
            If ``src`` is not of shape ``(batch, 7, 62, T)``.
        """

        # Validate EEG input dimensions
        if src.size(1) != 7 or src.size(2) != 62 or src.size(3) != self.T:
            raise ValueError(f"Expected src shape (B,7,62,{self.T}) but got {tuple(src.shape)}")

        # Reshape EEG input for embedding while remaining robust to non-contiguous tensors
        batch_size, seq_len, _, _ = src.shape
        if isinstance(self.eeg_embedding, MLPNetEmbedding):
            src = src.reshape(batch_size * seq_len, 62, self.T)
        else:
            src = src.reshape(batch_size * seq_len, 1, 62, self.T)
        src = src.reshape(batch_size, seq_len, -1)

        # Flatten video latents before linear embedding
        tgt = tgt.reshape(tgt.size(0), tgt.size(1), -1)
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
