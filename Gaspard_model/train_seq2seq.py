import torch
import torch.nn as nn
import math
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(512, d_model)   # EEG embedding dim → d_model
        self.output_proj = nn.Linear(256, d_model)  # Latent dim → d_model
        self.out_linear = nn.Linear(d_model, 256)   # Predicted output dim

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src, tgt):
        # src: (B, 7, 512), tgt: (B, 6, 256)
        src = self.input_proj(src)          # (B, 7, d_model)
        tgt = self.output_proj(tgt)         # (B, 6, d_model)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)  # (6, 6)

        memory = self.transformer_encoder(src)        # (B, 7, d_model)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # (B, 6, d_model)

        return self.out_linear(output)  # (B, 6, 256)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/EEG2Video" #"/Documents/School/Centrale Med/2A/SSE/EEG2Video"
    parser.add_argument('--eeg_dir', type=str, default=f"{root}/data/EEG_embeddings/", help='Path to EEG z_hat files')
    parser.add_argument('--video_dir', type=str, default=f'{root}/data/Video_latents', help='Path to block-wise video latent files')
    parser.add_argument('--save_path', type=str, default=f'./checkpoints/seq2seq/seq2seq_transformer.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true', help='Log metrics to wandb')
    return parser.parse_args()

def prepare_data(eeg_dir, video_dir):
    all_eeg = []
    for subj_file in sorted(os.listdir(eeg_dir)):
        if not subj_file.endswith('.npy'):
            continue
        eeg_path = os.path.join(eeg_dir, subj_file)
        z_hat = np.load(eeg_path)  # (7, 40, 5, 2, 512)
        print(f"Loaded EEG: {subj_file}, shape={z_hat.shape}")
        z_hat = z_hat.reshape(7, 40, 5, 2, 512)
        z_hat = z_hat[..., 0, :]  # Keep only first 1s EEG → shape: (7, 40, 5, 512)
        z_hat = z_hat.reshape(-1, 7, 512)  # (200, 7, 512)
        all_eeg.append(z_hat)

    vid_path = os.path.join(video_dir, 'block0_latents.npy')  # Match same block as EEG
    z0 = np.load(vid_path)  # (200, 6, 4, 36, 64)
    print(f"Loaded Latents: block0_latents.npy, shape={z0.shape}")
    z0 = z0.reshape(z0.shape[0], 6, -1)[:, :, :256]  # (200, 6, 256)

    eeg_tensor = torch.tensor(np.concatenate(all_eeg), dtype=torch.float32)
    vid_tensor = torch.tensor(z0, dtype=torch.float32)

    print(f"Final EEG tensor shape: {eeg_tensor.shape}")
    print(f"Final Video tensor shape: {vid_tensor.shape}")

    return TensorDataset(eeg_tensor, vid_tensor)

def train_seq2seq():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = prepare_data(args.eeg_dir, args.video_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Seq2SeqTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if args.use_wandb:
        wandb.init(project="eeg2video-Seq2eq", name="Seq2Seq", config=vars(args))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0
        for src, tgt in tqdm(dataloader, desc=f"Epoch {epoch}"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src, tgt)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_loss})

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    train_seq2seq()
