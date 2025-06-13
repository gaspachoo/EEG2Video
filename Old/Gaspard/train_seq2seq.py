import torch
import torch.nn as nn
import math
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from EEG2Video.Seq2Seq.my_autoregressive_transformer import PositionalEncoding

class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(512, d_model)   # EEG embedding dim → d_model
        self.output_proj = nn.Linear(9216, d_model)  # Latent dim → d_model
        self.out_linear = nn.Linear(d_model, 9216)   # Predicted output dim

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, src, tgt):
        # src: (B, 7, 512), tgt: (B, 6, 9216)
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
     #"/Documents/School/Centrale Med/2A/SSE/EEG2Video"
    parser.add_argument('--sub_emb', type=str, default="./data/EEG_embeddings/sub3.npy", help='Path to EEG z_hat file')
    parser.add_argument('--video_dir', type=str, default='./data/Video_latents', help='Path to block-wise video latent files')
    parser.add_argument('--save_path', type=str, default='./EEG2Video/checkpoints/seq2seq/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true', help='Log metrics to wandb')
    return parser.parse_args()

def prepare_data(sub_emb, video_dir, block_id):
    z_hat = np.load(sub_emb)  # (7*40*5*2, 512)

    #print(f"Loaded EEG: {sub_emb}, shape={z_hat.shape}")
    z_hat = z_hat.reshape(7, 40, 5, 2, 512)
    z_hat = z_hat[..., 0, :].transpose(1,2,0,3)  # Keep only first 1s EEG → shape: (7, 40, 5, 512) -> transpose to (40, 5, 7, 512)
    z_hat = z_hat.reshape(-1, 7, 512)  # (200, 7, 512)
    

    vid_path = os.path.join(video_dir, f'block{block_id}_latents.npy')  # Match same block as EEG
    z0 = np.load(vid_path)  # (200, 6, 4, 36, 64)
    #print(f"Loaded Latents: block{block_id}_latents.npy, shape={z0.shape}")
    z0 = z0.reshape(z0.shape[0], 6, -1) # (200, 6, 9216)

    eeg_tensor = torch.tensor(z_hat, dtype=torch.float32)
    vid_tensor = torch.tensor(z0, dtype=torch.float32)

    #print(f"Final EEG tensor shape: {eeg_tensor.shape}")
    #print(f"Final Video tensor shape: {vid_tensor.shape}")

    return TensorDataset(eeg_tensor, vid_tensor)


def cosine_similarity(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (pred * target).sum(dim=-1).mean().item()


def train_seq2seq(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    block_id = 0
    
    for block_id in tqdm(range(7),desc="Loading blocks"):
        dataset = prepare_data(args.sub_emb, args.video_dir, block_id=block_id)

        # Split
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
     
        model = Seq2SeqTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

        if args.use_wandb:
            wandb.init(project="eeg2video-Seq2Seq", name=f"Seq2Seq_{block_id}", config=vars(args))
            wandb.watch(model, log="all")

        for epoch in tqdm(range(1, args.epochs + 1),desc=f"Epochs for block {block_id}"):
            # Training
            model.train()
            train_loss = 0
            train_cos = 0
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                out = model(src, tgt)
                loss = criterion(out, tgt)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                train_cos += F.cosine_similarity(out, tgt, dim=-1).mean().item()
                
            
            train_loss /= len(train_loader)
            train_cos /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            val_cos = 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    out = model(src, tgt)
                    val_loss += criterion(out, tgt).item()
                    val_cos += F.cosine_similarity(out, tgt, dim=-1).mean().item()

            val_loss /= len(val_loader)
            val_cos /= len(val_loader)

            #print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Cos: {val_cos:.4f}")
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "cosine_similarity": train_cos,
                    "val_cosine_similarity": val_cos,
                    "lr": optimizer.param_groups[0]['lr']
                })

            
        wandb.finish()
        
        torch.save(model.state_dict(), os.path.join(args.save_path,f"seq2seq_block{block_id}.pth"))
        print(f"Model saved to {args.save_path}seq2seq_block{block_id}.pth")

if __name__ == '__main__':
    train_seq2seq(parse_args())
