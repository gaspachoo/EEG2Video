import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import wandb
import argparse
from models.my_autoregressive_transformer import PositionalEncoding

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


def parse_args():
    parser = argparse.ArgumentParser()
    root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
    parser.add_argument('--sub_emb',       type=str,
                        default=f"{root}/data/EEG_embeddings_sw/sub3.npy",
                        help='EEG embeddings (.npy) path')
    parser.add_argument('--video_dir',     type=str,
                        default=f"{root}/data/Video_latents",
                        help='Directory with block{block_id}_latents.npy')
    parser.add_argument('--save_path',     type=str,
                        default=f"{root}/Gaspard_model/checkpoints/seq2seq_sw/",
                        help='Where to save models')
    parser.add_argument('--epochs',        type=int,   default=200)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--use_wandb',     action='store_true', help='Log to wandb')
    return parser.parse_args()


def prepare_data(sub_emb, video_dir, block_id):
    # Load all embeddings: shape (7*40*5*7, 512)
    z_all = np.load(sub_emb)
    z_all = z_all.reshape(7, 40, 5, 7, 512)
    # Select block and reshape to (200, 7, 512)
    z_block = z_all[block_id]            # (40,5,7,512)
    z_hat   = z_block.reshape(-1, 7, 512) # (200,7,512)

    # Load video latents: shape (200,6,4,36,64)
    vid_path = os.path.join(video_dir, f'block{block_id}_latents.npy')
    z0 = np.load(vid_path)
    z0 = z0.reshape(z0.shape[0], 6, -1)  # (200,6,9216)

    # Tensors
    eeg_tensor = torch.tensor(z_hat, dtype=torch.float32)
    vid_tensor = torch.tensor(z0,    dtype=torch.float32)
    return TensorDataset(eeg_tensor, vid_tensor)


def train_seq2seq(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Using device: {device}")

    for block_id in range(7):
        # Prepare data per block
        dataset = prepare_data(args.sub_emb, args.video_dir, block_id)
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size)

        # Model, optimizer, scheduler
        model     = Seq2SeqTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

        # WandB init
        if args.use_wandb:
            wandb.init(project='eeg2video-Seq2Seq-SW', name=f'Seq2Seq_sw_block{block_id}', config=vars(args))
            wandb.watch(model, log='all')

        # Training loop
        for epoch in range(1, args.epochs+1):
            model.train()
            train_loss, train_cos, count = 0, 0, 0
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
                count += 1
            train_loss /= count
            train_cos  /= count

            # Validation
            model.eval()
            val_loss, val_cos, count = 0, 0, 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    out = model(src, tgt)
                    val_loss += criterion(out, tgt).item()
                    val_cos  += F.cosine_similarity(out, tgt, dim=-1).mean().item()
                    count += 1
            val_loss /= count
            val_cos  /= count

            # Log
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_cosine': train_cos,
                    'val_cosine': val_cos,
                    'lr': optimizer.param_groups[0]['lr']
                })

        # Save model per block
        ckpt_path = os.path.join(args.save_path, f'seq2seq_sw_block{block_id}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Block {block_id} model saved to {ckpt_path}")

        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    train_seq2seq(parse_args())
