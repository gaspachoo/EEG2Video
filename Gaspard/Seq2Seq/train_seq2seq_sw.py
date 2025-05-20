import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import wandb
import argparse
from models.transformer import Seq2SeqTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sub_emb',       type=str,
                        default="./data/EEG_embeddings_sw/sub3.npy",
                        help='EEG embeddings (.npy) path')
    parser.add_argument('--video_dir',     type=str,
                        default="./data/Video_latents",
                        help='Directory with block{block_id}_latents.npy')
    parser.add_argument('--save_path',     type=str,
                        default="./Gaspard/checkpoints/seq2seq_sw/",
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
