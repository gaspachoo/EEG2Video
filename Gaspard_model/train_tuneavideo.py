import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler
#from diffusers.pipeline_utils import DiffusionPipeline
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
import wandb


# ---------------- Dataset ----------------
class EEGVideoDataset(Dataset):
    def __init__(self, zhat_dir: str, sem_dir: str):
        # ---- video latents z_hat ----
        z_blocks = []
        for fname in sorted(os.listdir(zhat_dir)):
            if not fname.endswith('.npy'): continue
            arr = np.load(os.path.join(zhat_dir, fname))  # (200,6,4,36,64)
            z_blocks.append(arr)
        # concat over blocks → (7*200,6,4,36,64)
        self.z_hat = np.concatenate(z_blocks, axis=0)
        
        # ---- semantic embeddings e_t ----
        e_blocks = []
        for fname in sorted(os.listdir(sem_dir)):
            if not fname.endswith('.npy'): continue
            emb = np.load(os.path.join(sem_dir, fname))  # (200, 77*768)
            e_blocks.append(emb)
        # concat → (7*200,77*768) == (1400,59136)
        self.e_t = np.concatenate(e_blocks, axis=0)
        
        assert self.z_hat.shape[0] == self.e_t.shape[0], \
            f"Got {self.z_hat.shape[0]} ẑ vs {self.e_t.shape[0]} eₜ samples"
        
    def __len__(self):
        return self.z_hat.shape[0]
    
    def __getitem__(self, idx):
        # z0 : (6,4,36,64), e_t : (77*768,)
        z0 = torch.from_numpy(self.z_hat[idx]).float()
        et = torch.from_numpy(self.e_t[idx]).float()
        return z0, et


# ---------------- Training Script ----------------
def train():
    import argparse
    parser = argparse.ArgumentParser()
    root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
    parser.add_argument('--zhat_dir', type=str, default=f"{root}/data/Predicted_latents")
    parser.add_argument('--sem_dir', type=str, default=f"{root}/data/Semantic_embeddings")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="eeg2video-tuneavideo", config=vars(args))

    # Dataset
    dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Load pretrained components
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae").cuda()
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    unet = UNet3DConditionModel.from_pretrained_2d(f"{root}/EEG2Video/EEG2Video_New/Generation/stable-diffusion-v1-4", subfolder="unet").cuda()
    scheduler = PNDMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="scheduler")

    # Construct pipeline
    pipeline = TuneAVideoPipeline(vae=vae, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
    pipeline.unet.train()

    # Projection EEG → cross_attention_dim
    sem_dim   = dataset[0][1].shape[-1]  # 59136
    cross_dim = unet.config.cross_attention_dim  # 768
    proj_eeg  = nn.Linear(sem_dim, cross_dim).cuda()
    
        
    
    optimizer = torch.optim.Adam(list(pipeline.unet.parameters()) + list(proj_eeg.parameters()),
                                 lr=args.lr)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for epoch in range(1, args.epochs + 1):
        pipeline.unet.train()
        train_loss = 0

        for z0, et in train_loader:
            # 1) Latents : (B,6,4,H,W) → (B,4,6,H,W)
            print("▷ z0 before permute:", z0.shape)  # (B,6,4,36,64)
            z0 = z0.to(device).permute(0, 2, 1, 3, 4)
            print("▷ z0 after  permute:", z0.shape)  # (B,4,6,36,64)

            # 2) EEG embedding : (B, sem_dim) → (B,1,sem_dim)
            et = et.to(device).unsqueeze(1)
            print("▷ et before proj:", et.shape)     # (B,1,59136)

            # 3) Projection → (B,1,cross_dim)
            et = proj_eeg(et)  # applique linear sur la dernière dim
            print("▷ et after  proj:", et.shape)     # (B,1,768)

            # bruit & timesteps
            noise     = torch.randn_like(z0)
            timesteps = torch.randint(0, len(scheduler.timesteps), (z0.size(0),), device=device)
            print("▷ timesteps:", timesteps.shape)

            # add noise
            z_t = scheduler.add_noise(z0, noise, timesteps)
            print("▷ z_t:", z_t.shape)

            # forward UNet
            out = pipeline.unet(z_t, timesteps, encoder_hidden_states=et)
            print("▷ UNet output sample:", out.sample.shape)  # (B,4,6,36,64)
            noise_pred = out.sample

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation...
        # (même boucle sans backward, en .eval())

        print(f"Epoch {epoch} train_loss={train_loss/len(train_loader):.4f}")

    # Save
    pipeline.unet.save_pretrained(f"{root}/checkpoints/tuneavideo_unet")
    print("UNet saved.")

if __name__ == '__main__':
    train()
