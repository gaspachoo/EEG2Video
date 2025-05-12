import os
import torch
import torch.nn.functional as F
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
    parser.add_argument('--batch_size', type=int, default=8)
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

    optimizer = torch.optim.Adam(pipeline.unet.parameters(), lr=args.lr)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    # Training loop (noise prediction objective)
    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        for z0, et in train_loader:
            z0 = z0.cuda()
            et = et.cuda().unsqueeze(1)  # shape (B,1,seq_dim)
            # sample noise and timestep
            noise = torch.randn_like(z0)
            timesteps = torch.randint(0, len(scheduler.timesteps), (z0.shape[0],), device='cuda')
            # add noise
            z_t = scheduler.add_noise(z0, noise, timesteps)
            # predict noise
            noise_pred = pipeline.unet(z_t, timesteps, encoder_hidden_states=et).sample
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validation
        val_loss = 0.0
        pipeline.unet.eval()
        with torch.no_grad():
            for z0, et in val_loader:
                z0 = z0.cuda()
                et = et.cuda().unsqueeze(1)
                noise = torch.randn_like(z0)
                timesteps = torch.randint(0, len(scheduler.timesteps), (z0.shape[0],), device='cuda')
                z_t = scheduler.add_noise(z0, noise, timesteps)
                noise_pred = pipeline.unet(z_t, timesteps, encoder_hidden_states=et).sample
                val_loss += F.mse_loss(noise_pred, noise).item()
        val_loss /= len(val_loader)
        pipeline.unet.train()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # Save fine-tuned UNet
    out_dir = f'{root}/checkpoints/tuneavideo_unet'
    os.makedirs(out_dir, exist_ok=True)
    pipeline.unet.save_pretrained(out_dir)
    print(f"UNet saved to {out_dir}")

if __name__ == '__main__':
    train()
