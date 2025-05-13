import os
# Reduce CUDA allocation fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, PNDMScheduler
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from torch.cuda.amp import autocast, GradScaler

# Disable TF32 to improve FP16 stability
torch.backends.cuda.matmul.allow_tf32 = False

# ---------------- Dataset ----------------
class EEGVideoDataset(torch.utils.data.Dataset):
    def __init__(self, zhat_dir: str, sem_dir: str):
        import numpy as np
        z_blocks, e_blocks = [], []
        for fname in sorted(os.listdir(zhat_dir)):
            if fname.endswith('.npy'):
                z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        for fname in sorted(os.listdir(sem_dir)):
            if fname.endswith('.npy'):
                e_blocks.append(np.load(os.path.join(sem_dir, fname)))
        self.z_hat = np.concatenate(z_blocks, axis=0)
        self.e_t   = np.concatenate(e_blocks, axis=0)
        assert self.z_hat.shape[0] == self.e_t.shape[0]
    def __len__(self): return self.z_hat.shape[0]
    def __getitem__(self, idx):
        z0 = torch.from_numpy(self.z_hat[idx]).float()
        et = torch.from_numpy(self.e_t[idx]).float()
        return z0, et

# ---------------- Training Script ----------------
def train():
    import argparse
    parser = argparse.ArgumentParser()
    root = os.path.expanduser('~/EEG2Video')
    parser.add_argument('--zhat_dir',   type=str, default=f"{root}/data/Predicted_latents")
    parser.add_argument('--sem_dir',    type=str, default=f"{root}/data/Semantic_embeddings")
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--accum_steps',type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
    train_n = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_n, len(dataset) - train_n])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # Models
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to(device)
    vae.requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)

    unet = UNet3DConditionModel.from_pretrained_2d(f"{root}/EEG2Video/EEG2Video_New/Generation/stable-diffusion-v1-4", subfolder="unet").cuda()
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    scheduler = PNDMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='scheduler')
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    # Include text_encoder in pipeline instantiation
    pipeline = TuneAVideoPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler
    )
    pipeline.enable_vae_slicing()

    # Projection EEG → cross_attention_dim
    sem_dim   = dataset[0][1].shape[-1]
    cross_dim = unet.config.cross_attention_dim
    proj_eeg  = nn.Linear(sem_dim, cross_dim).to(device)

    # Optimizer
    parameters = list(unet.parameters()) + list(proj_eeg.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    # AMP scaler
    scaler = GradScaler()

    # Training
    for epoch in range(1, args.epochs + 1):
        unet.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (z0, et) in enumerate(train_loader):
            z0 = z0.to(device).permute(0,2,1,3,4)
            et = et.to(device).unsqueeze(1)

            with autocast():
                et_proj = proj_eeg(et)
                noise   = torch.randn_like(z0)
                timesteps = torch.randint(0, len(scheduler.timesteps), (z0.size(0),), device=device)
                z_t     = scheduler.add_noise(z0, noise, timesteps)
                try:
                    out = pipeline.unet(z_t, timesteps, encoder_hidden_states=et_proj)
                except RuntimeError as e:
                    if 'CUBLAS_STATUS_EXECUTION_FAILED' in str(e):
                        out = pipeline.unet(z_t.float(), timesteps, encoder_hidden_states=et_proj.float())
                    else:
                        raise
                loss = F.mse_loss(out.sample, noise) / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.accum_steps

        # Validation
        unet.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast():
            for z0, et in val_loader:
                z0 = z0.to(device).permute(0,2,1,3,4)
                et = et.to(device).unsqueeze(1)
                et_proj  = proj_eeg(et)
                noise    = torch.randn_like(z0)
                timesteps = torch.randint(0, len(scheduler.timesteps), (z0.size(0),), device=device)
                z_t      = scheduler.add_noise(z0, noise, timesteps)
                try:
                    out = pipeline.unet(z_t, timesteps, encoder_hidden_states=et_proj)
                except RuntimeError as e:
                    if 'CUBLAS_STATUS_EXECUTION_FAILED' in str(e):
                        out = pipeline.unet(z_t.float(), timesteps, encoder_hidden_states=et_proj.float())
                    else:
                        raise
                val_loss += F.mse_loss(out.sample, noise).item()
        print(f"Epoch {epoch}: train_loss={running_loss/len(train_loader):.4f}, val_loss={val_loss/len(val_loader):.4f}")

        torch.cuda.empty_cache()

    # Save
    out_dir = f"{root}/checkpoints/tuneavideo_unet"
    os.makedirs(out_dir, exist_ok=True)
    pipeline.unet.save_pretrained(out_dir)
    print("UNet saved.")

if __name__ == '__main__':
    train()
