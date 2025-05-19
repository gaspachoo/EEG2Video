import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler, BitsAndBytesConfig
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers import pipeline_utils
from accelerate import Accelerator
import wandb
from tqdm import tqdm

# Quantization config 8-bit
quant_config = BitsAndBytesConfig(load_in_8bit=True)
# Mitigate fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

class EEGVideoDataset(Dataset):
    def __init__(self, zhat_dir: str, sem_dir: str):
        z_blocks, e_blocks = [], []
        for fname in sorted(os.listdir(zhat_dir)):
            if fname.endswith('.npy'):
                z_blocks.append(np.load(os.path.join(zhat_dir, fname)))
        for fname in sorted(os.listdir(sem_dir)):
            if fname.endswith('.npy'):
                e_blocks.append(np.load(os.path.join(sem_dir, fname)))
        self.z_hat = np.concatenate(z_blocks, axis=0)
        self.e_t   = np.concatenate(e_blocks, axis=0)
        assert self.z_hat.shape[0] == self.e_t.shape[0], (
            f"Mismatch: {self.z_hat.shape[0]} vs {self.e_t.shape[0]}"
        )

    def __len__(self):
        return len(self.z_hat)

    def __getitem__(self, idx):
        z0 = torch.from_numpy(self.z_hat[idx]).float()
        et = torch.from_numpy(self.e_t[idx]).float()
        return z0, et

class TuneAVideoTrainer:
    def __init__(self, args):
        # Initialize Accelerator for DDP, mixed precision, device placement
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.device = self.accelerator.device

        # Prepare dataset and dataloaders
        dataset = EEGVideoDataset(args.zhat_dir, args.sem_dir)
        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False
        )

        # Load and freeze VAE on CPU
        self.vae = AutoencoderKL.from_pretrained(
            'CompVis/stable-diffusion-v1-4', subfolder='vae'
        ).to('cpu').eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # Tokenizer on CPU
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')

        # Load UNet with bitsandbytes quant + fp16 dtype and offload
        self.unet = UNet3DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            torch_dtype=torch.float16,
            quantization_config=quant_config
        ).to(self.device)
        pipeline_utils.get_accelerate_model_args().enable_model_cpu_offload()

        # Scheduler on CPU
        self.scheduler = PNDMScheduler.from_pretrained(
            'CompVis/stable-diffusion-v1-4', subfolder='scheduler'
        ).to('cpu')

        # Semantic projection on device (float32 to avoid dtype mismatches)
        sem_dim = dataset[0][1].shape[-1]
        cross_dim = self.unet.config.cross_attention_dim
        self.proj_eeg = nn.Linear(sem_dim, cross_dim).to(self.device)
        for p in self.proj_eeg.parameters():
            p.requires_grad_(False)  # freeze if not training

        # Optimizer only for UNet parameters
        self.optimizer = torch.optim.Adam(
            self.unet.parameters(), lr=args.lr
        )

        # Prepare with Accelerator: wraps DDP, AMP, device placement
        self.unet, self.proj_eeg, self.optimizer, train_loader, val_loader = (
            self.accelerator.prepare(
                self.unet,
                self.proj_eeg,
                self.optimizer,
                train_loader,
                val_loader
            )
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

        # WandB logging
        if args.use_wandb and self.accelerator.is_main_process:
            wandb.init(project='eeg2video-tuneavideo', config=vars(args))

        self.scaler = None  # Accelerator handles scaling
        self.args = args

    def _train_epoch(self, epoch: int):
        self.unet.train()
        total_loss = 0.0
        for z0, et in tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.args.epochs}"):
            # Move and reshape
            z0 = z0.to(self.device).permute(0,2,1,3,4)
            et = et.to(self.device).unsqueeze(1)

            noise = torch.randn_like(z0)
            timesteps = torch.randint(
                0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device
            )
            z_t = self.scheduler.add_noise(z0, noise, timesteps)

            # Forward + backward via Accelerator
            self.optimizer.zero_grad(set_to_none=True)
            with self.accelerator.autocast():
                out = self.unet(z_t, timesteps, encoder_hidden_states=et)
                loss = F.mse_loss(out.sample, noise)

            self.accelerator.backward(loss)
            self.optimizer.step()

            total_loss += loss.item()
            # Manual cache cleanup
            del z0, et, noise, timesteps, z_t, out, loss
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.unet.eval()
        total_loss = 0.0
        with torch.no_grad(), self.accelerator.autocast():
            for z0, et in tqdm(self.val_loader, desc="Validation"):
                z0 = z0.to(self.device).permute(0,2,1,3,4)
                et = et.to(self.device).unsqueeze(1)

                noise = torch.randn_like(z0)
                timesteps = torch.randint(
                    0, len(self.scheduler.timesteps), (z0.size(0),), device=self.device
                )
                z_t = self.scheduler.add_noise(z0, noise, timesteps)

                out = self.unet(z_t, timesteps, encoder_hidden_states=et)
                total_loss += F.mse_loss(out.sample, noise).item()

                del z0, et, noise, timesteps, z_t, out
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            tr_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch()
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}")

            if epoch % self.args.save_every == 0 and self.accelerator.is_main_process:
                ckpt_dir = os.path.join(self.args.root, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                self.unet.save_pretrained(os.path.join(ckpt_dir, f'unet_epoch{epoch}.pt'))
                print(f"Saved checkpoint epoch {epoch}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
    parser.add_argument('--zhat_dir',   type=str, default=f"{root}/data/Predicted_latents")
    parser.add_argument('--sem_dir',    type=str, default=f"{root}/data/Semantic_embeddings")
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--use_wandb',  action='store_true')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--root',       type=str, default=root)
    args = parser.parse_args()

    trainer = TuneAVideoTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
