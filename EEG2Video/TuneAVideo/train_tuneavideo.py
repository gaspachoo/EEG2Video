import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EEG2Video.TuneAVideo.tuneavideo.models.unet import UNet3DConditionModel

from EEG2Video.TuneAVideo.tuneavideo.util_tuneavideo import save_videos_grid
from EEG2Video.TuneAVideo.tuneavideo.datasets.dataset import TuneAVideoDataset


def train_one_video(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    unet = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_model_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_path, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Freeze UNet and unfreeze query projections only
    unet.requires_grad_(False)
    trainable_suffixes = ("attn1.to_q", "attn2.to_q", "attn_temp")
    for name, module in unet.named_modules():
        if name.endswith(trainable_suffixes):
            for p in module.parameters():
                p.requires_grad_(True)
    # Only these layers are fine-tuned to save memory

    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Prepare dataset
    dataset = TuneAVideoDataset(**cfg.train_data)
    dataset.prompt_ids = tokenizer(dataset.prompt, max_length=tokenizer.model_max_length,
                                   padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train_batch_size)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, unet.parameters()), lr=cfg.learning_rate
    )
    unet.train()
    for step, batch in enumerate(loader):
        if step >= cfg.max_train_steps:
            break
        pixel_values = batch["pixel_values"].to(device)
        b, f = pixel_values.shape[:2]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b)
            latents = latents * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (b,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        enc = text_encoder(batch["prompt_ids"].to(device))[0]
        pred = unet(noisy_latents, timesteps, enc).sample
        loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"step {step} loss {loss.item():.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(cfg.output_dir, "unet"))

def main():
    parser = argparse.ArgumentParser(description="Tiny training loop for TuneAVideo text pipeline")
    parser.add_argument("--config", type=str, default="Tune_A_Video/configs/man-skiing.yaml",
                        help="Path to a Tune-A-Video YAML config")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    train_one_video(cfg)
    print(f"[INFO] UNet saved in {cfg.output_dir}/unet")

if __name__ == "__main__":
    main()
