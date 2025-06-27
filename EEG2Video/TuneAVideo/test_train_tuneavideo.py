import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

from EEG2Video.TuneAVideo.models.unet import UNet3DConditionModel
from EEG2Video.TuneAVideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline

# Add original Tune-A-Video utilities to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.path.join(ROOT, "..", "Tune-A-Video")
sys.path.insert(0, ORIG)
from tuneavideo.util import save_videos_grid
from tuneavideo.data_t.dataset import TuneAVideoDataset


def train_one_video(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    unet = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_model_path, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_path, subfolder="scheduler")

    # Prepare dataset
    dataset = TuneAVideoDataset(**cfg.train_data)
    dataset.prompt_ids = tokenizer(dataset.prompt, max_length=tokenizer.model_max_length,
                                   padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train_batch_size)

    optimizer = torch.optim.Adam(unet.parameters(), lr=cfg.learning_rate)
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
    return vae, text_encoder, tokenizer, unet


def run_inference(models, cfg):
    vae, text_encoder, tokenizer, unet = models
    device = vae.device
    scheduler = DDIMScheduler.from_pretrained(cfg.pretrained_model_path, subfolder="scheduler")
    pipe = TuneAVideoPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    prompt = cfg.validation_data.prompts[0]
    out = pipe(prompt=prompt,
               video_length=cfg.validation_data.video_length,
               height=cfg.validation_data.height,
               width=cfg.validation_data.width,
               num_inference_steps=cfg.validation_data.num_inference_steps,
               guidance_scale=cfg.validation_data.guidance_scale)
    save_videos_grid(out.videos, os.path.join(cfg.output_dir, "train_test.gif"), rescale=False)


def main():
    parser = argparse.ArgumentParser(description="Tiny training loop for TuneAVideo text pipeline")
    parser.add_argument("--config", type=str, default="Tune-A-Video/configs/car-turn.yaml",
                        help="Path to a Tune-A-Video YAML config")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    models = train_one_video(cfg)
    run_inference(models, cfg)


if __name__ == "__main__":
    main()
