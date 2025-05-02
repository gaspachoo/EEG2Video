import os
import numpy as np
import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL

# === Config ===
save_root = os.path.expanduser("~/EEG2Video")
video_root = f"{save_root}/data/Video_gifs/"
out_path = f"{save_root}/data/1200_latent.npy"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda()
vae.eval()
transform = transforms.ToTensor()

all_latents = []

for block in tqdm(range(6),desc="Generating Latents"):  # Blocks 0–5 for training
    video_dir = os.path.join(video_root, f"Block{block}")
    for concept in tqdm(range(40), desc=f"Block {block}"):
        for rep in range(5):
            gif_index = concept * 5 + rep + 1
            gif_path = os.path.join(video_dir, f"{gif_index}.gif")
            if not os.path.exists(gif_path):
                print(f"❌ Missing video: {gif_path}")
                continue

            frames = imageio.mimread(gif_path)
            if len(frames) != 6:
                print(f"⚠️ GIF {gif_path} has {len(frames)} frames, expected 6.")
                continue

            frames = [transform(f) for f in frames]
            frames = torch.stack(frames).cuda()  # shape: (6, 3, 288, 512)

            with torch.no_grad():
                z_latents = vae.encode(frames).latent_dist.mean  # (6, 4, 36, 64)

            all_latents.append(z_latents.cpu().numpy())

# === Final assembly ===
latents_np = np.stack(all_latents, axis=0).reshape(40, 5, 6, 4, 36, 64).transpose(0, 1, 3, 2, 4, 5)  # (1200, 6, 4, 36, 64) -> shape: (40, 5, 4, 6, 36, 64)

np.save(out_path, latents_np)
print(f"✅ Saved latents to {out_path} with shape {latents_np.shape}")
