import os
import numpy as np
import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL

# === Config ===
save_root = os.path.expanduser("~/EEG2Video")
video_dir = os.path.join(save_root, "data/Video_gifs/Block6")
out_path = os.path.join(save_root, "data/40classes_latents.pt")

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda()
vae.eval()
transform = transforms.ToTensor()

# shape target: (40, 5, 6, 4, 36, 64)
all_latents = []

for concept in tqdm(range(40), desc="Block6 by concept"):
    reps = []
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
        frames = torch.stack(frames).cuda()

        with torch.no_grad():
            z_latents = vae.encode(frames).latent_dist.mean  # (6, 4, 36, 64)
        reps.append(z_latents.cpu())

    all_latents.append(torch.stack(reps))  # (5, 6, 4, 36, 64)

# Tensor: (40, 5, 6, 4, 36, 64)
final_tensor = torch.stack(all_latents).permute(0, 1, 3, 2, 4, 5)  # (40, 5, 4, 6, 36, 64)
torch.save(final_tensor, out_path)
print(f"✅ Saved latents to {out_path} with shape {final_tensor.shape}")
