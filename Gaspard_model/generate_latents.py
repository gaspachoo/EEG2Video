import os
import torch
import imageio
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from diffusers.models import AutoencoderKL

# Configuration
IMG_SIZE = (288, 512)  # height, width
N_FRAMES = 6           # number of frames to extract (2s at 3 FPS)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def extract_frames_from_gif(gif_path, n_frames=N_FRAMES):
    reader = imageio.get_reader(gif_path)
    frames = []
    for i, frame in enumerate(reader):
        if len(frames) >= n_frames:
            break
        frame_tensor = transform(frame)  # shape: (3, H, W)
        frames.append(frame_tensor)
    reader.close()
    if len(frames) < n_frames:
        # Pad with last frame if not enough
        frames += [frames[-1]] * (n_frames - len(frames))
    return torch.stack(frames)  # (N_FRAMES, 3, H, W)

def generate_latents(gif_dir, output_path, device='cuda'):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()

    all_latents = []
    gif_files = sorted([f for f in os.listdir(gif_dir) if f.endswith('.gif')])

    for fname in tqdm(gif_files, desc="Processing GIFs"):
        gif_path = os.path.join(gif_dir, fname)
        frames = extract_frames_from_gif(gif_path)  # (6, 3, H, W)
        with torch.no_grad():
            frames = frames.to(device)
            latents = vae.encode(frames).latent_dist.sample()  # (6, 4, H//8, W//8)
            latents = latents.cpu().numpy()
            all_latents.append(latents)  # append (6, 4, H//8, W//8)

    all_latents = np.stack(all_latents)  # (N, 6, 4, H//8, W//8)
    np.save(output_path, all_latents)
    print(f"Saved latents to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gif_dir', type=str, required=True, help="Directory with .gif files")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save .npy latents")
    args = parser.parse_args()

    generate_latents(args.gif_dir, args.output_path)