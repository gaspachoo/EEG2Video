import os
import torch
import imageio
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from PIL import Image

# Configuration
IMG_SIZE = (288, 512)  # height, width
N_FRAMES = 6           # number of frames to extract (2s at 3 FPS)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def extract_frames_from_gif(gif_path, n_frames=N_FRAMES):
    try:
        reader = imageio.get_reader(gif_path)
        frames = []
        for i, frame in enumerate(reader):
            if len(frames) >= n_frames:
                break
            if not isinstance(frame, np.ndarray):
                continue
            frame = Image.fromarray(frame)
            frame_tensor = transform(frame)  # shape: (3, H, W)
            frames.append(frame_tensor)
        reader.close()
        if len(frames) < n_frames:
            if len(frames) == 0:
                return None
            frames += [frames[-1]] * (n_frames - len(frames))
        return torch.stack(frames)  # (N_FRAMES, 3, H, W)
    except Exception as e:
        print(f"Skipping {gif_path}: {e}")
        return None

def generate_all_latents(gif_root, output_root, device='cuda'):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()

    os.makedirs(output_root, exist_ok=True)

    for i in range(7):
        all_latents = []
        block_dir = os.path.join(gif_root, f"Block{i}")
        gif_files = sorted([f for f in os.listdir(block_dir) if f.endswith('.gif')], key=lambda x: int(x.replace('.gif', '')))
        print(f"Processing Block {i} ({len(gif_files)} gifs)...")

        for fname in tqdm(gif_files):
            gif_path = os.path.join(block_dir, fname)
            frames = extract_frames_from_gif(gif_path)
            if frames is None:
                continue
            with torch.no_grad():
                frames = frames.to(device)
                latents = vae.encode(frames).latent_dist.sample()  # (6, 4, 36, 64)
                latents = latents.cpu().numpy()
                all_latents.append(latents)

        if len(all_latents) == 0:
            print(f"No valid GIFs found in Block {i}. Skipping save.")
            continue

        all_latents = np.stack(all_latents)  # (N, 6, 4, 36, 64)
        output_path = os.path.join(output_root, f"block{i}_latents.npy")
        np.save(output_path, all_latents)
        print(f"Saved Block {i} latents to {output_path} with shape {all_latents.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
     #"/Documents/School/Centrale Med/2A/SSE/EEG2Video"
    parser.add_argument('--gif_dir', type=str,  default = "./data/Video_gifs/", help="Directory with .gif files")
    parser.add_argument('--output_path', type=str, default = "./data/Video_latents/", help="Path to save .npy latents")
    args = parser.parse_args()

    generate_all_latents(args.gif_dir, args.output_path)