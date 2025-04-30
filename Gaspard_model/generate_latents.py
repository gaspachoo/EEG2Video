import os
import numpy as np
import torch
import imageio
from torchvision import transforms
from tqdm import tqdm
import argparse
from diffusers.models import AutoencoderKL

def parse_args():
    parser = argparse.ArgumentParser()
    home = os.environ["HOME"]
    parser.add_argument("--input_dir", type=str, default=f"{home}/Gaspard/EEG2Video/data/Video_gifs", help="Path to video clips as .gif")
    parser.add_argument("--output_dir", type=str, default=f"{home}/Gaspard/EEG2Video/data/Video_latents", help="Output .npy latents (6, 256)")
    parser.add_argument("--checkpoint", type=str,default=f"{home}/Gaspard/EEG2Video/Gaspard_model/checkpoints/vae/vae_epoch30.pth", help="Path to trained VAE checkpoint .pth")
    return parser.parse_args()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base architecture
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    
    # Load trained weights
    vae.load_state_dict(torch.load(args.checkpoint, map_location=device))
    vae.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for fname in tqdm(sorted(os.listdir(args.input_dir))):
        if not fname.endswith(".gif"): print(f"{fname} is not a gif");continue
        frames = imageio.mimread(os.path.join(args.input_dir, fname))  # list of PILs
        frames = [transform(f) for f in frames]  # list of (3, H, W)
        frames = torch.stack(frames).to(device)  # (6, 3, 288, 512)

        with torch.no_grad():
            z = vae.encode(frames).latent_dist.mean  # (6, 4, 8, 8)
            z = z.view(6, -1)  # flatten to (6, 256)

        out_path = os.path.join(args.output_dir, fname.replace(".gif", ".npy"))
        np.save(out_path, z.cpu().numpy())

if __name__ == "__main__":
    main(parse_args())
