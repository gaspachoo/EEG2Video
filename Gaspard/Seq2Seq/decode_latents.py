import os
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image


def decode_latents(latents_dir, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)

    # Load VAE decoder
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # List latent files
    latent_files = sorted([f for f in os.listdir(latents_dir) if f.endswith(".npy")])
    print(f"Found {len(latent_files)} latent files in {latents_dir}")
    for i_file,file in enumerate(tqdm(latent_files, desc="Decoding latent files")):
        latents = np.load(os.path.join(latents_dir, file))  # shape: (N, 6, 4, 36, 64)
        latents = torch.tensor(latents, dtype=torch.float32).to(device)
        print(f"Loaded latents: {file}, shape={latents.shape}")
        for i in tqdm(range(len(latents)),desc="Processing latents"):  # clip: (6, 4, 36, 64)
            clip = latents[i]  # (6, 4, 36, 64)
            clip_imgs = []
            for j in range(clip.shape[0]):
                latent = clip[j].unsqueeze(0)  # (1, 4, 36, 64)
                with torch.no_grad():
                    recon = vae.decode(latent).sample  # (1, 3, H, W)
                    recon = (recon.clamp(-1, 1) + 1) / 2  # to [0,1]
                    img = recon.squeeze(0).cpu()
                    img_pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    clip_imgs.append(img_pil)

            # Save as 
            os.makedirs(os.path.join(output_dir, f'Block{i_file}'), exist_ok=True)
            gif_path = os.path.join(output_dir,f'Block{i_file}', f"{i+1}.gif")
            clip_imgs[0].save(gif_path, save_all=True, append_images=clip_imgs[1:], duration=300, loop=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--latents_dir", type=str, default="./data/Seq2Seq/Video_latents")
    parser.add_argument("--output_dir", type=str, default="./data/Seq2Seq/Decoded_gifs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    decode_latents(args.latents_dir, args.output_dir, device=args.device)
