import os
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers.models.vae import AutoencoderKL
import torchvision.transforms as T
import imageio
from PIL import Image
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/Video_Gif", help="GIF dataset path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="./vae_reconstructions", help="Directory to save output GIFs")
    parser.add_argument("--max_samples", type=int, default=5, help="Number of GIFs to process")
    return parser.parse_args()

class EEGGIFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.samples = []
        for block in sorted(os.listdir(root_dir)):
            block_path = os.path.join(root_dir, block)
            if not os.path.isdir(block_path):
                continue
            for gif in sorted(os.listdir(block_path)):
                self.samples.append(os.path.join(block_path, gif))
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gif_path = self.samples[idx]
        frames = imageio.mimread(gif_path)
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, gif_path  # return path for saving

def save_comparison(original, reconstructed, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    images = []
    for o, r in zip(original, reconstructed):
        top = T.ToPILImage()(o.cpu().clamp(0, 1))
        bottom = T.ToPILImage()(r.cpu().clamp(0, 1))
        combined = Image.new('RGB', (top.width, top.height * 2))
        combined.paste(top, (0, 0))
        combined.paste(bottom, (0, top.height))
        images.append(combined)
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=333, loop=0)

def test_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.load_state_dict(torch.load(args.checkpoint, map_location=device))
    vae.eval()

    # Dataset
    transform = T.Compose([T.ToTensor()])
    dataset = EEGGIFDataset(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (batch, gif_path) in enumerate(tqdm(loader, desc="Testing")):
            if i >= args.max_samples:
                break
            batch = batch.squeeze(0).to(device)  # (T, 3, H, W)
            frames = batch

            posterior = vae.encode(frames).latent_dist
            z = posterior.sample()
            reconstructions = vae.decode(z).sample

            save_path = os.path.join(args.output_dir, f"recon_{os.path.basename(gif_path[0])}")
            save_comparison(frames, reconstructions, save_path)
            print(f"✅ Saved comparison to {save_path}")

if __name__ == "__main__":
    test_vae(parse_args())
