import os
import argparse
import numpy as np
import torch
from torch import nn
from decord import VideoReader, cpu
import imageio


class SimpleVideoVAE(nn.Module):
    """Minimal video VAE encoder returning a latent vector per frame."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        return self.fc(h)


def load_vae(weights_path: str, device: str = "cpu") -> nn.Module:
    """Load the pretrained VAE and freeze it."""
    model = SimpleVideoVAE()
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_clip_latents(path: str, vae: nn.Module, device: str = "cpu") -> np.ndarray:
    """Return latent tensor of shape ``(6, 256)`` for the given clip."""
    reader = VideoReader(path, width=512, height=288, ctx=cpu(0))
    frames = None
    if len(reader) < 6 and path.lower().endswith(".gif"):
        # Fall back to imageio when Decord does not decode all GIF frames
        gif_reader = imageio.get_reader(path)
        images = [im for im in gif_reader]
        gif_reader.close()
        if len(images) < 6:
            raise RuntimeError(f"Video {path} is too short: {len(images)} frames")
        indices = np.linspace(0, len(images) - 1, 6).astype(np.int64)
        frames = np.stack([images[i] for i in indices], axis=0)
    else:
        if len(reader) < 6:
            raise RuntimeError(f"Video {path} is too short: {len(reader)} frames")
        indices = np.linspace(0, len(reader) - 1, 6).astype(np.int64)
        frames = reader.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = frames.to(device)
    with torch.no_grad():
        latents = vae(frames).cpu().numpy()
    return latents


def process_all_clips(input_dir: str, weights: str, output_dir: str, device: str = "cpu") -> None:
    """Generate latents for every video file in ``input_dir`` and its subfolders."""
    os.makedirs(output_dir, exist_ok=True)
    vae = load_vae(weights, device)

    for root, _, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)
        out_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith((".mp4", ".gif", ".mov")):
                continue
            path = os.path.join(root, fname)
            latents = extract_clip_latents(path, vae, device)
            out_name = os.path.splitext(fname)[0] + ".npz"
            out_path = os.path.join(out_dir, out_name)
            np.savez(out_path, latents=latents)
            print(f"Saved latents for {path} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video latents with a frozen VAE")
    parser.add_argument(
        "--input_dir",
        default="./data/Seq2Seq/Video_gifs",
        help="directory with 2s clips organized by block",
    )
    parser.add_argument("--weights", default="encoders/video_vae/vae.pt", help="path to VAE weights")
    parser.add_argument("--output_dir", default="data/gaspardnew/video_latents", help="where to store latent tensors")
    parser.add_argument("--device", default="cpu", help="computation device")
    args = parser.parse_args()

    process_all_clips(args.input_dir, args.weights, args.output_dir, device=args.device)
