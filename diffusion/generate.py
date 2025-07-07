import argparse
import torch
from torch import nn
from diffusers import DiffusionPipeline


def build_mlp(in_dim, out_dim):
    """Simple two-layer MLP mapping latent to text embedding."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim * 2),
        nn.GELU(),
        nn.Linear(out_dim * 2, out_dim),
    )


def main(args):
    pipe = DiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)

    text_dim = pipe.text_encoder.config.hidden_size
    mlp = build_mlp(args.latent_dim, text_dim).to(args.device)
    pipe.text_encoder = mlp

    latent = torch.load(args.latent).to(args.device)
    with torch.no_grad():
        embedding = pipe.text_encoder(latent)
        video = pipe(prompt_embeds=embedding, num_inference_steps=25).videos

    video[0].save(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from latent")
    parser.add_argument(
        "--model",
        type=str,
        default="Open-Sora-Plan-1.3",
        help="Diffusion model checkpoint to load",
    )
    parser.add_argument("--latent", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)
