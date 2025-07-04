import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
import decord


decord.bridge.set_bridge('torch')


class VideoFolderDataset(torch.utils.data.Dataset):
    """Dataset loading all videos from a folder."""

    def __init__(self, root, size=256):
        self.root = root
        self.size = size
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(('.mp4', '.gif'))
        ]
        self.transform = transforms.Compose(
            [transforms.Resize(size), transforms.CenterCrop(size)]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vr = decord.VideoReader(self.files[idx])
        frames = vr.get_batch(range(len(vr)))  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        frames = self.transform(frames)
        return frames


def add_lora(model, ratio=0.05):
    """Replace Linear layers with LoRA-equipped layers."""

    class LoRALinear(nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            self.linear = linear
            self.lora_down = nn.Linear(linear.in_features, rank, bias=False)
            self.lora_up = nn.Linear(rank, linear.out_features, bias=False)
            nn.init.zeros_(self.lora_up.weight)

        def forward(self, x):
            return self.linear(x) + self.lora_up(self.lora_down(x))

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_dim, out_dim = module.in_features, module.out_features
            rank = max(1, int(min(in_dim, out_dim) * ratio))
            lora = LoRALinear(module, rank)
            parent = model
            *path, last = name.split('.')
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, lora)


async def train(args):
    accelerator = Accelerator()
    pipe = DiffusionPipeline.from_pretrained(
        args.diffusion_weights, torch_dtype=torch.float16
    )
    pipe.to(accelerator.device)

    for param in pipe.unet.parameters():
        param.requires_grad = False
    add_lora(pipe.unet, ratio=0.05)

    dataset = VideoFolderDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.unet.parameters()), lr=1e-4
    )
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=args.steps
    )

    pipe.train()
    global_step = 0
    while global_step < args.steps:
        for videos in loader:
            videos = videos.to(accelerator.device)
            with accelerator.accumulate(pipe.unet):
                noise = torch.randn_like(videos)
                timesteps = torch.randint(
                    0, pipe.scheduler.num_train_timesteps, (videos.shape[0],), device=videos.device
                ).long()
                with torch.no_grad():
                    latents = pipe.vae.encode(videos).latent_dist.sample()
                noise_pred = pipe.unet(latents, timesteps).sample
                loss = nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step >= args.steps:
                    break

    accelerator.wait_for_everyone()
    pipe.save_pretrained(args.output)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training on real clips")
    parser.add_argument("--data", type=str, required=True, help="Folder with videos")
    parser.add_argument("--output", type=str, required=True, help="Path to save LoRA")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--diffusion_weights",
        type=str,
        default="hpcai-tech/Open-Sora-Plan-1.3",
        help="Diffusion model checkpoint to load",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.empty_cache()
    import asyncio

    asyncio.run(train(args))
