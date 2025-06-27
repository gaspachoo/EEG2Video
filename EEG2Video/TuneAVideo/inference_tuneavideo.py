import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EEG2Video.TuneAVideo.tuneavideo.models.unet import UNet3DConditionModel
from EEG2Video.TuneAVideo.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline

# Add original Tune-A-Video package to path for utility functions
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_TAV_PATH = os.path.join(ROOT_DIR, "..", "Tune-A-Video")
sys.path.insert(0, ORIG_TAV_PATH)
from EEG2Video.TuneAVideo.tuneavideo.util_tuneavideo import save_videos_grid

def main():
    parser = argparse.ArgumentParser(
        description="Test TuneAVideo text pipeline using a YAML config",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Tune_A_Video/configs/man-skiing.yaml",
        help="Path to a Tune-A-Video YAML configuration file",
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        default='./Tune_A_Video/outputs/man-skiing/unet',
        help="Optional path to a saved UNet directory",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = cfg.pretrained_model_path
    prompt = cfg.validation_data.prompts[0]
    video_length = cfg.validation_data.video_length
    height = cfg.validation_data.height
    width = cfg.validation_data.width
    num_steps = cfg.validation_data.num_inference_steps
    guidance_scale = cfg.validation_data.guidance_scale

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae")
    # When --unet_path is specified, load the UNet weights from this location
    # instead of the base pretrained model. This helps testing trained checkpoints.
    if args.unet_path:
        unet = UNet3DConditionModel.from_pretrained_2d(args.unet_path)
    else:
        unet = UNet3DConditionModel.from_pretrained_2d(pretrained_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")

    pipe = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    output = pipe(
        prompt=prompt,
        video_length=video_length,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_videos_grid(output.videos, os.path.join(cfg.output_dir, "test.gif"), rescale=False)


if __name__ == "__main__":
    main()
