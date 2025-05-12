import os
import sys
# Project root (must be added to sys.path for module imports)
root = os.path.expanduser("~/EEG2Video")
sys.path.insert(0, root)

import torch
torch.backends.cuda.matmul.allow_tf32 = False
import numpy as np
from diffusers import DDIMScheduler
# Absolute imports from project modules
from models.unet import UNet3DConditionModel
from pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from train_semantic import SemanticPredictor
from train_seq2seq import Seq2SeqTransformer
from einops import rearrange
from tuneavideo.util import save_videos_grid

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to pretrained checkpoints
SEM_CKPT = os.path.join(root, "Gaspard_model/checkpoints/semantic_predictor.pth")
SEQ2SEQ_DIR = os.path.join(root, "Gaspard_model/checkpoints/seq2seq")  # contains seq2seq_block{block_id}.pth

# Input data directories
eeg_dir = os.path.join(root, "data/EEG_embeddings")       # .npy files of raw EEG features (310 dim)
video_latents_dir = os.path.join(root, "data/Video_latents")  # blockX_latents.npy files (200,6,4,36,64)

# Negative/static noise for DANA
NEG_PATH = os.path.join(root, "data/negative.npy")

# Stable Diffusion checkpoint for UNet & tokenizer
PRETRAINED_SD = "CompVis/stable-diffusion-v1-4" #"Zhoutianyi/huggingface/stable-diffusion-v1-4"

# Manual beta definition for DANA (fixed or schedule)
DYNAMIC_BETA = 0.95


def load_models():
    # Semantic predictor: maps raw EEG features (310) -> text embedding (77×768)
    sem_model = SemanticPredictor(input_dim=310, output_dim=77*768).to(device)
    sem_model.load_state_dict(torch.load(SEM_CKPT, map_location=device))
    sem_model.eval()

    # Seq2Seq transformer: maps EEG embeddings -> video latents
    seq_model = Seq2SeqTransformer().to(device)
    # Example loads block0; loop below will override per-block if needed
    ckpt0 = os.path.join(SEQ2SEQ_DIR, "seq2seq_block0.pth")
    seq_model.load_state_dict(torch.load(ckpt0, map_location=device))
    seq_model.eval()
    return sem_model, seq_model


def build_pipeline():
    # Load UNet3D and scheduler
    unet = UNet3DConditionModel.from_pretrained(PRETRAINED_SD, subfolder="unet", torch_dtype=torch.float16).to(device)
    scheduler = DDIMScheduler.from_pretrained(PRETRAINED_SD, subfolder="scheduler")

    # CLIP text encoder for semantic conditioning
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_SD, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_SD, subfolder="text_encoder").to(device)

    # Assemble pipeline
    pipe = TuneAVideoPipeline(
        vae=None,              # default VAE from pretrained
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler
    ).to(device)

    # Memory optimizations
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()
    return pipe


def main():
    sem_model, seq_model = load_models()
    pipe = build_pipeline()

    # Load negative/static noise for DANA
    negative = torch.from_numpy(np.load(NEG_PATH)).half().to(device)

    # Prepare output directory
    out_dir = os.path.join(root, "outputs/inference_dana_seq2seq")
    os.makedirs(out_dir, exist_ok=True)

    # Iterate over EEG feature files
    for feat_file in sorted(os.listdir(eeg_dir)):
        if not feat_file.endswith('.npy'): continue
        # Determine block id from filename if follows 'block{block}_...' else default to 0
        block_id = 0
        if 'block' in feat_file:
            try:
                block_id = int(feat_file.split('block')[1].split('_')[0])
            except:
                pass

        # Load raw features for semantic predictor: shape (N, 310)
        feats = np.load(os.path.join(eeg_dir, feat_file))
        x_eeg = torch.from_numpy(feats.astype(np.float32)).to(device)

        # Semantic embeddings: (N, 77, 768)
        with torch.no_grad():
            sem_out = sem_model(x_eeg)                  # (N, 77*768)
        prompt_embeds = sem_out.view(-1, 77, 768).half()

        # Load video latents for seq2seq: shape (200, 6, 4, 36, 64)
        latents0 = np.load(os.path.join(video_latents_dir, f"block{block_id}_latents.npy"))
        z0 = torch.from_numpy(latents0.astype(np.float32)).to(device)
        # Flatten latents for transformer: (N, 6, 4*36*64)
        z0_flat = z0.reshape(z0.shape[0], z0.shape[1], -1)

        # Predict latents via seq2seq: (N, 6, 4*36*64)
        with torch.no_grad():
            z_pred_flat = seq_model(x_eeg.unsqueeze(1).repeat(1,7,1) if len(x_eeg.shape)==2 else x_eeg, z0_flat)
        # Reshape back to (N, 6, 4, 36, 64)
        z_hat = z_pred_flat.view(-1, z0.shape[1], z0.shape[2], z0.shape[3], z0.shape[4]).half()

        # Inference per sample
        for i in range(z_hat.shape[0]):
            video = pipe(
                prompt_embeds=prompt_embeds[i:i+1],
                latents=z_hat[i:i+1],
                negative_eeg=negative,
                beta=DYNAMIC_BETA,
                video_length=z_hat.shape[1],
                height=z_hat.shape[3],
                width=z_hat.shape[4],
                num_inference_steps=100,
                guidance_scale=12.5
            ).videos

            save_videos_grid(
                video,
                os.path.join(out_dir, f"{os.path.splitext(feat_file)[0]}_sample{i}.gif")
            )

if __name__ == '__main__':
    main()
