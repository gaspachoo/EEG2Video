#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script “v7” pour EEG→Video avec votre version train_tuneavideo_v7.

Ce script :
1. Parse les arguments (chemins, checkpoint…).
2. Charge les latents Seq2Seq (ẑ) et les embeddings sémantiques (êt) depuis fichiers.
3. Reconstruit la pipeline TuneAVideoPipeline avec vos composants fine-tunés.
4. Lance la génération vidéo pour chaque paire de latents.
5. Sauvegarde les GIFs dans le dossier “Fullmodel”.
"""

import os
import argparse
import torch
import numpy as np
from einops import rearrange

from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTokenizer

from pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference EEG→Video v7 : Seq2Seq + Semantic Embeddings"
    )
    p.add_argument(
        "--base_model_path", type=str,
        default="./Gaspard/stable-diffusion-v1-4",
        help="Chemin vers SD-v1-4 pré-entraîné"
    )
    p.add_argument(
        "--ckpt_dir", type=str,
        default="./Gaspard/checkpoints/TuneAVideo",
        help="Répertoire des checkpoints UNet (unet_epXX.pt)"
    )
    p.add_argument(
        "--ckpt_epoch", type=int, default=50,
        help="Numéro de l’epoch à charger"
    )
    p.add_argument(
        "--seq2seq_dir", type=str, default="./data/Video_latents",
        help="Répertoire des latents Seq2Seq (*.npy)"
    )
    p.add_argument(
        "--sem_dir", type=str, default="./data/Semantic_embeddings",
        help="Répertoire des embeddings sémantiques (*.npy)"
    )
    p.add_argument(
        "--output_dir", type=str, default="./outputs/inference_v7",
        help="Répertoire de sortie pour les GIFs"
    )
    p.add_argument(
        "--video_length", type=int, default=6,
        help="Nombre de frames par vidéo"
    )
    p.add_argument(
        "--height", type=int, default=288,
        help="Hauteur des frames"
    )
    p.add_argument(
        "--width", type=int, default=512,
        help="Largeur des frames"
    )
    p.add_argument(
        "--num_inference_steps", type=int, default=100,
        help="Nombre de pas de diffusion"
    )
    p.add_argument(
        "--guidance_scale", type=float, default=12.5,
        help="Coefficient de guidance"
    )
    return p.parse_args()

def load_zlatents(path_or_dir: str, device: torch.device) -> torch.Tensor:
    """
    Charge un fichier ou répertoire de latents vidéo (.npy ou .pt) 
    et renvoie un Tensor (B, F, C, H, W).
    """
    blocks = []
    paths = sorted(os.listdir(path_or_dir)) if os.path.isdir(path_or_dir) else [path_or_dir]
    for fname in paths:
        full = os.path.join(path_or_dir, fname) if os.path.isdir(path_or_dir) else fname
        if not (full.endswith('.npy') or full.endswith('.pt')):
            continue
        data = np.load(full) if full.endswith('.npy') else torch.load(full)
        lat = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
        lat = lat.half().to(device)
        lat = rearrange(lat, 'b c f h w -> b f c h w')
        blocks.append(lat)
    return torch.cat(blocks, dim=0)


def load_semantics(path_or_dir: str, device: torch.device) -> torch.Tensor:
    """
    Charge un fichier ou répertoire d'embeddings sémantiques (*.npy)
    et renvoie un Tensor (B, D).
    """
    blocks = []
    paths = sorted(os.listdir(path_or_dir)) if os.path.isdir(path_or_dir) else [path_or_dir]
    for fname in paths:
        full = os.path.join(path_or_dir, fname) if os.path.isdir(path_or_dir) else fname
        if not full.endswith('.npy'):
            continue
        arr = np.load(full)
        emb = torch.from_numpy(arr).to(device)
        blocks.append(emb)
    return torch.cat(blocks, dim=0)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(114514)

    # → Charger latents Seq2Seq et embeddings sémantiques
    z_hat = load_zlatents(args.seq2seq_dir, device)
    e_t   = load_semantics(args.sem_dir, device)

    # → Préparer le negative baseline pour classifier-free guidance
    # On crée un fichier './negative.npy' attendu par la pipeline
    neg = e_t.mean(dim=0, keepdim=True).cpu().numpy()
    np.save(os.path.join(os.getcwd(), 'negative.npy'), neg)

    # → Composants figés
    vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder='vae').to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    scheduler = PNDMScheduler.from_pretrained(args.base_model_path, subfolder='scheduler')

    # → Charger UNet fine-tuné
    ckpt_path = os.path.join(args.ckpt_dir, f'unet_ep{args.ckpt_epoch}.pt')
    unet = UNet3DConditionModel.from_pretrained(ckpt_path).to(device).eval()

        # → Construire la pipeline
    pipe = TuneAVideoPipeline(
        vae=vae,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler
    )
    # déplacer les modules et passer en mode évaluation
    pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()

    pipe.enable_xformers_memory_efficient_attention()
        # ajustement VRAM: slicing plutôt que tiling
    pipe.vae.enable_slicing()
    pipe.scheduler.set_timesteps(args.num_inference_steps)

    # → Préparer dossier de sortie
    out_root = os.path.join(args.output_dir, 'Fullmodel')
    os.makedirs(out_root, exist_ok=True)

        # → Boucle d’inférence
    for i in range(z_hat.size(0)):
        lat_z = z_hat[i:i+1]
        sem   = e_t[i:i+1]

        # Inference : sem is passed as "eeg" argument, lat_z as latents
        result = pipe(
            None,
            sem,
            args.video_length,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            latents=lat_z
        )
        videos = result.videos

        save_videos_grid(videos, os.path.join(out_root, f'sample_{i}.gif'))
        print(f"[INFO] Saved sample_{i}.gif → {out_root}")(f"[INFO] Saved sample_{i}.gif → {out_root}")

if __name__ == '__main__':
    main()
