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

import os,sys
import argparse
import torch
import numpy as np
from einops import rearrange
from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from EEG2Video.TuneAVideo.pipelines.pipeline_tuneeeg2video import TuneAVideoPipeline
from EEG2Video.TuneAVideo.models.unet import UNet3DConditionModel
from EEG2Video.TuneAVideo.tuneavideo.util import save_videos_grid


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference EEG→Video v7 : Seq2Seq + Semantic Embeddings"
    )
    p.add_argument(
        "--diffusion_model_path", type=str,
        default="./stable-diffusion-v1-4",
        help="Chemin vers SD-v1-4 pré-entraîné"
    )
    p.add_argument(
        "--ckpt_dir", type=str,
        default="./EEG2Video/checkpoints/TuneAVideo",
        help="Répertoire des checkpoints UNet (unet_epXX.pt)"
    )
    p.add_argument(
        "--ckpt_epoch", type=int, default=50,
        help="Numéro de l’epoch à charger"
    )
    p.add_argument(
        "--seq2seq_dir", type=str, default="./data/Seq2Seq/Predicted_latents",
        help="Répertoire des latents Seq2Seq (*.npy)"
    )
    p.add_argument(
        "--sem_dir", type=str, default="./data/SemanticPredictor/Semantic_embeddings",
        help="Répertoire des embeddings sémantiques (*.npy)"
    )
    p.add_argument(
        "--output_dir", type=str, default="./data/TuneAVideo_outputs",
        help="Répertoire de sortie pour les GIFs"
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

def load_files(seq2seq_dir: str, sem_dir: str):
    """Load latent and semantic embedding blocks from disk.

    Parameters
    ----------
    seq2seq_dir : str
        Directory containing latent video blocks (``.npy``).
    sem_dir : str
        Directory containing semantic embedding blocks (``.npy``).

    Returns
    -------
    latent_data : list[np.ndarray]
        List of latent blocks loaded from ``seq2seq_dir``.
    embedding_data : list[np.ndarray]
        List of embedding blocks loaded from ``sem_dir``.
    """
    latent_files = sorted(
        [f for f in os.listdir(seq2seq_dir) if f.endswith(".npy") or f.endswith(".pt")]
    )
    latent_data, embedding_data = [], []
    for fname in latent_files:
        latent_data.append(np.load(os.path.join(seq2seq_dir, fname)))
        embedding_data.append(np.load(os.path.join(sem_dir, fname)))
    return latent_data, embedding_data


def load_pairs(latent_data, embedding_data, device="cuda"):
    """Preprocess and concatenate latent and embedding blocks.

    Parameters
    ----------
    latent_data : list[np.ndarray] | np.ndarray
        Latent video blocks.
    embedding_data : list[np.ndarray] | np.ndarray
        Corresponding semantic embedding blocks.
    device : str or torch.device, optional
        Device where tensors will be allocated.

    Returns
    -------
    video_latents : torch.Tensor
        Tensor of shape ``(B, F, C, H, W)``.
    semantic_embeddings : torch.Tensor
        Tensor of shape ``(B, 77, 768)``.
    """
    if isinstance(latent_data, np.ndarray):
        latent_data = [latent_data]
    if isinstance(embedding_data, np.ndarray):
        embedding_data = [embedding_data]
        
    assert len(latent_data) == len(embedding_data), (
        f"Mismatch between latent blocks ({len(latent_data)}) and embedding blocks ({len(embedding_data)})"
    )
    video_list, sem_list = [], []
    for lat_arr, emb_arr in zip(latent_data, embedding_data):
        lat = torch.from_numpy(lat_arr).to(device).half()
        lat = rearrange(lat, "b c f h w -> b f c h w")
        video_list.append(lat)

        arr = emb_arr
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim == 2 and arr.shape[1] != 768:
            D = arr.shape[1]
            if D % 768 != 0:
                raise ValueError(f"Embedding length {D} not divisible by 768")
            seq_len = D // 768
            arr = arr.reshape(-1, seq_len, 768)
        emb = torch.from_numpy(arr).to(device).half()
        sem_list.append(emb)

    video_latents, semantic_embeddings = torch.cat(video_list, dim=0), torch.cat(sem_list, dim=0)
    assert video_latents.size(0) == semantic_embeddings.size(0), \
        f"Mismatch video ({video_latents.size(0)}) vs sem ({semantic_embeddings.size(0)})"
    return video_latents, semantic_embeddings


def load_tuneavideo_from_checkpoint(tuneavideo_ckpt, diffusion_model_path, device):
     # Charger modules en float16
    vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder='vae').to(device).half().eval()
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    scheduler = PNDMScheduler.from_pretrained(diffusion_model_path, subfolder='scheduler')
    unet = UNet3DConditionModel.from_pretrained(tuneavideo_ckpt).to(device).half().eval()

    # Construire pipeline
    pipe = TuneAVideoPipeline(vae=vae, tokenizer=tokenizer, unet=unet, scheduler=scheduler).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.vae.enable_slicing()

    # Override méthode interne pour embeddings
    def _encode_eeg_override(self, model, eeg, device, num_videos_per_eeg,
                              do_classifier_free_guidance, negative_eeg):
        neg = negative_eeg.expand(-1, eeg.size(1), -1)
        return torch.cat([neg, eeg], dim=0)
    pipe._encode_eeg = _encode_eeg_override.__get__(pipe, TuneAVideoPipeline)
    return pipe
    
def inf_tuneavideo(pipe, emb,z0,num_inference_steps, guidance_scale, device='cuda'):
    neg_emb = emb.mean(dim=1, keepdim=True)
    result = pipe(
        None,
        emb,
        negative_eeg=neg_emb,
        latents=z0,
        video_length=6,
        height=288,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )
    return result.videos  # Tensor [1, F, H, W, 3]

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger données
    latent_data, embedding_data = load_files(args.seq2seq_dir, args.sem_dir)
    
    video_latents, semantic_embeddings = load_pairs(latent_data, embedding_data, device)
    
        
    unet_path = os.path.join(args.ckpt_dir, f'unet_ep{args.ckpt_epoch}.pt')
    pipe = load_tuneavideo_from_checkpoint(unet_path,args.diffusion_model_path, device)
    pipe.scheduler.set_timesteps(args.num_inference_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    B = video_latents.size(0)

    # --- DEBUG inf z0 before any scaling ---
    #print("[DEBUG inf] video_latents raw:", "shape", video_latents.shape,"mean", video_latents.mean().item(),"std",  video_latents.std().item(),"min",  video_latents.min().item(),"max",  video_latents.max().item())
    
    # Inférence
    for i in range(B):
        z0 = video_latents[i:i+1]
        emb = semantic_embeddings[i:i+1]
        videos = inf_tuneavideo(pipe, emb, z0,args.num_inference_steps, args.guidance_scale, device)
        
        os.makedirs(os.path.join(args.output_dir, f'Block{i//200}'), exist_ok=True)

        # Sauvegarde sans double rescale (déjà normalisé)
        save_videos_grid(
            videos,
            os.path.join(args.output_dir,f'Block{i//200}', f'{i%200+1}.gif'),
            rescale=False
        )
        print(f"[INFO] {i%200+1}.gif saved in {args.output_dir}")

if __name__ == '__main__':
    main()
