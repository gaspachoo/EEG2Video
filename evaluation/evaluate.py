"""Evaluation utilities for EEG2Video pipeline."""

import os
import glob
from typing import Sequence

from decord import VideoReader
from PIL import Image

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    from pytorch_fvd import FVD
except Exception:
    FVD = None

try:
    import torch
    import clip
except Exception:
    clip = None
    torch = None

from skimage.metrics import structural_similarity as ssim


def compute_latent_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute MSE, cosine similarity and R2 between predicted and true latents."""
    mse = mean_squared_error(gt, pred)
    cos = cosine_similarity(gt, pred).mean()
    r2 = r2_score(gt, pred)
    return {"mse": mse, "cosine": cos, "r2": r2}


def check_classification(pred: np.ndarray, gt_labels: Sequence[int],
                         train_latents: np.ndarray, train_labels: Sequence[int]) -> float:
    """Train a logistic regression on training latents and check accuracy on predictions."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_latents, train_labels)
    pred_labels = clf.predict(pred)
    return accuracy_score(gt_labels, pred_labels)


def compute_fvd(gen_dir: str, real_dir: str) -> float:
    """Compute Frechet Video Distance between folders of generated and real videos."""
    if FVD is None:
        raise ImportError("pytorch_fvd is required for FVD computation")
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
    return FVD()(gen_files, real_files)


def compute_ssim(gen_dir: str, real_dir: str) -> float:
    """Compute average SSIM over matching video frames."""
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
    if len(gen_files) != len(real_files):
        raise ValueError("Number of generated and real videos must match")
    scores = []
    for gf, rf in zip(gen_files, real_files):
        gvid = VideoReader(gf)
        rvid = VideoReader(rf)
        min_len = min(len(gvid), len(rvid))
        for i in range(min_len):
            gframe = gvid[i].asnumpy()
            rframe = rvid[i].asnumpy()
            scores.append(ssim(gframe, rframe, channel_axis=-1))
    return float(np.mean(scores))


def compute_clip_sim(gen_dir: str, real_dir: str, device: str = "cpu") -> float:
    """Compute average CLIP similarity between generated and real videos."""
    if clip is None or torch is None:
        raise ImportError("openai clip and torch are required for CLIP-SIM")
    model, preprocess = clip.load("ViT-B/32", device=device)
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
    sims = []
    for gf, rf in zip(gen_files, real_files):
        gvid = VideoReader(gf)
        rvid = VideoReader(rf)
        gimg = preprocess(Image.fromarray(gvid[0].asnumpy())).unsqueeze(0).to(device)
        rimg = preprocess(Image.fromarray(rvid[0].asnumpy())).unsqueeze(0).to(device)
        with torch.no_grad():
            gfeat = model.encode_image(gimg)
            rfeat = model.encode_image(rimg)
            gfeat = gfeat / gfeat.norm(dim=-1, keepdim=True)
            rfeat = rfeat / rfeat.norm(dim=-1, keepdim=True)
        sims.append((gfeat * rfeat).sum().item())
    return float(np.mean(sims))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate generated videos and latents")
    parser.add_argument("--pred_latent", required=True, help="Path to predicted latents .npy")
    parser.add_argument("--gt_latent", required=True, help="Path to ground truth latents .npy")
    parser.add_argument("--train_latent", required=True, help="Path to training latents .npy")
    parser.add_argument("--train_labels", required=True, help="Path to training labels .npy")
    parser.add_argument("--labels", required=True, help="Path to labels for predicted latents .npy")
    parser.add_argument("--gen_videos", required=True, help="Folder with generated videos")
    parser.add_argument("--real_videos", required=True, help="Folder with ground truth videos")
    args = parser.parse_args()

    pred = np.load(args.pred_latent)
    gt = np.load(args.gt_latent)
    train_lat = np.load(args.train_latent)
    train_labels = np.load(args.train_labels)
    labels = np.load(args.labels)

    metrics = compute_latent_metrics(pred, gt)
    acc = check_classification(pred, labels, train_lat, train_labels)
    fvd = compute_fvd(args.gen_videos, args.real_videos)
    ssim_score = compute_ssim(args.gen_videos, args.real_videos)
    clip_score = compute_clip_sim(args.gen_videos, args.real_videos)

    print("Latent metrics:", metrics)
    print("Classification accuracy:", acc)
    print("FVD:", fvd)
    print("SSIM:", ssim_score)
    print("CLIP-SIM:", clip_score)


if __name__ == "__main__":
    main()
