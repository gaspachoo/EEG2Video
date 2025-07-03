import os, sys
import torch
import numpy as np
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from EEGtoVideo.GLMNet.modules.utils_glmnet import (
    GLMNet,
    standard_scale_features,
    normalize_raw,
    load_scaler,
    load_raw_stats,
)
from EEGtoVideo.GLMNet.modules.models_paper import mlpnet


OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels


def inf_glmnet(model, scaler, raw_sw, stats, device="cuda"):

    # always compute spectral features from the raw windows
    raw_flat = raw_sw.reshape(-1, raw_sw.shape[-2], raw_sw.shape[-1])
    feat_sw = mlpnet.compute_features(raw_flat)
    # reshape back to (runs, videos, trials, windows, channels, features)
    feat_sw = feat_sw.reshape(raw_sw.shape[:-2] + feat_sw.shape[-2:])

    # flatten for batch inference
    raw_flat = raw_sw.reshape(-1, raw_sw.shape[-2], raw_sw.shape[-1])
    feat_flat = feat_sw.reshape(-1, feat_sw.shape[-2], feat_sw.shape[-1])

    raw_flat = normalize_raw(raw_flat, stats[0], stats[1])

    # scale features
    feat_scaled = standard_scale_features(feat_flat, scaler=scaler)

    embeddings = []
    with torch.no_grad():
        for raw_seg, feat_seg in zip(raw_flat, feat_scaled):
            x_raw = torch.tensor(raw_seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            x_feat = torch.tensor(feat_seg, dtype=torch.float32).unsqueeze(0).to(device)

            z = model(x_raw, x_feat, return_features=True)
            embeddings.append(z.squeeze(0).cpu().numpy())

    return np.stack(embeddings)  # shape: (N_segments, emb_dim // 2)

# --- Main generation loop ---
def generate_all_embeddings(
    raw_dir,
    ckpt_path,
    output_dir,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)

    scaler_path = os.path.join(ckpt_path, "scaler.pkl")
    stats_path = os.path.join(ckpt_path, "raw_stats.npz")
    model_path = os.path.join(ckpt_path, "glmnet_best.pt")
    
    scaler = load_scaler(scaler_path)
    stats = load_raw_stats(stats_path)

    for fname in os.listdir(raw_dir):
        if not (fname.endswith('.npy') and fname.startswith('sub3')):
            continue
        print(f"Processing {fname}...")
        subj = os.path.splitext(fname)[0]

        # load pre-segmented windows
        RAW_SW = np.load(os.path.join(raw_dir, fname))
        # expect shape: (7, 40, 5, 7, 62, T)
        time_len = RAW_SW.shape[-1]
        num_channels = RAW_SW.shape[-2]
        model = GLMNet.load_from_checkpoint(
            model_path, OCCIPITAL_IDX, C = num_channels, T= time_len, device=device
        )
        embeddings = inf_glmnet(model, scaler, RAW_SW, stats, device)
        
        out_path = os.path.join(output_dir, f"{subj}.npy")
        np.save(out_path, embeddings)
        print(f"Saved embeddings for {subj}, shape {embeddings.shape}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--raw_dir', default="./data/Preprocessing/Segmented_500ms_sw", help='directory of pre-windowed raw EEG .npy files')
    parser.add_argument('--checkpoint_path', default="./EEGtoVideo/checkpoints/glmnet/sub3_label_cluster", help='path to GLMNet checkpoint')
    parser.add_argument('--output_dir', default="./data/eeg_segments", help='where to save projected embeddings')
    args = parser.parse_args()
    generate_all_embeddings(
        args.raw_dir,
        args.checkpoint_path,
        args.output_dir,
    )
