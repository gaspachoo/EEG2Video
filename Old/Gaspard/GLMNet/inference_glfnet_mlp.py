import os
import sys
import argparse
import pickle
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Old.EEG2Video.GLMNet.utils_glfnet_mlp import GLFNetMLP
from EEG2Video.GLMNet.modules.utils_glmnet import standard_scale_features

OCCIPITAL_IDX = list(range(50, 62))


def load_glfnet_mlp_from_checkpoint(ckpt_path, device="cuda"):
    state = torch.load(ckpt_path, map_location=device)
    if "fc.weight" in state:
        num_classes = state["fc.weight"].shape[0]
    elif "out.weight" in state:
        num_classes = state["out.weight"].shape[0]
    else:
        raise KeyError("Cannot infer class count from checkpoint")
    model = GLFNetMLP(OCCIPITAL_IDX, out_dim=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def inf_glfnet_mlp(model, scaler, feat_sw, device="cuda"):
    feat_flat = feat_sw.reshape(-1, feat_sw.shape[-2], feat_sw.shape[-1])
    feat_scaled = standard_scale_features(feat_flat, scaler=scaler)
    embeddings = []
    with torch.no_grad():
        for feat_seg in feat_scaled:
            x_feat = torch.tensor(feat_seg, dtype=torch.float32).unsqueeze(0).to(device)
            z = model(x_feat, return_features=True)
            embeddings.append(z.squeeze(0).cpu().numpy())
    return np.stack(embeddings)


def generate_all_embeddings(feat_dir, ckpt_path, scaler_path, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    scaler = load_scaler(scaler_path)
    model = load_glfnet_mlp_from_checkpoint(ckpt_path, device)
    for fname in os.listdir(feat_dir):
        if not (fname.endswith(".npy") and fname.startswith("sub3")):
            continue
        print(f"Processing {fname}...")
        subj = os.path.splitext(fname)[0]
        FEAT_SW = np.load(os.path.join(feat_dir, fname))
        embeddings = inf_glfnet_mlp(model, scaler, FEAT_SW, device)
        out_path = os.path.join(output_dir, f"{subj}.npy")
        np.save(out_path, embeddings)
        print(f"Saved embeddings for {subj}, shape {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", default="./data/Preprocessing/DE_500ms_sw", help="directory of feature .npy files")
    parser.add_argument("--checkpoint_path", default="./EEG2Video/checkpoints/glfnet_mlp/sub3_fold0_color_best.pt", help="path to checkpoint")
    parser.add_argument("--scaler_path", default="./EEG2Video/checkpoints/glfnet_mlp/sub3_fold0_color_scaler.pkl", help="path to saved scaler")
    parser.add_argument("--output_dir", default="./data/GLFNetMLP/EEG_embeddings_sw", help="where to save embeddings")
    args = parser.parse_args()

    generate_all_embeddings(args.feat_dir, args.checkpoint_path, args.scaler_path, args.output_dir)
