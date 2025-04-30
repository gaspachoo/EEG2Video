import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

from models.models import ShallowNetEncoder, MLPEncoder_feat, GLMNetFeatureExtractor



def parse_args():
    parser = argparse.ArgumentParser()
    home=os.environ["HOME"]
    parser.add_argument("--eeg_raw_dir", type=str,default=f"{home}/Gaspard/EEG2Video/data/EEG_500ms_sw", help="Path to EEG raw segmented .npy files (7, 62, T)")
    parser.add_argument("--de_feat_dir", type=str,default=f"{home}/Gaspard/EEG2Video/data/DE_500ms_sw", help="Path to DE features .npy files (7, 62, 5)")
    parser.add_argument("--output_dir", type=str, default=f"{home}/Gaspard/EEG2Video/data/EEG_embeddings", help="Where to save EEG embeddings (7, 4096)")
    parser.add_argument("--g_ckpt", type=str, default=f"{home}/Gaspard/EEG2Video/Gaspard_model/checkpoints/cv_shallownet/best_fold0.pth", help="Path to ShallowNet checkpoint")
    parser.add_argument("--l_ckpt", type=str, default=f"{home}/Gaspard/EEG2Video/Gaspard_model/checkpoints/cv_mlp_DE/best_fold0.pt", help="Path to MLP checkpoint")
    return parser.parse_args()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load global encoder (ShallowNet)
    g_encoder = ShallowNetEncoder(62, time_len=128).to(device)
    g_encoder.load_state_dict(torch.load(args.g_ckpt, map_location=device)["encoder"])
    g_encoder.eval()

    # Load local encoder (MLP on DE features)
    l_encoder = MLPEncoder_feat(input_dim=62*5).to(device)
    l_encoder.load_state_dict(torch.load(args.l_ckpt, map_location=device), strict=False)
    l_encoder.eval()

    model = GLMNetFeatureExtractor(g_encoder, l_encoder).to(device)
    model.eval()

    eeg_files = sorted([f for f in os.listdir(args.eeg_raw_dir) if f.endswith(".npz")])
    de_files = sorted([f for f in os.listdir(args.de_feat_dir) if f.endswith(".npz")])

    assert len(eeg_files) == len(de_files), "Mismatch between EEG raw and DE feature files!"

    for eeg_f, de_f in tqdm(zip(eeg_files, de_files), total=len(eeg_files), desc="Generating embeddings"):
        eeg_path = os.path.join(args.eeg_raw_dir, eeg_f)
        de_path = os.path.join(args.de_feat_dir, de_f)

        eeg_npz = np.load(eeg_path)
        de_npz = np.load(de_path)

        eeg = eeg_npz["eeg"]  # (N, 62, T) — N=7 segments
        de = de_npz["de"]     # (N, 62, 5)

        eeg = torch.tensor(eeg, dtype=torch.float32, device=device)  # (7, 62, T)
        de = torch.tensor(de, dtype=torch.float32, device=device)    # (7, 62, 5)

        with torch.no_grad():
            features = model(eeg, de)  # (7, 4096)

        out_path = os.path.join(args.output_dir, eeg_f.replace(".npz", ".npy"))
        np.save(out_path, features.cpu().numpy())

if __name__ == "__main__":
    main(parse_args())