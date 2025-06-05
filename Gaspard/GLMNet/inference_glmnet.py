import os, sys
import torch
import numpy as np
import argparse
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from Gaspard.GLMNet.modules.utils_glmnet import GLMNet, standard_scale_features



OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels

# --- Load the pretrained GLMNet ---
def load_glmnet_from_checkpoint(ckpt_path, device='cuda'):
    model = GLMNet(OCCIPITAL_IDX, out_dim=40).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def inf_glmnet(model, scaler, raw_sw, feat_sw, device='cuda'):
       
    # verify consistency
    assert raw_sw.shape[:4] == feat_sw.shape[:4], \
        f"Raw windows {raw_sw.shape} and feat windows {feat_sw.shape} mismatch"

    # flatten for batch inference
    raw_flat = raw_sw.reshape(-1, raw_sw.shape[-2], raw_sw.shape[-1])
    feat_flat = feat_sw.reshape(-1, feat_sw.shape[-2], feat_sw.shape[-1])

    # scale features
    feat_scaled = standard_scale_features(feat_flat, scaler=scaler)

    embeddings = []
    with torch.no_grad():
        for raw_seg, feat_seg in zip(raw_flat, feat_scaled):
            x_raw = torch.tensor(raw_seg, dtype=torch.float32)
            x_raw = x_raw.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,62,100)
            x_feat = torch.tensor(feat_seg, dtype=torch.float32)
            x_feat = x_feat.unsqueeze(0).to(device)            # (1,62,5)

            g = model.raw_global(x_raw)                # (1, emb_dim)
            l = model.freq_local(x_feat[:, OCCIPITAL_IDX, :])  # (1, emb_dim)
            z = torch.cat([g, l], dim=1).squeeze(0).cpu().numpy()
            embeddings.append(z)

    return np.stack(embeddings)  # shape: (N_segments, emb_dim*2)

# --- Main generation loop ---
def generate_all_embeddings(raw_dir, feat_dir, ckpt_path, scaler_path, output_dir, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)

    scaler = load_scaler(scaler_path)

    for fname in os.listdir(raw_dir):
        if not fname.endswith('.npy'):
            continue
        print(f"Processing {fname}...")
        subj = os.path.splitext(fname)[0]

        # load pre-segmented windows
        RAW_SW = np.load(os.path.join(raw_dir, fname))
        FEAT_SW = np.load(os.path.join(feat_dir, fname))
        # expect shape: (7, 40, 5, 7, 62, 100) and (7, 40, 5, 7, 62, 5)
        model = load_glmnet_from_checkpoint(ckpt_path, device)
        embeddings = inf_glmnet(model, scaler, RAW_SW, FEAT_SW, device)
        
        out_path = os.path.join(output_dir, f"{subj}.npy")
        np.save(out_path, embeddings)
        print(f"Saved embeddings for {subj}, shape {embeddings.shape}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--raw_dir', default="./data/Preprocessing/Segmented_500ms_sw", help='directory of pre-windowed raw EEG .npy files')
    parser.add_argument('--feat_dir', default="./data/Preprocessing/DE_500ms_sw", help='directory of pre-windowed feature .npy files')
    parser.add_argument('--checkpoint_path', default="./Gaspard/checkpoints/glmnet/sub3_fold0_best.pt", help='path to GLMNet checkpoint')
    parser.add_argument('--scaler_path', default="./Gaspard/checkpoints/glmnet/sub3_fold0_scaler.pkl", help='path to saved StandardScaler')
    parser.add_argument('--output_dir', default="./data/GLMNet/EEG_embeddings_sw", help='where to save concatenated embeddings')
    args = parser.parse_args()
    generate_all_embeddings(args.raw_dir, args.feat_dir, args.checkpoint_path, args.scaler_path, args.output_dir)
