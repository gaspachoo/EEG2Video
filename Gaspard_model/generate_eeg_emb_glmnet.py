import os
import torch
import numpy as np
from Gaspard_model.train_glmnet import GLMNet, OCCIPITAL_IDX, RAW_T, split_raw_2s_to_1s,standard_scale_features
import argparse

def load_glmnet_from_checkpoint(checkpoint_path, device='cuda'):
    model = GLMNet(out_dim=40).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def generate_all_embeddings(raw_dir, feat_dir, checkpoint_path, output_dir, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    model = load_glmnet_from_checkpoint(checkpoint_path, device)

    for fname in os.listdir(raw_dir):
        if not fname.endswith(".npy") : continue 
        if fname != 'sub3.npy': continue ## edit this if you want to process all files
        print(f"Processing {fname}...")
        subj = fname.replace(".npy", "")
        raw2s = np.load(os.path.join(raw_dir, fname))  # (7,40,5,62,400)
        feat = np.load(os.path.join(feat_dir, fname))  # (7,40,5,62,5)

        raw1s = split_raw_2s_to_1s(raw2s)  # (7,40,5,2,62,200)
        raw = raw1s.reshape(-1, 62, 200)  # (N, 62, 200)
        feat = feat.reshape(-1, 62, 5)    # (N, 62, 5)
        feat_scaled = standard_scale_features(feat)

        with torch.no_grad():
            z_all = []
            for i in range(len(raw)):
                x_raw = torch.tensor(raw[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,62,200)
                x_feat = torch.tensor(feat_scaled[i], dtype=torch.float32).unsqueeze(0).to(device)  # (1,62,5)
                z = model.raw_global(x_raw)
                l = model.freq_local(x_feat[:, OCCIPITAL_IDX, :])
                z_cat = torch.cat([z, l], dim=1)  # (1, 512)
                z_all.append(z_cat.squeeze(0).cpu().numpy())
            z_all = np.stack(z_all)
            np.save(os.path.join(output_dir, f"{subj}.npy"), z_all)
            print(f"Saved: {subj}.npy")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/EEG2Video" #"/Documents/School/Centrale Med/2A/SSE/EEG2Video"
    parser.add_argument("--raw_dir",  default = f"{root}/data/Segmented_Rawf_200Hz_2s", help="directory with .npy files") 
    parser.add_argument("--feat_dir", default=f"{root}/data/DE_1per1s/", help="directory with .npy files")
    parser.add_argument("--checkpoint_path", default=f"{root}/Gaspard_model/checkpoints/cv_glmnetv2/sub3_fold0_best.pt", help="checkpoint path")
    parser.add_argument('--output_dir', default=f"{root}/data/EEG_embeddings/", help="Where to save EEG embeddings")
    args = parser.parse_args()

    generate_all_embeddings(args.raw_dir, args.feat_dir, args.checkpoint_path, args.output_dir)
