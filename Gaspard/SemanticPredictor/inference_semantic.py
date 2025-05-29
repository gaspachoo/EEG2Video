import os
import torch
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from models import CLIP

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything()
    # default paths
    
    default_eeg = "./data/Preprocessing/DE_1per2s/sub1.npy"
    default_model = "./Gaspard/checkpoints/semantic/eeg2text_clip.pt"
    default_save = "./data/Semantic_embeddings"

    parser = argparse.ArgumentParser(
        description="Generate semantic embeddings per block using pretrained Semantic Predictor"
    )
    parser.add_argument('--eeg_file', type=str, default=default_eeg,
                        help='EEG .npy file (blocks, concepts, trials, channels, windows)')
    parser.add_argument('--model_path', type=str, default=default_model,
                        help='Path to pretrained SemanticPredictor (eeg2text_clip.pt)')
    parser.add_argument('--save_dir', type=str, default=default_save,
                        help='Output directory for block-wise embeddings (.npy)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    eeg = np.load(args.eeg_file)
    if eeg.ndim != 5:
        raise ValueError(f"Expected EEG ndim=5, got {eeg.shape}")
    b, c, t, ch, w = eeg.shape

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIP().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # fit scaler
    all_eeg = eeg.reshape(b * c * t, ch * w).astype(np.float32)
    scaler = StandardScaler().fit(all_eeg)

    # process blocks
    for bi in range(b):
        block = eeg[bi]  # (concepts, trials, ch, w)
        flat = block.reshape(-1, ch * w)
        std = scaler.transform(flat)

        embs = []
        with torch.no_grad():
            for i in range(0, std.shape[0], args.batch_size):
                batch = torch.from_numpy(std[i:i+args.batch_size]).float().to(device)
                out = model(batch)
                embs.append(out.cpu().numpy())
        emb_block = np.vstack(embs)  # (c*t, 77*768)

        save_path = os.path.join(args.save_dir, f"block{bi}.npy")
        np.save(save_path, emb_block)
        print(f"Saved block {bi} embeddings to {save_path}")

    print("Done generating semantic embeddings.")
