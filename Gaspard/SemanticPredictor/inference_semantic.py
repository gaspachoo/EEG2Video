import os,sys
import torch
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Gaspard.SemanticPredictor.models.clip import CLIP

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def load_semantic_predictor(model_path, device):
    model = CLIP().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def inf_semantic_predictor(model, eeg_data, device='cuda'):
    """Generate semantic embeddings for EEG data using a pretrained model.

    Parameters
    ----------
    model : nn.Module
        Pretrained SemanticPredictor model.
    eeg_data : np.ndarray
        EEG data with shape (blocks, concepts, trials, channels, windows).
    batch_size : int
        Batch size for inference.
    device : torch.device
        Device to run the model on.

    Returns
    -------
    np.ndarray
        Semantic embeddings of shape (blocks, concepts, trials, embedding_dim).
    """
    c, t, ch, w = eeg_data.shape
    eeg_reshaped = eeg_data.reshape(c * t, ch * w).astype(np.float32)
    scaler = StandardScaler().fit(eeg_reshaped)
    flat = eeg_data.reshape(-1, ch * w)
    std = scaler.transform(flat)

    embs = []
    with torch.no_grad():
        for i in range(0, std.shape[0]):
            batch = torch.from_numpy(std[i]).float().to(device)
            out = model(batch)
            embs.append(out.cpu().numpy())
    emb_block = np.vstack(embs)  # (c*t, 77*768)
    return emb_block
    

if __name__ == '__main__':
    seed_everything()
    # default paths
    
    default_eeg = "./data/Preprocessing/DE_1per2s/sub1.npy"
    default_model = "./Gaspard/checkpoints/semantic/eeg2text_clip.pt"
    default_save = "./data/SemanticPredictor/Semantic_embeddings"

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


    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_semantic_predictor(args.model_path, device)
    
    # fit scaler
    eeg = np.load(args.eeg_file)
    if eeg.ndim != 5:
        raise ValueError(f"Expected EEG ndim=5, got {eeg.shape}")

    # process blocks
    for bi in tqdm(range(eeg.shape[0]), desc="Processing blocks"):
        block = eeg[bi]  # (concepts, trials, ch, w)
        emb_block = inf_semantic_predictor(model, block, device=device)
        save_path = os.path.join(args.save_dir, f"block{bi}.npy")
        np.save(save_path, emb_block)
        print(f"Saved block {bi} embeddings to {save_path}")
    print("Done generating semantic embeddings.")
