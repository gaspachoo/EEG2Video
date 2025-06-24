"""Inference script for the autoregressive transformer.

The EEG encoder can be switched between **EEGNet**, **ShallowNet** or
**MLPNet** using ``--eeg_encoder``. Pretrained weights for ShallowNet or
MLPNet can be loaded via ``--encoder_ckpt``.
"""

import os
import sys
import argparse
import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EEG2Video.Seq2Seq.models.my_autoregressive_transformer import myTransformer

# class mapping used during training
GT_LABEL = np.load("./data/meta_info/All_video_label.npy")
CHOSEN_LABELS = list(range(1, 41))

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for myTransformer")
    parser.add_argument('--ckpt_dir', type=str, default = "./EEG2Video/checkpoints/Seq2Seq_v2", help='Model checkpoint')
    parser.add_argument('--eeg_path',type=str, default = './data/Preprocessing/Segmented_500ms_sw/sub3.npy', help='Path to an EEG file from data/Preprocessing/Segmented_500ms_sw')
    parser.add_argument('--output_dir', type=str, default = "./data/Seq2Seq/Latents_autoreg", help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eeg_encoder', choices=['eegnet', 'shallownet', 'mlpnet','glmnet'],default='eegnet', help='EEG encoder type')
    parser.add_argument('--encoder_ckpt', type=str,default="EEG2Video/checkpoints/glmnet/sub3_label_cluster", help='Path to pretrained encoder weights')
    return parser.parse_args()

def load_scaler(path: str) -> StandardScaler:
    """Load a fitted ``StandardScaler`` saved with ``pickle``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(ckpt_path: str, device: torch.device,
               eeg_encoder: str = 'eegnet', encoder_ckpt: str | None = None,
               C: int | None = None, T: int | None = None) -> myTransformer:
    """Load ``myTransformer`` from ``ckpt_path`` using the chosen EEG encoder."""
    model = myTransformer(
        eeg_encoder=eeg_encoder,
        encoder_ckpt=encoder_ckpt,
        C=C,
        T=T,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
        model.load_state_dict(state)
    model.eval()
    return model


def load_eeg_data(raw_path: str, scaler: StandardScaler) -> torch.Tensor:
    """Load EEG data from ``raw_path`` and apply scaling."""
    raw = np.load(raw_path).astype(np.float32)
    raw = raw.reshape(-1, *raw.shape[-3:])
    raw_flat = raw.reshape(len(raw), -1)
    raw_scaled = scaler.transform(raw_flat).reshape(raw.shape)
    return torch.from_numpy(raw_scaled)

def generate_latents(model: myTransformer, eeg: torch.Tensor, device: torch.device, batch_size: int = 32):
    preds = []
    model.eval()
    with torch.no_grad():
        print(eeg.shape) # 1400, 7, 62, 100
        for i in range(0, eeg.size(0), batch_size):
            src = eeg[i:i + batch_size].to(device)
            tgt = torch.zeros(src.size(0), 7, 4, 36, 64, device=device)
            _, out = model(src, tgt)
            preds.append(out[:, :-1].cpu())
    return torch.cat(preds, dim=0).numpy()


def main():
    args= parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    scaler_path = os.path.join(args.ckpt_dir, f"scaler_{args.eeg_encoder}.pkl")
    model_ckpt_path = os.path.join(args.ckpt_dir, f"best_{args.eeg_encoder}.pt")
    scaler = load_scaler(scaler_path)
    eeg_raw = load_eeg_data(args.eeg_path, scaler)
    
    C, T = eeg_raw[0].shape[-2:]
    
    model = load_model(
        model_ckpt_path,
        device,
        eeg_encoder=args.eeg_encoder,
        encoder_ckpt=args.encoder_ckpt,
        C=C,
        T=T
        )
    preds = generate_latents(model, eeg_raw, device, args.batch_size)
    preds = preds.reshape(7, 40*5, 6, 4, 36, 64)
    
    for blk in range(7):
        out_path = os.path.join(args.output_dir, f'block{blk}.npy')
        np.save(out_path, preds[blk])
        print(f'Saved block {blk} latents to {out_path} with shape {preds[blk].shape}')


if __name__ == '__main__':
    main()
