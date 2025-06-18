import os
import sys
import argparse
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EEG2Video.Seq2Seq.models.my_autoregressive_transformer import myTransformer

# class mapping used during training
GT_LABEL = np.load("./data/meta_info/All_video_label.npy")
CHOSEN_LABELS = list(range(1, 41))


def load_model(ckpt_path: str, device: torch.device) -> myTransformer:
    model = myTransformer().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_eeg(eeg_path: str):
    eeg = np.load(eeg_path)

    # If the file already contains sliding windows, keep them
    if eeg.ndim == 6:
        windows = eeg  # (7, 40, 5, 7, 62, 100)
    else:
        # Otherwise create the windows as in training
        win = 100
        step = 50

        reordered = np.zeros_like(eeg)
        for blk in range(7):
            idx = [list(GT_LABEL[blk]).index(lbl) for lbl in CHOSEN_LABELS]
            reordered[blk] = eeg[blk][idx]

        win_list = []
        for start in range(0, reordered.shape[-1] - win + 1, win - step):
            win_list.append(reordered[..., start:start + win])
        windows = np.stack(win_list, axis=-1)  # (7,40,5,62,100,7)

    # When coming from segment_sliding_window the layout is (7,40,5,7,62,100)
    if windows.shape[-3] == 62:
        windows = windows.transpose(0, 1, 2, 4, 5, 3)

    data = windows.reshape(-1, 62, 100, windows.shape[-1])
    b, c, l, f = data.shape
    data = data.reshape(b, -1)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data.reshape(b, c, l, f)

    data = np.transpose(data, (0, 3, 1, 2))  # (b,7,62,100)
    return torch.from_numpy(data).float(), scaler


def generate_latents(model: myTransformer, eeg: torch.Tensor, device: torch.device, batch_size: int = 32):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, eeg.size(0), batch_size):
            src = eeg[i:i + batch_size].to(device)
            tgt = torch.zeros(src.size(0), 7, 4, 36, 64, device=device)
            _, out = model(src, tgt)
            preds.append(out[:, :-1].cpu())
    return torch.cat(preds, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="Inference for myTransformer")
    parser.add_argument('--ckpt', type=str, default = "./EEG2Video/checkpoints/Seq2Seq_v2/best.pt", help='Model checkpoint')
    parser.add_argument(
        '--eeg_path',
        type=str,
        default = './data/Preprocessing/Segmented_500ms_sw/sub3.npy',
        help='Path to an EEG file from data/Preprocessing/Segmented_500ms_sw'
    )
    parser.add_argument('--output_dir', type=str, default = "./data/Seq2Seq/Latents_autoreg", help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    model = load_model(args.ckpt, device)
    eeg, _ = preprocess_eeg(args.eeg_path)

    preds = generate_latents(model, eeg, device, args.batch_size)
    preds = preds.reshape(7, 40, 5, 6, 4, 36, 64)

    for blk in range(7):
        out_path = os.path.join(args.output_dir, f'block{blk}.npy')
        np.save(out_path, preds[blk])
        print(f'Saved block {blk} latents to {out_path} with shape {preds[blk].shape}')


if __name__ == '__main__':
    main()
