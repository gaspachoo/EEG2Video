import os, sys
import argparse
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Import Seq2SeqTransformer defined in your training script
from Gaspard.Seq2Seq.models.transformer import Seq2SeqTransformer
    
def load_s2s_from_checkpoint(ckpt_path, device):
    model = Seq2SeqTransformer().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def inf_seq2seq(model, embeddings, device, seq_len=6):
    """Run autoregressive inference without using ground truth latents.

    Parameters
    ----------
    embeddings : np.ndarray
        EEG embeddings with shape ``(7, 40, 5, 7, 512)``.
    model : Seq2SeqTransformer
        Trained seq2seq model.
    device : torch.device
        Device on which to run the model.
    seq_len : int, optional
        Number of latent frames to generate (default: 6).

    Returns
    -------
    np.ndarray
        Predicted video latents of shape ``(200, seq_len, 4, 36, 64)``.
    """

    # flatten concepts & repetitions -> (200, 7, 512)
    src = embeddings.reshape(-1, embeddings.shape[-2], embeddings.shape[-1])
    src = torch.from_numpy(src).float().to(device)
    batch_size = src.size(0)
    tgt_pred = torch.zeros(batch_size, seq_len, 9216, device=device)

    with torch.no_grad():
        for t in range(seq_len):
            out = model(src, tgt_pred)
            tgt_pred[:, t] = out[:, t]

    pred = tgt_pred.view(batch_size, seq_len, 4, 36, 64)
    return pred.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--emb_path',      type=str,
                        default="./data/GLMNet/EEG_embeddings_sw/sub3.npy",
                        help='Path to embeddings .npy (shape 7*40*5*7,512)')
    parser.add_argument('--ckpt_file',      type=str,
                        default="./Gaspard/checkpoints/seq2seq/seq2seq_v2_color.pth",
                        help='Path of the model checkpoint file .pth')
    parser.add_argument('--output_dir',    type=str,
                        default="./data/Seq2Seq/Predicted_latents/",
                        help='Where to save predicted latents')
    parser.add_argument('--device',        default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--stats_path',    type=str,
                        help='Directory containing stats.npz with mean/std')
    args = parser.parse_args()

    stats = None
    if args.stats_path:
        stats_file = os.path.join(args.stats_path, 'stats.npz')
        if os.path.isfile(stats_file):
            stats = np.load(stats_file)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    # load full embeddings and reshape
    all_emb = np.load(args.emb_path)  # (7*40*5*7, 512)
    emb_flat = all_emb.reshape(7, 40, 5, 7, 512)
    
    model = load_s2s_from_checkpoint(args.ckpt_file, device)
    for block_id in range(7):

        print(f"Predicting latents for block {block_id}...")
        emb_block = emb_flat[block_id]             # (40,5,7,512)

        pred = inf_seq2seq(model, emb_block, device)

        if stats is not None:
            mean = stats['mean']
            std = stats['std']
            pred = pred * std + mean

        out_path = os.path.join(args.output_dir, f'block{block_id}.npy')

        np.save(out_path, pred)
        print(f"Saved predicted latents at {out_path}, shape {pred.shape}")

if __name__ == '__main__':
    main()
