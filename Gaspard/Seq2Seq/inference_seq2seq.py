import os
import argparse
import torch
import numpy as np

# Import Seq2SeqTransformer defined in your training script
from models.transformer import Seq2SeqTransformer


def load_model(ckpt_path, device):
    model = Seq2SeqTransformer().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_block_latents(block_id, embeddings, video_latents, model, device):
    """
    embeddings: np.array shape (7,40,5,7,512)
    video_latents: np.array shape (200,6,4,36,64)
    model: loaded Seq2SeqTransformer
    returns predicted latents np.array shape (200,6,4,36,64)
    """
    # select embeddings for this block
    emb_block = embeddings[block_id]             # (40,5,7,512)
    # flatten concepts & repetitions -> (200,7,512)
    src = emb_block.reshape(-1, emb_block.shape[2], emb_block.shape[3])  # (200,7,512)
    src = torch.from_numpy(src).float().to(device)

    # prepare target latents as teacher input
    tgt = torch.from_numpy(video_latents).float().to(device)  # (200,6,4,36,64)
    # flatten spatial dims for decoder input
    tgt_flat = tgt.reshape(tgt.shape[0], tgt.shape[1], -1)    # (200,6,9216)

    with torch.no_grad():
        pred_flat = model(src, tgt_flat)  # (200,6,9216)
    # reshape back to (200,6,4,36,64)
    pred = pred_flat.view(tgt.shape)
    return pred.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--emb_path',      type=str,
                        default="./data/GLMNet/EEG_embeddings_sw/sub3.npy",
                        help='Path to embeddings .npy (shape 7*40*5*7,512)')
    parser.add_argument('--video_dir',     type=str,
                        default="./data/Seq2Seq/Video_latents",
                        help='Directory with block{block}_latents.npy (shape 200,6,4,36,64)')
    parser.add_argument('--ckpt_dir',      type=str,
                        default="./Gaspard/checkpoints/seq2seq/",
                        help='Directory with seq2seq_sw_block{block}.pth')
    parser.add_argument('--output_dir',    type=str,
                        default="./data/Seq2Seq/Predicted_latents/",
                        help='Where to save predicted latents')
    parser.add_argument('--device',        default='cuda',
                        help='cuda or cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    # load full embeddings and reshape
    all_emb = np.load(args.emb_path)  # (7*40*5*7, 512)
    emb_flat = all_emb.reshape(7, 40, 5, 7, 512)

    for block_id in range(7):
        ckpt_path = os.path.join(args.ckpt_dir, f'seq2seq_sw_block{block_id}.pth')
        model = load_model(ckpt_path, device)

        vid_path = os.path.join(args.video_dir, f'block{block_id}_latents.npy')
        z0 = np.load(vid_path)  # (200,6,4,36,64)

        print(f"Predicting latents for block {block_id}...")
        pred = predict_block_latents(block_id, emb_flat, z0, model, device)
        out_path = os.path.join(args.output_dir, f'block{block_id}_predicted_latents.npy')
        
        # --- DEBUG Seq2Seq preds ---
        # pred : np.ndarray shape (B_i, 77*768) ou (B_i, 77,768)
        #print(f"[DEBUG Seq2Seq] block {block_id}: shape", pred.shape, "mean", pred.mean(), "std", pred.std(), "min", pred.min(), "max", pred.max())
        # --------------------------------
        # → Normalisation per-sample zero-mean / unit-std
        # On réduit la moyenne et on divise par l’écart-type pour chaque instance
        # utilises axes (1,2,3,4) pour couvrir tous les dims sauf le batch
        mean = pred.mean(axis=(1,2,3,4), keepdims=True)
        std  = pred.std(axis=(1,2,3,4), keepdims=True) + 1e-6
        #pred = (pred - mean) / std ----------------------------------------to edit

        # DEBUG après normalisation
        #print(f"[DEBUG Seq2Seq post-norm] mean {pred.mean():.4f}, std {pred.std():.4f}, min {pred.min():.4f}, max {pred.max():.4f}")

        
        np.save(out_path, pred)
        print(f"Saved predicted latents at {out_path}, shape {pred.shape}")

if __name__ == '__main__':
    main()
