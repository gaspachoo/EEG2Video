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
    
def load_model(ckpt_path, device):
    model = Seq2SeqTransformer().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def inf_seq2seq(embeddings, video_latents, model, device):
    """
    embeddings: np.array shape (7,40,5,7,512)
    video_latents: np.array shape (200,6,4,36,64)
    model: loaded Seq2SeqTransformer
    returns predicted latents np.array shape (200,6,4,36,64)
    """
    
    # flatten concepts & repetitions -> (200,7,512)
    src = embeddings.reshape(-1, embeddings.shape[2], embeddings.shape[3])  # (200,7,512)
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
    parser.add_argument('--ckpt_file',      type=str,
                        default="./Gaspard/checkpoints/seq2seq/seq2seq_v2_classic.pth",
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
    
    model = load_model(args.ckpt_file, device)
    for block_id in range(7):

        vid_path = os.path.join(args.video_dir, f'block{block_id}.npy')
        z0 = np.load(vid_path)  # (200,6,4,36,64)

        print(f"Predicting latents for block {block_id}...")
        emb_block = emb_flat[block_id]             # (40,5,7,512)
        
        pred = inf_seq2seq(emb_block, z0, model, device)
        out_path = os.path.join(args.output_dir, f'block{block_id}.npy')
        

        np.save(out_path, pred)
        print(f"Saved predicted latents at {out_path}, shape {pred.shape}")

if __name__ == '__main__':
    main()
