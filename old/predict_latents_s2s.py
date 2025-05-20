import os
import torch
import numpy as np
from Gaspard_model.old.train_seq2seq import Seq2SeqTransformer
from tqdm import tqdm

@torch.no_grad()
def predict_latents(ckpt_path, z_hat_path, output_path, device="cuda"):
    # Load model
    model = Seq2SeqTransformer().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Load z_hat embeddings (7, 40, 5, 2, 512)
    z_hat = np.load(z_hat_path)
    print(f"Loaded z_hat: {z_hat.shape}")

    z_hat = z_hat.reshape(7, 40, 5, 2, 512)
    z_hat = z_hat[..., 0, :]  # Keep only first 1s EEG
    z_hat = z_hat.transpose(1, 2, 0, 3)  # (40, 5, 7, 512)
    z_hat = z_hat.reshape(-1, 7, 512)  # (200, 7, 512)

    z0_pred = model(torch.tensor(z_hat, dtype=torch.float32).to(device), torch.zeros((200, 6, 9216), device=device))  # (200, 6, 9216)
    z0_pred = z0_pred.cpu().numpy().reshape(200, 6, 4, 36, 64)

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "block0_latents.npy")
    np.save(save_path, z0_pred)
    print(f"Saved predicted latents to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/EEG2Video"
    parser.add_argument('--z_hat', type=str, default=f"{root}/data/EEG_embeddings/sub3.npy")
    parser.add_argument('--ckpt', type=str, default=f"{root}/Gaspard_model/checkpoints/seq2seq/seq2seq_block0.pth")
    parser.add_argument('--output', type=str, default=f"{root}/data/Predicted_latents")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    predict_latents(args.ckpt, args.z_hat, args.output, args.device)