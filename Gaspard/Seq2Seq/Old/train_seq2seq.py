import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import wandb
import argparse
from models.transformer import Seq2SeqTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--sub_emb',       type=str,
                        default="./data/GLMNet/EEG_embeddings_sw/sub3.npy",
                        help='EEG embeddings (.npy) path')
    parser.add_argument('--video_dir',     type=str,
                        default="./data/Seq2Seq/Video_latents",
                        help='Directory with block{block_id}.npy')
    parser.add_argument('--save_path',     type=str,
                        default="./Gaspard/checkpoints/seq2seq/",
                        help='Where to save models')
    
    # Training parameters
    parser.add_argument('--epochs',        type=int,   default=200)
    parser.add_argument('--batch_size',    type=int,   default=64)
    
    #Learning rate parameters
    parser.add_argument('--lr',            type=float, default=1e-4)
    
    parser.add_argument('--scheduler', type=str, default='cosine',
                    choices=['cosine', 'step', 'plateau'],
                    help='Type of LR scheduler to use')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Decay factor for StepLR or ExponentialLR')

    # WandB logging
    parser.add_argument('--use_wandb',     action='store_true', help='Log to wandb')
    return parser.parse_args()


def load_and_prepare_data(args):
    # 1. Charger embeddings EEG
    sub_emb = np.load(args.sub_emb)  # shape attendu : soit (7,40,5,7,512) soit (9800,512)

    # Gérer l’éventuel flatten des embeddings
    if sub_emb.ndim == 5:
        # Format original : (7,40,5,7,512)
        num_blocks, num_concepts, num_repeats, seq_len, d_eeg = sub_emb.shape
        all_eeg = sub_emb.reshape(-1, seq_len, d_eeg)  # (1400,7,512)
    elif sub_emb.ndim == 2:
        # Format déjà aplati : (7*40*5*7,512) = (9800,512)
        total_timepoints, d_eeg = sub_emb.shape
        # Regrouper chaque 7 timepoints en un échantillon de longueur 7
        if total_timepoints % 7 != 0:
            raise ValueError(f"Le nombre de lignes {total_timepoints} n'est pas divisible par 7 pour reshaper en (N,7,512)")
        all_eeg = sub_emb.reshape(-1, 7, d_eeg)       # (1400,7,512)
        seq_len = 7
    else:
        raise ValueError(f"sub_emb.ndim inattendu : {sub_emb.ndim}, attendu 2 ou 5.")

    # 2. Charger tous les latents vidéo et concaténer
    z0_list = []
    num_blocks = 7
    for b in range(num_blocks):
        path_zb = os.path.join(args.video_dir, f"block{b}.npy")
        if not os.path.isfile(path_zb):
            raise FileNotFoundError(f"Fichier introuvable : {path_zb}")
        z0b = np.load(path_zb)  # shape : (200,6,4,36,64)
        z0_list.append(z0b)
    z0_all = np.concatenate(z0_list, axis=0)    # (1400,6,4,36,64)

    # 3. Calcul de mean/std globaux si normalisation demandée
    if args.normalize:
        # Calculer mean et std sur axes (0,2,3,4) -> shape (1,6,1,1,1)
        mean_z = z0_all.mean(axis=(0,2,3,4), keepdims=True)
        std_z  = z0_all.std(axis=(0,2,3,4), keepdims=True) + 1e-6
        # Appliquer normalisation
        z0_all_norm = (z0_all - mean_z) / std_z
        # Aplatir vers (1400,6,9216)
        z0_flat_norm = z0_all_norm.reshape(z0_all_norm.shape[0], 6, -1)
    else:
        mean_z, std_z = None, None
        # Aplatir les latents bruts sans normalisation
        z0_flat_norm = z0_all.reshape(z0_all.shape[0], 6, -1)

    # 4. Construire TensorDataset
    eeg_tensor = torch.tensor(all_eeg, dtype=torch.float32)      # (1400,7,512)
    vid_tensor = torch.tensor(z0_flat_norm, dtype=torch.float32)  # (1400,6,9216)
    dataset = TensorDataset(eeg_tensor, vid_tensor)

    # 5. Répartition train / val / test
    total_samples = len(dataset)
    train_size = int(args.train_ratio * total_samples)
    val_size   = int(args.val_ratio   * total_samples)
    test_size  = total_samples - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # 6. DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)

    return train_loader, val_loader, test_loader, mean_z, std_z

# ------------------ Entraînement Seq2Seq ------------------

def train_seq2seq(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger et préparer les données
    train_loader, val_loader, test_loader, mean_z, std_z = load_and_prepare_data(args)

    # Initialiser le modèle Transformer
    model = Seq2SeqTransformer().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=args.min_lr
    )

    criterion_mse = nn.MSELoss()

    # Si wandb activé
    if args.use_wandb:
        wandb.init(project="seq2seq_eeg2video_v2", config=vars(args))

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # ----------- Phase d'entraînement -----------
        model.train()
        running_train_loss = 0.0
        for EEG_emb, z0_flat in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            EEG_emb = EEG_emb.to(device)            # (B,7,512)
            z0_flat = z0_flat.to(device)            # (B,6,9216)

            optimizer.zero_grad()
            # Forward Seq2Seq -> prédire latent normalisé ou brut
            z_pred_flat = model(EEG_emb, z0_flat)            # (B,6,9216)

            # Dé-normalisation si nécessaire
            if args.normalize:
                z_pred_5d = z_pred_flat.view(-1, 6, 4, 36, 64) * std_z + mean_z
                z_true_5d = z0_flat.view(-1, 6, 4, 36, 64) * std_z + mean_z
            else:
                z_pred_5d = z_pred_flat.view(-1, 6, 4, 36, 64)
                z_true_5d = z0_flat.view(-1, 6, 4, 36, 64)

            # Décoder via le VAE frame-by-frame
            B, T, C, H, W = z_pred_5d.shape
            z_pred_frames = z_pred_5d.view(B * T, C, H, W)
            z_true_frames = z_true_5d.view(B * T, C, H, W)
            img_pred = model.vae.decode(z_pred_frames)
            img_true = model.vae.decode(z_true_frames)

            # Calcul de la loss
            loss = criterion_mse(z_pred_flat, z0_flat)

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # ----------- Phase de validation -----------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for EEG_emb, z0_flat in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                EEG_emb = EEG_emb.to(device)
                z0_flat = z0_flat.to(device)

                z_pred_flat = model(EEG_emb)
                if args.normalize:
                    z_pred_5d = z_pred_flat.view(-1, 6, 4, 36, 64) * std_z + mean_z
                    z_true_5d = z0_flat.view(-1, 6, 4, 36, 64) * std_z + mean_z
                else:
                    z_pred_5d = z_pred_flat.view(-1, 6, 4, 36, 64)
                    z_true_5d = z0_flat.view(-1, 6, 4, 36, 64)

                B, T, C, H, W = z_pred_5d.shape
                z_pred_frames = z_pred_5d.view(B * T, C, H, W)
                z_true_frames = z_true_5d.view(B * T, C, H, W)
                img_pred = model.vae.decode(z_pred_frames)
                img_true = model.vae.decode(z_true_frames)

                loss = criterion_mse(z_pred_flat, z0_flat)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # Logging WandB
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })

        print(f"Epoch {epoch:03d}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6e}")

        # Sauvegarder le meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.save_path, exist_ok=True)
            ckpt_path = os.path.join(args.save_path, 'seq2seq_v2_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Meilleur modèle sauvegardé -> {ckpt_path}")

    # ----------- Test final -----------
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for EEG_emb, z0_flat in tqdm(test_loader, desc="Test" ):
            EEG_emb = EEG_emb.to(device)
            z0_flat = z0_flat.to(device)
            z_pred_flat = model(EEG_emb)
            if args.normalize:
                z_pred_5d = z_pred_flat.view(-1, 6, 4, 36, 64) * std_z + mean_z
                z_true_5d = z0_flat.view(-1, 6, 4, 36, 64) * std_z + mean_z
            else:
                z_pred_5d = z_pred_flat.view(-1, 6, 4, 36, 64)
                z_true_5d = z0_flat.view(-1, 6, 4, 36, 64)

            B, T, C, H, W = z_pred_5d.shape
            z_pred_frames = z_pred_5d.view(B * T, C, H, W)
            z_true_frames = z_true_5d.view(B * T, C, H, W)
            img_pred = model.vae.decode(z_pred_frames)
            img_true = model.vae.decode(z_true_frames)

            loss = criterion_mse(z_pred_flat, z0_flat)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test final  ► test_loss={avg_test_loss:.4f}")

    if args.use_wandb:
        wandb.log({'test_loss': avg_test_loss})
        wandb.finish()

if __name__ == '__main__':
    train_seq2seq(parse_args())
