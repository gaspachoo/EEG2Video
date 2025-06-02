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

# ------------------ Argument Parsing ------------------

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--sub_emb',       type=str,
                        default="./data/GLMNet/EEG_embeddings_sw/sub3.npy",
                        help='Chemin vers le fichier .npy contenant les embeddings EEG de forme (7,40,5,7,512)')
    parser.add_argument('--video_dir',     type=str,
                        default="./data/Seq2Seq/Video_latents",
                        help='Répertoire contenant les fichiers block0.npy ... block6.npy de latents vidéo')
    parser.add_argument('--save_path',     type=str,
                        default="./Gaspard/checkpoints/seq2seq",
                        help='Dossier où sauvegarder le checkpoint final')
    parser.add_argument('--use_wandb',     action='store_true',
                        help='Activer la journalisation sur Weights & Biases')

    # Hyperparameters
    parser.add_argument('--epochs',        type=int,   default=300,
                        help='Nombre d\u2019époques')
    parser.add_argument('--batch_size',    type=int,   default=64,
                        help='Taille de batch')
    parser.add_argument('--lr',            type=float, default=5e-4,
                        help='Taux d\u2019apprentissage initial')
    parser.add_argument('--min_lr',        type=float, default=1e-6,
                        help='Taux d\u2019apprentissage minimal pour le scheduler')
    parser.add_argument('--scheduler',     type=str, choices=['cosine', 'step', 'plateau'], default="step",
                        help='Taux d\u2019apprentissage initial')
    parser.add_argument('--warmup_epochs', type=int,   default=10,
                        help='Nombre d\u2019époques pour le warm-up linéaire du learning rate')

    # Ratios train/val/test (non optionnel)
    parser.add_argument('--train_ratio',   type=float, default=0.8,
                        help='Proportion des données pour l\u2019ensemble d\u2019entraînement')
    parser.add_argument('--val_ratio',     type=float, default=0.1,
                        help='Proportion des données pour l\u2019ensemble de validation')
    # test_ratio sera calculé comme 1 - train_ratio - val_ratio

    # Normalisation (optionnel)
    parser.add_argument('--normalize',     action='store_true',
                        help='Activer la normalisation des latents vidéo (calculer mean/std global)')
    parser.add_argument('--noise_gamma', type=float, default=0.0, help='Ajouter du bruit à l entrainement')

    return parser.parse_args()

# ------------------ Scheduler builder ----------------------
def build_lr_scheduler(args,optimizer):
        sched = args.scheduler
        min_lr = args.min_lr
        if sched == 'none':
            return None
        if sched == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 200, eta_min = args.min_lr)
        if sched == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,gamma=0.5)
        if sched == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=min_lr)
        raise ValueError(sched)
    
# ------------------ Préparation des données ------------------

def load_and_prepare_data(args):
    # Charger embeddings EEG, supporte shape (7,40,5,7,512) ou (9800,512)
    sub_emb = np.load(args.sub_emb)
    if sub_emb.ndim == 5:
        num_blocks, num_concepts, num_repeats, seq_len, d_eeg = sub_emb.shape
        all_eeg = sub_emb.reshape(-1, seq_len, d_eeg)  # (1400,7,512)
    elif sub_emb.ndim == 2:
        total_timepoints, d_eeg = sub_emb.shape
        if total_timepoints % 7 != 0:
            raise ValueError(f"Le nombre de lignes {total_timepoints} n'est pas divisible par 7 pour reshaper en (N,7,512)")
        all_eeg = sub_emb.reshape(-1, 7, d_eeg)       # (1400,7,512)
        seq_len = 7
    else:
        raise ValueError(f"sub_emb.ndim inattendu : {sub_emb.ndim}, attendu 2 ou 5.")

    # Charger et concaténer les latents vidéo de tous les blocs
    z0_list = []
    num_blocks = 7
    for b in range(num_blocks):
        path_zb = os.path.join(args.video_dir, f"block{b}.npy")
        if not os.path.isfile(path_zb):
            raise FileNotFoundError(f"Fichier introuvable : {path_zb}")
        z0b = np.load(path_zb)  # shape : (200,6,4,36,64)
        z0_list.append(z0b)
    z0_all = np.concatenate(z0_list, axis=0)    # (1400,6,4,36,64)

    # Calcul de mean/std et normalisation si demandé
    if args.normalize:
        mean_z = z0_all.mean(axis=(0,2,3,4), keepdims=True)
        std_z  = z0_all.std(axis=(0,2,3,4), keepdims=True) + 1e-6
        z0_all_norm = (z0_all - mean_z) / std_z
        z0_flat = z0_all_norm.reshape(z0_all_norm.shape[0], 6, -1)  # (1400,6,9216)
    else:
        mean_z, std_z = None, None
        z0_flat = z0_all.reshape(z0_all.shape[0], 6, -1)             # (1400,6,9216)

    # Construction du TensorDataset
    eeg_tensor = torch.tensor(all_eeg, dtype=torch.float32)      # (1400,7,512)
    vid_tensor = torch.tensor(z0_flat, dtype=torch.float32)       # (1400,6,9216)
    dataset = TensorDataset(eeg_tensor, vid_tensor)

    # Split train/val/test
    total = len(dataset)
    train_size = int(args.train_ratio * total)
    val_size   = int(args.val_ratio   * total)
    test_size  = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)

    return train_loader, val_loader, test_loader, mean_z, std_z

# ------------------ Entraînement Seq2Seq ------------------

def train_seq2seq(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Préparation des données
    train_loader, val_loader, test_loader, mean_z, std_z = load_and_prepare_data(args)

    # Initialiser le modèle
    model = Seq2SeqTransformer().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='min', factor=0.5, patience=10, min_lr=args.min_lr
    #)
    scheduler = build_lr_scheduler(args, optimizer)

    criterion = nn.MSELoss()

    if args.use_wandb:
        wandb.init(project="eeg2video-seq2seq-v2", config=vars(args))

    best_val = float('inf')
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Phase entraînement
        model.train()
        train_loss = 0.0
        for src, tgt in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            src = src.to(device)   # (B,7,512)
            tgt = tgt.to(device)   # (B,6,9216)

            # Injecter un bruit léger (σ = 0.01 ou 0.02)
            noise = torch.randn_like(tgt) * args.noise_gamma
            tgt_noisy = tgt + noise
            
            # Warm-up linéaire du learning rate
            if global_step < warmup_steps:
                lr_scale = float(global_step) / float(max(1, warmup_steps))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * args.lr
                    
            optimizer.zero_grad()
            out = model(src, tgt_noisy)  # (B,6,9216)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            
            # Calcul de moyenne et écart-type de (out - tgt)
            diff = out - tgt
            diff_mean = diff.mean().item()
            diff_std = diff.std().item()
            
            train_loss += loss.item()
            global_step += 1

        avg_train = train_loss / len(train_loader)

        # Phase validation
        model.eval()
        val_loss = 0.0
        val_diff_mean_sum = 0.0
        val_diff_std_sum = 0.0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                src = src.to(device)
                tgt = tgt.to(device)
                out = model(src, tgt)
                val_loss += criterion(out, tgt).item()
                
                diff = out - tgt
                val_diff_mean_sum += diff.mean().item()
                val_diff_std_sum += diff.std().item()
                
        avg_val = val_loss / len(val_loader)
        avg_diff_mean_val = val_diff_mean_sum / len(val_loader)
        avg_diff_std_val = val_diff_std_sum / len(val_loader)
        
        # LR scheduler step
        if args.scheduler != "none":
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if args.use_wandb:
            wandb.log({ 'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val,
                        'val_diff_mean': avg_diff_mean_val,
                        'val_diff_std': avg_diff_std_val,
                        'lr': optimizer.param_groups[0]['lr'] })

        print(f"Epoch {epoch:03d}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  lr={optimizer.param_groups[0]['lr']:.6e}")

        # Sauvegarde du meilleur modèle
        if avg_val < best_val:
            best_val = avg_val
            os.makedirs(args.save_path, exist_ok=True)
            ckpt = os.path.join(args.save_path, f'seq2seq_v3_{args.scheduler}_{args.noise_gamma}.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"Meilleur modèle sauvegardé -> {ckpt}")

    # Test final
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Test"):
            src = src.to(device)
            tgt = tgt.to(device)
            out = model(src, tgt)
            test_loss += criterion(out, tgt).item()
    avg_test = test_loss / len(test_loader)
    print(f"Test final  ► test_loss={avg_test:.4f}")

    if args.use_wandb:
        wandb.log({'test_loss': avg_test})
        wandb.finish()

if __name__ == '__main__':
    train_seq2seq(parse_args())
