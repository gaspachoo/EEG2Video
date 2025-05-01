from __future__ import annotations

import os, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import  wandb
# -------- W&B -------------------------------------------------------------
PROJECT_NAME = "eeg2video-GLMNetv2"  # <‑‑ change if you need another project

# -------- Model parts (reuse existing codebase) ---------------------------
from models.models import shallownet, mlpnet  # <- adjust if package path differs

OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels
RAW_T = 200 # time points in raw EEG, 1 second at 200Hz

# ------------------------------ model -------------------------------------
class GLMNet(nn.Module):
    """ShallowNet (raw) + MLP (freq) → concat → FC"""
    def __init__(self, out_dim: int = 40, emb_dim: int = 128):
        super().__init__()
        self.raw_global  = shallownet(emb_dim, 62, 200)  # (B,1,62,200)
        self.freq_local  = mlpnet(emb_dim, len(OCCIPITAL_IDX) * 5)  # (B,12,5)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim), nn.GELU(), nn.Linear(emb_dim, out_dim)
        )

    def forward(self, x_raw, x_feat):
        g_raw  = self.raw_global(x_raw)  # (B,emb)
        l_freq = self.freq_local(x_feat[:, OCCIPITAL_IDX, :])
        return self.fc(torch.cat([g_raw, l_freq], dim=1))

# ------------------------------ utils -------------------------------------
def parse_args():
    root = os.environ["HOME"] + "/EEG2Video"
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",  default = f"{root}/data/EEG/", help="directory with *_raw.npy files") 
    p.add_argument("--feat_dir", default=f"{root}/data/DE_500ms_sw/", help="directory with *_de.npy files")
    p.add_argument("--label_dir", default=f"{root}/data/meta_info/All_video_label.npy", help="Label file")
    p.add_argument("--save_dir", default=f"{root}/Gaspard_model/checkpoints/cv_glmnetv2/")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--bs",       type=int, default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()


def split_raw_2s_to_1s(raw2s: np.ndarray) -> np.ndarray:
    """Convert (7,40,5,62,400) 2‑second raw → (7,40,5,2,62,200).
    The new dim axis=3 indexes the first / second second.
    """
    assert raw2s.shape[-1] == 2 * RAW_T, "last dim must be 400 (=2 s)"
    first  = raw2s[..., :RAW_T]          # (7,40,5,62,200)
    second = raw2s[..., RAW_T: 2*RAW_T]  # (7,40,5,62,200)
    return np.stack([first, second], axis=3)  # (7,40,5,2,62,200)

def reshape_labels(labels: np.ndarray) -> np.ndarray:
    """Convert labels from (7,40) to (7,40,5,2)"""
    labels = labels[..., None, None]            # (7,40,1,1)
    labels = np.repeat(labels, 5, axis=2)       # (7,40,5,1)
    labels = np.repeat(labels, 2, axis=3)       # (7,40,5,2)
    assert labels.shape == (7,40,5,2), "Label shape must be (7,40,5,2) after expansion"
    return labels

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    raw_files = sorted([f for f in os.listdir(args.raw_dir) if f.endswith(".npy")])
    if not raw_files:
        raise FileNotFoundError("No .npy files found in " + args.raw_dir)

    os.makedirs(args.save_dir, exist_ok=True)

    acc_subjects = []
    for raw_file in tqdm(raw_files,desc="Raw files"):
        subj_name = raw_file.replace(".npy", "")
        raw_npy = np.load(os.path.join(args.raw_dir, subj_name))
        raw_npy = split_raw_2s_to_1s(raw_npy)  # (7,40,5,2,62,400) → (7,40,5,2,62,200)
        feat_npy = np.load(os.path.join(args.feat_dir, subj_name))
        label_npy = np.load(args.label_dir)
        label_npy = reshape_labels(label_npy)  # (7,40,5,2)

        # ---------- reshape with N=7*40*5*2 ----------
        raw_npy = raw_npy.reshape(-1, 62, 200)  # N×62×200
        feat_np = feat_npy.reshape(-1, 62, 5)   # N×62×5  (already split)
        labels = label_npy.reshape(-1)        # N

        assert raw_npy.shape[0] == feat_np.shape[0] == labels.shape[0]

        # tensors
        X_raw = torch.tensor(raw_npy, dtype=torch.float32)  # (M,62,200)
        X_feat = torch.tensor(feat_np, dtype=torch.float32) # (M,62,5)
        y = torch.tensor(labels, dtype=torch.long)

        # 80/10/10 split
        M = len(y); print(M)
        idx = np.arange(M); np.random.shuffle(idx)
        tr, va = int(0.8 * M), int(0.9 * M)
        ds_train = TensorDataset(X_raw[idx[:tr]], X_feat[idx[:tr]], y[idx[:tr]])
        ds_val = TensorDataset(X_raw[idx[tr:va]], X_feat[idx[tr:va]], y[idx[tr:va]])
        ds_test = TensorDataset(X_raw[idx[va:]], X_feat[idx[va:]], y[idx[va:]])

        dl_train = DataLoader(ds_train, args.bs, shuffle=True)
        dl_val = DataLoader(ds_val, args.bs)
        dl_test = DataLoader(ds_test, args.bs)

        criterion = nn.CrossEntropyLoss()

        net = GLMNet(out_dim=40).to(device)
        opt = optim.AdamW(net.parameters(), lr=args.lr)

        if args.use_wandb:
            wandb.init(project=PROJECT_NAME, name=subj_name, config=vars(args))
            wandb.watch(net, log="gradients", log_freq=100)

        best_val = 0.0
        for ep in range(1, args.epochs + 1):
            # --- train ---
            net.train(); tl = ta = n = 0
            for xb_raw, xb_feat, yb in dl_train:
                xb_raw, xb_feat, yb = xb_raw.to(device), xb_feat.to(device), yb.to(device)
                opt.zero_grad(); pred = net(xb_raw, xb_feat)
                loss = criterion(pred, yb); loss.backward(); opt.step()
                tl += loss.item() * len(yb); ta += (pred.argmax(1) == yb).sum().item(); n += len(yb)
            train_loss, train_acc = tl / n, ta / n

            # --- val ---
            net.eval(); vl = va = nv = 0
            with torch.no_grad():
                for xb_raw, xb_feat, yb in dl_val:
                    xb_raw, xb_feat, yb = xb_raw.to(device), xb_feat.to(device), yb.to(device)
                    pred = net(xb_raw, xb_feat)
                    vl += criterion(pred, yb).item() * len(yb); va += (pred.argmax(1) == yb).sum().item(); nv += len(yb)
            val_loss, val_acc = vl / nv, va / nv

            if args.use_wandb and wandb is not None:
                wandb.log({"epoch": ep,
                           "train/loss": train_loss, "train/acc": train_acc,
                           "val/loss": val_loss,     "val/acc": val_acc})

            if val_acc > best_val:
                best_val = val_acc
                torch.save(net.state_dict(), os.path.join(args.save_dir, f"{subj_name}_best.pt"))

            if ep % 5 == 0:
                print(f"[{subj_name}] Ep{ep:03d} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        # ---- test ----
        net.load_state_dict(torch.load(os.path.join(args.save_dir, f"{subj_name}_best.pt")))
        net.eval(); ta = ntest = 0
        with torch.no_grad():
            for xb_raw, xb_feat, yb in dl_test:
                xb_raw, xb_feat, yb = xb_raw.to(device), xb_feat.to(device), yb.to(device)
                pred = net(xb_raw, xb_feat)
                ta += (pred.argmax(1) == yb).sum().item(); ntest += len(yb)
        test_acc = ta / ntest
        acc_subjects.append(test_acc)
        print(f"[{subj_name}] BEST test_acc={test_acc:.3f}")
        if args.use_wandb and wandb is not None:
            wandb.log{"test/acc": test_acc}; wandb.finish()

    print("\nOverall mean±std test acc:", np.mean(acc_subjects), "±", np.std(acc_subjects))

if __name__ == "__main__":
    main()
