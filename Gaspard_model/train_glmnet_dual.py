#from __future__ import annotations
import os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.models_paper import shallownet, mlpnet  # <- adjust if package path differs
from tqdm import tqdm
import  wandb
from sklearn.preprocessing import StandardScaler

# -------- W&B -------------------------------------------------------------
PROJECT_NAME = "eeg2video-GLMNetv2"  # <‑‑ change if you need another project

# -------- Model parts (reuse existing codebase) ---------------------------


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
    root = os.environ.get("HOME", os.environ.get("USERPROFILE")) + "/Documents/School/Centrale Med/2A/SSE/EEG2Video"
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",  default = f"{root}/data/Segmented_Rawf_200Hz_2s", help="directory with .npy files") 
    p.add_argument("--feat_dir", default=f"{root}/data/DE_1per1s/", help="directory with .npy files")
    p.add_argument("--label_dir", default=f"{root}/data/meta_info/All_video_label.npy", help="Label file")
    p.add_argument("--save_dir", default=f"{root}/Gaspard_model/checkpoints/cv_glmnetv2/")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--bs",       type=int, default=128)
    p.add_argument("--lr",       type=float, default=1e-4)
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
    return labels -1 # 0-39 labels



def standard_scale_features(X_train, X_val, X_test):
    # X: (N, features...)
    # Flatten to 2D if needed
    orig_shape = X_train.shape[1:]
    X_train_2d = X_train.reshape(len(X_train), -1)
    X_val_2d = X_val.reshape(len(X_val), -1)
    X_test_2d = X_test.reshape(len(X_test), -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back
    X_train_scaled = X_train_scaled.reshape((len(X_train),) + orig_shape)
    X_val_scaled = X_val_scaled.reshape((len(X_val),) + orig_shape)
    X_test_scaled = X_test_scaled.reshape((len(X_test),) + orig_shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled


# ------------------------------ main -------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Sélection d’un seul sujet
    filename = "sub3.npy"  # ou args.subj_name
    subj_name = filename.replace(".npy", "")

    raw2s = np.load(os.path.join(args.raw_dir, filename))  # (7,40,5,62,400)
    feat = np.load(os.path.join(args.feat_dir, filename))  # (7,40,5,62,5)
    labels = np.load(args.label_dir)                       # (7,40)
    labels = reshape_labels(labels)                        # (7,40,5,2)

    raw1s = split_raw_2s_to_1s(raw2s)                      # (7,40,5,2,62,200)

    acc_folds = []


    for test_block in range(7):
        print(f"\n🔁 Fold {test_block}")
        val_block = (test_block - 1) % 7
        train_blocks = [i for i in range(7) if i not in [test_block, val_block]]

        def get_data(block_ids):
            x_raw = raw1s[block_ids].reshape(-1, 62, 200)
            x_feat = feat[block_ids].reshape(-1, 62, 5)
            y = labels[block_ids].reshape(-1)
            return x_raw, x_feat, y

        X_train, F_train, y_train = get_data(train_blocks)
        X_val, F_val, y_val = get_data([val_block])
        X_test, F_test, y_test = get_data([test_block])
        F_train_scaled, F_val_scaled, F_test_scaled = standard_scale_features(F_train, F_val, F_test)


        # Conversion en tenseurs
        ds_train = TensorDataset(torch.tensor(X_train,dtype=torch.float32).unsqueeze(1), torch.tensor(F_train_scaled,dtype=torch.float32), torch.tensor(y_train))
        ds_val   = TensorDataset(torch.tensor(X_val,dtype=torch.float32).unsqueeze(1), torch.tensor(F_val_scaled,dtype=torch.float32), torch.tensor(y_val))
        ds_test  = TensorDataset(torch.tensor(X_test,dtype=torch.float32).unsqueeze(1), torch.tensor(F_test_scaled,dtype=torch.float32), torch.tensor(y_test))

        dl_train = DataLoader(ds_train, args.bs, shuffle=True)
        dl_val   = DataLoader(ds_val, args.bs)
        dl_test  = DataLoader(ds_test, args.bs)

        # Initialisation modèle
        model = GLMNet(out_dim=40).to(device)
        opt = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # W&B
        if args.use_wandb:
            wandb.init(project=PROJECT_NAME, name=f"{subj_name}_fold{test_block}", config=vars(args))
            wandb.watch(model, log="all")

        # Entraînement
        best_val = 0.0
        for ep in tqdm(range(1, args.epochs + 1), desc=f"Fold {test_block}"):

            model.train(); tl = ta = 0
            for xb, xf, yb in dl_train:
                xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
                opt.zero_grad(); pred = model(xb, xf)
                loss = criterion(pred, yb); loss.backward(); opt.step()
                tl += loss.item() * len(yb); ta += (pred.argmax(1) == yb).sum().item()
            train_acc = ta / len(ds_train)

            # Validation
            model.eval(); va = 0
            with torch.no_grad():
                for xb, xf, yb in dl_val:
                    xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
                    pred = model(xb, xf); va += (pred.argmax(1) == yb).sum().item()
            val_acc = va / len(ds_val)

            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), f"{args.save_dir}/{subj_name}_fold{test_block}_best.pt")

            #if ep % 5 == 0:
            #   print(f"[{subj_name}-Fold{test_block}] Ep{ep:03d} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

            if args.use_wandb:
                wandb.log({"epoch": ep, "train/acc": train_acc, "val/acc": val_acc})

        # Test
        model.load_state_dict(torch.load(f"{args.save_dir}/{subj_name}_fold{test_block}_best.pt"))
        model.eval(); test_acc = 0
        with torch.no_grad():
            for xb, xf, yb in dl_test:
                xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
                pred = model(xb, xf); test_acc += (pred.argmax(1) == yb).sum().item()
        test_acc /= len(ds_test)
        acc_folds.append(test_acc)
        print(f"[{subj_name}-Fold{test_block}] BEST test_acc={test_acc:.3f}")
        if args.use_wandb: wandb.log({"test/acc": test_acc}); wandb.finish()

    print(f"Subject {subj_name}: mean±std test acc = {np.mean(acc_folds):.3f} ± {np.std(acc_folds):.3f}")

if __name__ == "__main__":
    main()

