import os, time, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from sklearn.metrics import confusion_matrix

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from Gaspard.GLMNet.modules.utils_glmnet import (
    GLMNet,
    standard_scale_features,
    compute_raw_stats,
    normalize_raw,
)


# -------- W&B -------------------------------------------------------------
PROJECT_NAME = "eeg2video-GLMNetv2"  # <‑‑ change if you need another project

# ------------------------------ constants ---------------------------------
OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels
RAW_T = 200 # time points in raw EEG, 1 second at 200Hz


# ------------------------------ utils -------------------------------------
def parse_args():
     #"/Documents/School/Centrale Med/2A/SSE/EEG2Video"
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",  default = "./data/Preprocessing/Segmented_Rawf_200Hz_2s", help="directory with .npy files") 
    p.add_argument("--feat_dir", default="./data/Preprocessing/DE_1per1s/", help="directory with .npy files")
    p.add_argument("--label_dir", default="./data/meta_info", help="Label file")
    p.add_argument("--category", default="label",choices=['color', 'face_appearance', 'human_appearance','label_cluster','label','obj_number','optical_flow_score'], help="Label file")
    p.add_argument("--save_dir", default="./Gaspard/checkpoints/glmnet")
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
    if labels.shape[1] == 40:
        labels = labels[..., None, None]            # (7,40,1,1)
        labels = np.repeat(labels, 5, axis=2)       # (7,40,5,1)
    else:
        assert labels.shape[1] == 200, "Labels must be (7,40,200) or (7,40)"
        labels = labels.reshape(-1,40,5)[..., None]              # (7,40,5,1)

    labels = np.repeat(labels, 2, axis=3)       # (7,40,5,2)
    assert labels.shape == (7,40,5,2), "Label shape must be (7,40,5,2) after expansion"
    return labels 

def format_labels(labels: np.ndarray, category:str) -> np.ndarray:
    match category:
        case "color" | "face_appearance" | "human_appearance" | "label_cluster":
            return labels.astype(np.int64)
        case "label" | "obj_number" :
            labels = labels-1
            return labels.astype(np.int64)
        case "optical_flow_score":
            threshold = 1.799
            return (labels > threshold).astype(np.int64)
        case _:
            raise ValueError(f"Unknown category: {category}. Must be one of: color, face_appearance, human_appearance, object, label_cluster, label, obj_number, optical_flow_score.")
# ------------------------------ main -------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Sélection d’un seul sujet
    filename = "sub3.npy"  # ou args.subj_name
    subj_name = filename.replace(".npy", "")

    raw2s = np.load(os.path.join(args.raw_dir, filename))  # (7,40,5,62,400)
    feat = np.load(os.path.join(args.feat_dir, filename))  # (7,40,5,62,5)
    labels_raw = np.load(f'{args.label_dir}/All_video_{args.category}.npy')                       # (7,40)
    unique_labels, counts_labels = np.unique(labels_raw, return_counts=True)
    label_distribution = {int(u): int(c) for u, c in zip(unique_labels, counts_labels)}
    print("Label distribution:", label_distribution)

    # Display label distribution for each block
    for b in range(labels_raw.shape[0]):
        u_b, c_b = np.unique(labels_raw[b], return_counts=True)
        dist_b = {int(u): int(c) for u, c in zip(u_b, c_b)}
        print(f"Block {b} distribution: {dist_b}")
    labels = format_labels(reshape_labels(labels_raw), args.category)                        # (7,40,5,2)
    print(labels.shape)
    num_unique_labels = len(np.unique(labels))
    print("Number of categories:", num_unique_labels)
    
    raw1s = split_raw_2s_to_1s(raw2s)                      # (7,40,5,2,62,200)

    acc_folds = []


    for test_block in range(7):
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

        # Compute normalization statistics on training raw EEG
        raw_mean, raw_std = compute_raw_stats(X_train)

        # Apply normalization to all splits
        X_train = normalize_raw(X_train, raw_mean, raw_std)
        X_val = normalize_raw(X_val, raw_mean, raw_std)
        X_test = normalize_raw(X_test, raw_mean, raw_std)

        # Fit scaler on training features and apply to all splits
        F_train_scaled, scaler = standard_scale_features(F_train, return_scaler=True)
        F_val_scaled = standard_scale_features(F_val, scaler=scaler)
        F_test_scaled = standard_scale_features(F_test, scaler=scaler)

        # Save scaler for this fold
        scaler_path = os.path.join(args.save_dir, f"{subj_name}_fold{test_block}_{args.category}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Save raw EEG normalization parameters
        norm_path = os.path.join(
            args.save_dir,
            f"{subj_name}_fold{test_block}_{args.category}_rawnorm.npz",
        )
        np.savez(norm_path, mean=raw_mean, std=raw_std)


        # Conversion en tenseurs
        ds_train = TensorDataset(torch.tensor(X_train,dtype=torch.float32).unsqueeze(1), torch.tensor(F_train_scaled,dtype=torch.float32), torch.tensor(y_train))
        ds_val   = TensorDataset(torch.tensor(X_val,dtype=torch.float32).unsqueeze(1), torch.tensor(F_val_scaled,dtype=torch.float32), torch.tensor(y_val))
        ds_test  = TensorDataset(torch.tensor(X_test,dtype=torch.float32).unsqueeze(1), torch.tensor(F_test_scaled,dtype=torch.float32), torch.tensor(y_test))

        dl_train = DataLoader(ds_train, args.bs, shuffle=True)
        dl_val   = DataLoader(ds_val, args.bs)
        dl_test  = DataLoader(ds_test, args.bs)

        # Initialisation modèle
        model = GLMNet(OCCIPITAL_IDX, out_dim=num_unique_labels).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.8, patience=10, verbose=True)
        criterion = nn.CrossEntropyLoss()

        # W&B
        if args.use_wandb:
            wandb.init(project=PROJECT_NAME, name=f"{subj_name}_fold{test_block}_{args.category}", config=vars(args))
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
            model.eval(); vl = va = 0
            with torch.no_grad():
                for xb, xf, yb in dl_val:
                    xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
                    pred = model(xb, xf)
                    vloss = criterion(pred, yb)
                    vl += vloss.item() * len(yb)
                    va += (pred.argmax(1) == yb).sum().item()
            val_acc = va / len(ds_val)
            val_loss = vl / len(ds_val)
            scheduler.step(val_acc)

            # Print training progress
            print(
                f"Fold {test_block} | Epoch {ep:02d} - "
                f"train_acc: {train_acc:.3f}, train_loss: {tl/len(ds_train):.3f}, "
                f"val_acc: {val_acc:.3f}, val_loss: {val_loss:.3f}"
            )

            if val_acc > best_val:
                best_val = val_acc
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{args.save_dir}/{subj_name}_fold{test_block}_{args.category}_best.pt")

            if args.use_wandb:
                wandb.log({"epoch": ep, "train/acc": train_acc, "val/acc": val_acc,
                           "train/loss": tl / len(ds_train), "val/loss": val_loss})

        # Test
        
        model.load_state_dict(torch.load(f"{args.save_dir}/{subj_name}_fold{test_block}_{args.category}_best.pt"))
        model.eval(); test_acc = 0
        preds, labels_test = [], []
        with torch.no_grad():
            for xb, xf, yb in dl_test:
                xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
                out = model(xb, xf)
                pred_labels = out.argmax(1)
                test_acc += (pred_labels == yb).sum().item()
                preds.append(pred_labels.cpu())
                labels_test.append(yb.cpu())
        preds = torch.cat(preds).numpy()
        labels_test = torch.cat(labels_test).numpy()
        cm = confusion_matrix(labels_test, preds)
        test_acc /= len(ds_test)
        acc_folds.append(test_acc)
        print(f"[{subj_name}-Fold{test_block}] BEST test_acc={test_acc:.3f}")
        print("Confusion matrix:\n", cm)
        if args.use_wandb:
            class_names = [str(c) for c in np.unique(labels)]
            cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=labels_test, preds=preds, class_names=class_names)
            wandb.log({"test/acc": test_acc, "test/confusion_matrix": cm_plot})
            wandb.finish()

    print(f"Subject {subj_name}: mean±std test acc = {np.mean(acc_folds):.3f} ± {np.std(acc_folds):.3f}")

if __name__ == "__main__":
    main()

