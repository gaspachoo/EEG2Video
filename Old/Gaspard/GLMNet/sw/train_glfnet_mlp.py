import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import pickle
from sklearn.metrics import confusion_matrix

project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Old.EEG2Video.GLMNet.utils_glfnet_mlp import GLFNetMLP
from EEG2Video.GLMNet.modules.utils_glmnet import standard_scale_features

PROJECT_NAME = "eeg2video-GLFNetMLP"
OCCIPITAL_IDX = list(range(50, 62))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default="./data/Preprocessing/DE_1per1s/", help="directory with .npy files")
    p.add_argument("--label_dir", default="./data/meta_info", help="Label file")
    p.add_argument(
        "--category",
        default="label",
        choices=[
            "color",
            "face_appearance",
            "human_appearance",
            "label_cluster",
            "label",
            "obj_number",
            "optical_flow_score",
        ],
        help="Label type",
    )
    p.add_argument("--save_dir", default="./EEG2Video/checkpoints/glfnet_mlp")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6,
                   help="Minimum learning rate for the scheduler")
    p.add_argument(
        "--scheduler",
        type=str,
        choices=["steplr", "reducelronplateau", "cosine"],
        default="reducelronplateau",
        help="Type of learning rate scheduler",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument(
        "--window",
        choices=["1s", "500ms"],
        default="1s",
        help="length of EEG windows used for training",
    )
    return p.parse_args()


def reshape_labels(labels: np.ndarray, n_win: int) -> np.ndarray:
    if labels.shape[1] == 40:
        labels = labels[..., None, None]
        labels = np.repeat(labels, 5, axis=2)
    else:
        assert labels.shape[1] == 200, "Labels must be (7,40,200) or (7,40)"
        labels = labels.reshape(-1, 40, 5)[..., None]
    labels = np.repeat(labels, n_win, axis=3)
    assert labels.shape[:3] == (7, 40, 5) and labels.shape[3] == n_win
    return labels


def format_labels(labels: np.ndarray, category: str) -> np.ndarray:
    match category:
        case "color" | "face_appearance" | "human_appearance" | "label_cluster":
            return labels.astype(np.int64)
        case "label" | "obj_number":
            labels = labels - 1
            return labels.astype(np.int64)
        case "optical_flow_score":
            threshold = 1.799
            return (labels > threshold).astype(np.int64)
        case _:
            raise ValueError(
                "Unknown category: {category}. Must be one of: color, face_appearance, human_appearance, label_cluster, label, obj_number, optical_flow_score."
            )


def main():
    args = parse_args()
    if args.window == "500ms" and args.feat_dir == "./data/Preprocessing/DE_1per1s/":
        args.feat_dir = "./data/Preprocessing/DE_500ms_sw"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    filename = "sub3.npy"
    subj_name = filename.replace(".npy", "")

    feat = np.load(os.path.join(args.feat_dir, filename))
    if args.window == "1s":
        if feat.ndim == 5:
            feat = np.repeat(feat[:, :, :, None, :, :], 2, axis=3)
        n_win = 2
        assert feat.ndim == 6 and feat.shape[:4] == (7, 40, 5, n_win), "Unexpected feature shape"
    else:
        n_win = feat.shape[3]
        assert feat.ndim == 6 and feat.shape[:4] == (7, 40, 5, n_win), "Unexpected feature shape"

    labels_raw = np.load(f"{args.label_dir}/All_video_{args.category}.npy")
    unique_labels, counts_labels = np.unique(labels_raw, return_counts=True)
    label_distribution = {int(u): int(c) for u, c in zip(unique_labels, counts_labels)}
    print("Label distribution:", label_distribution)

    labels = format_labels(reshape_labels(labels_raw, n_win), args.category)
    num_classes = len(np.unique(labels))

    acc_folds = []

    # Flatten features and randomly split data
    F_all = feat.reshape(-1, 62, 5)
    y_all = labels.reshape(-1)

    n = len(y_all)
    idx = np.random.permutation(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    F_train, y_train = F_all[train_idx], y_all[train_idx]
    F_val, y_val = F_all[val_idx], y_all[val_idx]
    F_test, y_test = F_all[test_idx], y_all[test_idx]

    F_train_scaled, scaler = standard_scale_features(F_train, return_scaler=True)
    F_val_scaled = standard_scale_features(F_val, scaler=scaler)
    F_test_scaled = standard_scale_features(F_test, scaler=scaler)

    scaler_path = os.path.join(args.save_dir, f"{subj_name}_{args.category}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    ds_train = TensorDataset(torch.tensor(F_train_scaled, dtype=torch.float32),
                             torch.tensor(y_train))
    ds_val = TensorDataset(torch.tensor(F_val_scaled, dtype=torch.float32),
                           torch.tensor(y_val))
    ds_test = TensorDataset(torch.tensor(F_test_scaled, dtype=torch.float32),
                            torch.tensor(y_test))

    dl_train = DataLoader(ds_train, args.bs, shuffle=True)
    dl_val = DataLoader(ds_val, args.bs)
    dl_test = DataLoader(ds_test, args.bs)

    model = GLFNetMLP(OCCIPITAL_IDX, out_dim=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            opt, mode="max", factor=0.8, patience=10, verbose=True, min_lr=args.min_lr
        )
    elif args.scheduler == "steplr":
        scheduler = StepLR(opt, step_size=10, gamma=0.5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs // 2, eta_min=args.min_lr)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        wandb.init(project=PROJECT_NAME, name=f"{subj_name}_{args.category}", config=vars(args))
        wandb.watch(model, log="all")

    best_val = 0.0
    for ep in tqdm(range(1, args.epochs + 1)):
        model.train(); tl = ta = 0
        for xf, yb in dl_train:
            xf, yb = xf.to(device), yb.to(device)
            opt.zero_grad(); pred = model(xf)
            loss = criterion(pred, yb); loss.backward(); opt.step()
            tl += loss.item() * len(yb); ta += (pred.argmax(1) == yb).sum().item()
        train_acc = ta / len(ds_train)

        model.eval(); vl = va = 0
        with torch.no_grad():
            for xf, yb in dl_val:
                xf, yb = xf.to(device), yb.to(device)
                pred = model(xf)
                vloss = criterion(pred, yb)
                vl += vloss.item() * len(yb)
                va += (pred.argmax(1) == yb).sum().item()
        val_acc = va / len(ds_val)
        val_loss = vl / len(ds_val)
        if scheduler is not None:
            if args.scheduler == "reducelronplateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
            for pg in opt.param_groups:
                if pg["lr"] < args.min_lr:
                    pg["lr"] = args.min_lr
        current_lr = opt.param_groups[0]["lr"]

        print(f"Epoch {ep:02d} - train_acc: {train_acc:.3f}, train_loss: {tl/len(ds_train):.3f}, "
              f"val_acc: {val_acc:.3f}, val_loss: {val_loss:.3f}, lr: {current_lr:.2e}")

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.save_dir}/{subj_name}_{args.category}_best.pt")

        if args.use_wandb:
            wandb.log({"epoch": ep, "train/acc": train_acc, "val/acc": val_acc,
                       "train/loss": tl / len(ds_train), "val/loss": val_loss,
                       "lr": current_lr})

    model.load_state_dict(torch.load(f"{args.save_dir}/{subj_name}_{args.category}_best.pt"))
    model.eval(); test_acc = 0
    preds, labels_test = [], []
    with torch.no_grad():
        for xf, yb in dl_test:
            xf, yb = xf.to(device), yb.to(device)
            out = model(xf)
            pred_labels = out.argmax(1)
            test_acc += (pred_labels == yb).sum().item()
            preds.append(pred_labels.cpu())
            labels_test.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    labels_test = torch.cat(labels_test).numpy()
    cm = confusion_matrix(labels_test, preds)
    test_acc /= len(ds_test)
    print(f"Test accuracy = {test_acc:.3f}")
    print("Confusion matrix:\n", cm)
    if args.use_wandb:
        class_names = [str(c) for c in np.unique(labels)]
        cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=labels_test, preds=preds, class_names=class_names)
        wandb.log({"test/acc": test_acc, "test/confusion_matrix": cm_plot})
        wandb.finish()
if __name__ == "__main__":
    main()

