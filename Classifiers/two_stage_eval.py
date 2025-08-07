import argparse
import os
import re
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Classifiers.modules.models import glmnet, eegnet, deepnet, mlpnet
from Classifiers.modules.utils import (
    compute_raw_stats,
    normalize_raw,
    standard_scale_features,
    block_split,
)
from Classifiers.multi_inference import majority_vote, CLUSTER_RANGES

OCCIPITAL_IDX = list(range(50, 62))


def format_labels(labels: np.ndarray, category: str) -> np.ndarray:
    match category:
        case "color" | "face_appearance" | "human_appearance" | "label_cluster":
            return labels.astype(np.int64)
        case "color_binary":
            return (labels != 0).astype(np.int64)
        case "label" | "obj_number":
            return (labels - 1).astype(np.int64)
        case "optical_flow_score":
            threshold = 1.799
            return (labels > threshold).astype(np.int64)
        case _:
            raise ValueError(f"Unknown category: {category}")


def prepare_datasets(
    raw: np.ndarray,
    feat: Optional[np.ndarray],
    labels: np.ndarray,
    block_ids: np.ndarray,
    val_block: int,
    test_block: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    return_idx: bool = False,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Optional[np.ndarray],
]:
    n_win, C, T = raw.shape[1:]
    X_all = raw.reshape(-1, C, T)
    y_all = labels.reshape(-1)
    block_ids_win = np.repeat(block_ids, n_win)

    if feat is not None:
        F_all = feat.reshape(-1, C, feat.shape[-1])
    else:
        F_all = None

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_all))
        train_end = int(0.8 * len(idx))
        val_end = int(0.9 * len(idx))
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]
    else:
        train_mask = (block_ids_win != val_block) & (block_ids_win != test_block)
        val_mask = block_ids_win == val_block
        test_mask = block_ids_win == test_block
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

    def split(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return arr[train_idx], arr[val_idx], arr[test_idx]

    X_train, X_val, X_test = split(X_all)
    y_train, y_val, y_test = split(y_all)
    if F_all is not None:
        F_train, F_val, F_test = split(F_all)
        X_train = np.concatenate([X_train, F_train], axis=2)
        X_val = np.concatenate([X_val, F_val], axis=2)
        X_test = np.concatenate([X_test, F_test], axis=2)

    if return_idx:
        return (
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            test_idx,
        )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), None


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feat_dim: int,
    out_dim: int,
    device: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    C: int,
    T: int,
) -> nn.Module:
    if model_type == "glmnet":
        model = glmnet(OCCIPITAL_IDX, C=C, T=T, feat_dim=feat_dim, out_dim=out_dim).to(device)
    else:
        model_cls = eegnet if model_type == "eegnet" else deepnet
        model = model_cls(out_dim=out_dim, C=C, T=T).to(device)

    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.long),
    )
    ds_val = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.long),
    )
    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state: Dict[str, torch.Tensor] | None = None

    for _ in tqdm(range(epochs)):
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
        val_acc = val_correct / len(ds_val)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def build_category_data(
    raw: np.ndarray,
    feat: Optional[np.ndarray],
    label_dir: str,
    category: str,
    cluster: int | None,
    n_concepts: int,
    n_rep: int,
    n_blocks: int,
    val_block: int,
    test_block: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Optional[np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    np.ndarray,
    int,
]:
    label_path = os.path.join(label_dir, f"All_video_{category}.npy")
    if category == "color_binary" and not os.path.exists(label_path):
        label_path = os.path.join(label_dir, "All_video_color.npy")
    labels_raw = np.load(label_path)
    if labels_raw.shape[1] == n_concepts:
        labels_raw = np.repeat(labels_raw[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)

    mask_2d = (labels_raw != 0) if category == "color" else np.ones_like(labels_raw, bool)

    if cluster is not None:
        clusters = np.load(os.path.join(label_dir, "All_video_label_cluster.npy"))
        if clusters.shape[1] == n_concepts:
            clusters = np.repeat(clusters[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)
        mask_2d &= clusters == cluster

    block_ids = np.repeat(np.arange(n_blocks), n_concepts * n_rep)
    mask_flat = mask_2d.reshape(-1)

    block_ids = block_ids[mask_flat]
    raw = raw.reshape(-1, raw.shape[2], raw.shape[3], raw.shape[4])[mask_flat]
    if feat is not None:
        feat = feat.reshape(-1, feat.shape[2], feat.shape[3], feat.shape[4])[mask_flat]
    labels_flat = labels_raw.reshape(-1)[mask_flat] - (1 if category == "color" else 0)

    labels = format_labels(np.repeat(labels_flat[:, None], raw.shape[1], axis=1), category)

    if cluster is not None and category == "label":
        uniq = np.sort(np.unique(labels))
        mapping = {v: i for i, v in enumerate(uniq)}
        labels = np.vectorize(mapping.get)(labels)

    unique_labels = np.unique(labels)
    out_dim = len(unique_labels)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), test_idx = prepare_datasets(
        raw,
        feat,
        labels,
        block_ids,
        val_block,
        test_block,
        shuffle=shuffle,
        seed=seed,
        return_idx=True,
    )

    raw_reshaped = raw.reshape(-1, raw.shape[2], raw.shape[3])
    raw_mean, raw_std = compute_raw_stats(raw_reshaped)
    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        test_idx,
        (raw_mean, raw_std),
        unique_labels,
        out_dim,
    )


def train_category(
    raw: np.ndarray,
    feat: Optional[np.ndarray],
    label_dir: str,
    category: str,
    cluster: int | None,
    n_concepts: int,
    n_rep: int,
    n_blocks: int,
    val_block: int,
    test_block: int,
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    C: int,
    T: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
):
    (
        (X_train, y_train),
        (X_val, y_val),
        (X_test_raw, y_test),
        test_idx,
        stats,
        uniq_labels,
        out_dim,
    ) = build_category_data(
        raw,
        feat,
        label_dir,
        category,
        cluster,
        n_concepts,
        n_rep,
        n_blocks,
        val_block,
        test_block,
        shuffle=shuffle,
        seed=seed,
    )

    raw_mean, raw_std = stats
    # Normalize only the raw EEG part and keep features untouched
    X_train_raw = normalize_raw(X_train[:, :, :T], raw_mean, raw_std)
    X_val_raw = normalize_raw(X_val[:, :, :T], raw_mean, raw_std)
    X_test_raw_norm = normalize_raw(X_test_raw[:, :, :T], raw_mean, raw_std)

    X_train = np.concatenate([X_train_raw, X_train[:, :, T:]], axis=2)
    X_val = np.concatenate([X_val_raw, X_val[:, :, T:]], axis=2)
    X_test = np.concatenate([X_test_raw_norm, X_test_raw[:, :, T:]], axis=2)

    scaler = None
    feat_dim = 0
    if feat is not None:
        # Extract spectral features from the end of the time dimension
        F_train_scaled, scaler = standard_scale_features(
            X_train[:, :, T:], return_scaler=True
        )
        F_val_scaled = standard_scale_features(X_val[:, :, T:], scaler=scaler)
        F_test_scaled = standard_scale_features(X_test[:, :, T:], scaler=scaler)

        # Replace unscaled features with the standardized ones
        X_train = np.concatenate([X_train[:, :, :T], F_train_scaled], axis=2)
        X_val = np.concatenate([X_val[:, :, :T], F_val_scaled], axis=2)
        X_test = np.concatenate([X_test[:, :, :T], F_test_scaled], axis=2)

        feat_dim = F_train_scaled.shape[-1]

    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        feat_dim,
        out_dim,
        device,
        model_type,
        epochs,
        batch_size,
        lr,
        C,
        T,
    )

    # Return unnormalized raw EEG for evaluation
    return model, (raw_mean, raw_std), scaler, (X_test_raw[:, :, :T], y_test)


def two_stage_predict(eeg: np.ndarray, models: Dict[str, nn.Module], scalers, stats, device: str, model_type: str) -> int:
    idx_cluster, _ = majority_vote(
        eeg,
        models["label_cluster"],
        scalers.get("label_cluster"),
        stats["label_cluster"],
        device,
        model_type,
    )
    cat = f"label_cluster{idx_cluster}"
    idx_label, _ = majority_vote(
        eeg,
        models[cat],
        scalers.get(cat),
        stats[cat],
        device,
        model_type,
    )
    start, _ = CLUSTER_RANGES[idx_cluster]
    return start + idx_label - 1


def evaluate(
    raw: Optional[np.ndarray],
    labels_all: Optional[np.ndarray],
    test_block: int,
    models: Dict[str, nn.Module],
    scalers,
    stats,
    device: str,
    model_type: str,
    *,
    shuffle: bool = False,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, float, np.ndarray]:
    if shuffle:
        if X_test is None or y_test is None:
            raise ValueError("Test data must be provided when shuffle is True")
        preds_label = []
        preds_two = []
        for eeg in X_test:
            idx_one, _ = majority_vote(
                eeg[None, ...],
                models["label"],
                scalers.get("label"),
                stats["label"],
                device,
                model_type,
            )
            preds_label.append(idx_one)
            preds_two.append(
                two_stage_predict(eeg[None, ...], models, scalers, stats, device, model_type)
            )

        labels_true = y_test
        preds_label = np.array(preds_label)
        preds_two = np.array(preds_two)
    else:
        if raw is None or labels_all is None:
            raise ValueError("Raw data and labels must be provided when shuffle is False")
        n_blocks, n_concepts, n_rep = labels_all.shape[:3]
        preds_label = []
        preds_two = []
        labels_true = []
        for c in range(n_concepts):
            for r in range(n_rep):
                eeg = raw[test_block, c, r]
                lbl = labels_all[test_block, c, r] - 1
                labels_true.append(lbl)
                idx_one, _ = majority_vote(
                    eeg,
                    models["label"],
                    scalers.get("label"),
                    stats["label"],
                    device,
                    model_type,
                )
                preds_label.append(idx_one)
                preds_two.append(
                    two_stage_predict(eeg, models, scalers, stats, device, model_type)
                )

        labels_true = np.array(labels_true)
        preds_label = np.array(preds_label)
        preds_two = np.array(preds_two)

    acc_one = (preds_label == labels_true).mean()
    acc_two = (preds_two == labels_true).mean()
    cm_one = confusion_matrix(labels_true, preds_label)
    cm_two = confusion_matrix(labels_true, preds_two)
    return acc_one, cm_one, acc_two, cm_two


def main() -> None:
    p = argparse.ArgumentParser(description="Train label models and evaluate two-stage approach")
    p.add_argument("--raw_dir", default="./data/Preprocessing/Segmented_500ms_sw")
    p.add_argument("--label_dir", default="./data/meta_info")
    p.add_argument("--subj_name", default="sub3")
    p.add_argument("--save_dir", default="./Classifiers/checkpoints")
    p.add_argument("--model", choices=["glmnet", "eegnet", "deepnet"], default="glmnet")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shuffle", action="store_true")
    args = p.parse_args()

    device = args.device
    raw = np.load(os.path.join(args.raw_dir, f"{args.subj_name}.npy"))
    n_blocks, n_concepts, n_rep, n_win, C, T = raw.shape
    ckpt_seed_dir = os.path.join(args.save_dir, "mono", args.subj_name, f"seed{args.seed}")
    val_block, test_block = block_split(args.seed, n_blocks, ckpt_seed_dir)

    if args.model == "glmnet":
        duration_ms = int(re.search(r"_(\d+)ms_", os.path.basename(args.raw_dir)).group(1)) / 1000
        feat_all = mlpnet.compute_features(raw.reshape(-1, C, T), win_sec=duration_ms).reshape(
            n_blocks, n_concepts * n_rep, n_win, C, -1
        )
    else:
        feat_all = None

    raw = raw.reshape(n_blocks, n_concepts * n_rep, n_win, C, T)

    categories = ["label", "label_cluster"] + [f"label_cluster{i}" for i in range(len(CLUSTER_RANGES))]

    models: Dict[str, nn.Module] = {}
    scalers: Dict[str, Any] = {}
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    test_data_label: Tuple[np.ndarray, np.ndarray] | None = None

    for cat in categories:
        if cat.startswith("label_cluster") and cat != "label_cluster":
            cluster_idx = int(cat.replace("label_cluster", ""))
            c = cluster_idx
            model, st, sc, _ = train_category(
                raw,
                feat_all,
                args.label_dir,
                "label",
                c,
                n_concepts,
                n_rep,
                n_blocks,
                val_block,
                test_block,
                args.model,
                args.epochs,
                args.bs,
                args.lr,
                device,
                C,
                T,
                shuffle=args.shuffle,
                seed=args.seed,
            )
        else:
            cluster_idx = None
            model, st, sc, test_pair = train_category(
                raw,
                feat_all,
                args.label_dir,
                cat,
                cluster_idx,
                n_concepts,
                n_rep,
                n_blocks,
                val_block,
                test_block,
                args.model,
                args.epochs,
                args.bs,
                args.lr,
                device,
                C,
                T,
                shuffle=args.shuffle,
                seed=args.seed,
            )
            if cat == "label":
                test_data_label = test_pair
        models[cat] = model
        scalers[cat] = sc
        stats[cat] = st

    labels_all = np.load(os.path.join(args.label_dir, "All_video_label.npy"))
    if labels_all.shape[1] == n_concepts:
        labels_all = np.repeat(labels_all[:, :, None], n_rep, axis=2)

    if args.shuffle:
        if test_data_label is None:
            raise RuntimeError("Label test data missing")
        X_test, y_test = test_data_label
        acc_one, cm_one, acc_two, cm_two = evaluate(
            None,
            None,
            0,
            models,
            scalers,
            stats,
            device,
            args.model,
            shuffle=True,
            X_test=X_test,
            y_test=y_test,
        )
    else:
        acc_one, cm_one, acc_two, cm_two = evaluate(
            raw.reshape(n_blocks, n_concepts, n_rep, n_win, C, T),
            labels_all,
            test_block,
            models,
            scalers,
            stats,
            device,
            args.model,
        )

    print(f"One-step accuracy: {acc_one:.3f}")
    print("Confusion matrix one-step:\n", cm_one)
    print(f"Two-stage accuracy: {acc_two:.3f}")
    print("Confusion matrix two-stage:\n", cm_two)


if __name__ == "__main__":
    main()
