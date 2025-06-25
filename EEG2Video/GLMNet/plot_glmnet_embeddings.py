import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_embeddings(path: str) -> np.ndarray:
    """Load and reshape embeddings to (blocks, concepts, reps, windows, dim)."""
    data = np.load(path)
    if data.ndim == 2:
        # Flattened (7*40*5*7, dim)
        dim = data.shape[-1]
        data = data.reshape(7, 40, 5, 7, dim)
    if data.ndim != 5:
        raise ValueError(f"Unexpected embedding shape {data.shape}")
    return data


def plot_block(embeddings: np.ndarray, block_idx: int, concept_ids: list[int], save_path: str | None = None) -> None:
    """Plot PCA-reduced embeddings for given block and concepts."""
    n_concepts = len(concept_ids)
    fig, axes = plt.subplots(1, n_concepts, figsize=(5 * n_concepts, 4), squeeze=False)

    for i, cid in enumerate(concept_ids):
        ax = axes[0, i]
        emb = embeddings[block_idx, cid]  # shape (5, 7, dim)
        reps, n_win, dim = emb.shape
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb.reshape(-1, dim))
        coords = coords.reshape(reps, n_win, 2)

        for r in range(reps):
            ax.plot(coords[r, :, 0], coords[r, :, 1], marker="o", label=f"rep {r + 1}")
        ax.set_title(f"Concept {cid}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GLMNet embeddings for a block")
    parser.add_argument("--embeddings", default = "./data/GLMNet/EEG_embeddings_sw/sub3.npy", help="Path to embeddings .npy file")
    parser.add_argument("--block", type=int, default=0, help="Block index")
    parser.add_argument(
        "--concepts", type=str, default="0,1,2", help="Comma-separated concept indices"
    )
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    concept_ids = [int(c) for c in args.concepts.split(",")]
    emb = load_embeddings(args.embeddings)
    plot_block(emb, args.block, concept_ids, args.save)
