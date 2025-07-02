import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings with optional reshaping."""
    data = np.load(path)
    if data.ndim == 2:
        # Flattened (7*40*5*7, dim)
        dim = data.shape[-1]
        data = data.reshape(7, 40, 5, 7, dim)
    if data.ndim not in {5, 6}:
        raise ValueError(f"Unexpected embedding shape {data.shape}")
    return data


def plot_block(
    embeddings: np.ndarray,
    block_idx: int,
    window_idx: int,
    concept_ids: list[int],
    save_path: str | None = None,
) -> None:
    """Plot temporal curves for a selected channel."""

    n_concepts = len(concept_ids)
    fig, axes = plt.subplots(1, n_concepts, figsize=(5 * n_concepts, 3), squeeze=False)
    
    for i, cid in enumerate(concept_ids):
        ax = axes[0, i]
        emb = embeddings[block_idx, cid, :, window_idx]

        reps, time_len = emb.shape

        for r in range(reps):
            series = emb[r, :].reshape(-1)
            t = np.arange(series.size)
            ax.plot(t, series, label=f"rep {r + 1}")

        ax.set_title(f"Concept {cid}")
        ax.set_xlabel("time")
        ax.set_ylabel(f"Value")
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
        "--concepts",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Space-separated list of concept indices (e.g., --concepts 0 1 2)",
    )
    parser.add_argument("--window", type=int, default=0, help="Window index")
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Index of the embedding channel to plot",
    )
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    emb = load_embeddings(args.embeddings)
    plot_block(emb, args.block, args.window, args.concepts, args.save)

