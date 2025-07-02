import numpy as np
import argparse
import os


def build_label_clusters(labels: np.ndarray) -> np.ndarray:
    """Map raw labels to cluster ids.

    Parameters
    ----------
    labels : np.ndarray
        Array containing label ids in the range 1-40.

    Returns
    -------
    np.ndarray
        Array with the same shape as ``labels`` where ids have been
        mapped to 9 cluster indices in the range 0-8.
    """
    clusters = np.zeros_like(labels, dtype=np.int64)
    ranges = [
        (1, 7),
        (8, 11),
        (12, 14),
        (15, 18),
        (19, 21),
        (22, 27),
        (28, 32),
        (33, 35),
        (36, 40),
    ]
    for idx, (start, end) in enumerate(ranges):
        mask = (labels >= start) & (labels <= end)
        clusters[mask] = idx
    return clusters


def main(label_path: str, output_path: str) -> None:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    labels = np.load(label_path)
    clusters = build_label_clusters(labels)
    np.save(output_path, clusters)
    unique_vals = np.unique(clusters)
    print(f"Saved clustered labels to {output_path}. Unique clusters: {unique_vals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create clustered labels from All_video_label.npy")
    parser.add_argument(
        "--label_path",
        default="./data/meta_info/All_video_label.npy",
        help="Path to All_video_label.npy",
    )
    parser.add_argument(
        "--output_path",
        default="./data/meta_info/All_video_label_cluster.npy",
        help="Where to save the clustered labels",
    )
    args = parser.parse_args()
    main(args.label_path, args.output_path)
