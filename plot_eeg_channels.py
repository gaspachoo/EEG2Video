#!/usr/bin/env python3
"""Plot a subset of channels from a raw EEG recording."""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_first_sample(data: np.ndarray) -> np.ndarray:
    """Collapse leading dimensions to get a single (channels, time) array."""
    print(f"Data shape: {data.shape}")
    if data.ndim < 2:
        raise ValueError("EEG data must have at least 2 dimensions")
    if data.ndim > 2:
        reshaped = data.reshape(-1, data.shape[-2], data.shape[-1])
        sample = reshaped[0]
    else:
        sample = data
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw EEG channels")
    parser.add_argument("--file", type=str, help="Path to the raw EEG .npy file")
    parser.add_argument("--channels", type=int, default=10,
                        help="Number of channels to plot")
    parser.add_argument("--timepoints", type=int, default=500,
                        help="Number of time points to display")
    args = parser.parse_args()

    eeg_data = np.load(args.file)
    sample = load_first_sample(eeg_data)

    num_channels = min(args.channels, sample.shape[0])
    t = min(args.timepoints, sample.shape[1])
    chosen = np.random.choice(sample.shape[0], num_channels, replace=False)

    plt.figure(figsize=(12, num_channels * 1.5))
    for idx, ch in enumerate(chosen):
        offset = idx * np.max(np.abs(sample))
        plt.plot(sample[ch, :t] + offset, label=f"Channel {ch}", linewidth=3)

    plt.xlabel("Time")
    plt.title("Raw EEG Channels")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("eeg_channels_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
