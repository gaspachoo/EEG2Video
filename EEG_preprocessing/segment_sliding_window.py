import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def seg_sliding_window(data, win_s, step_s, fs=200):
    """Segment data into sliding windows.
    data : np.ndarray
        Input data of shape (7, 40, 5, 62, 2 * fs)
    win_s : float
        Window size in seconds (e.g., 0.5 for 500 ms)
    step_s : float
        Step size in seconds (e.g., 0.25 for 250 ms)
    fs : int
        Sampling frequency in Hz (default 200 Hz)"""
    
    win_t = int(fs * win_s)   # number of time points per window (100)
    step_t = int(fs * step_s) # step between windows (50)
    # Sliding window along the time axis (-1)
    windows = sliding_window_view(data, window_shape=win_t, axis=-1)
    # windows.shape -> (7, 40, 5, 62, 301, 100)

    # Subsample with step STEP_T to obtain 7 windows
    windows = windows[..., ::step_t, :]
    # windows.shape -> (7, 40, 5, 62, 7, 100)

    # Rearrange to get (7, 40, 5, 7, 62, 100)
    windows = windows.transpose(0, 1, 2, 4, 3, 5)

    return windows


if __name__ == "__main__":

    # Input directory
    INPUT_DIR = './data/Preprocessing/Segmented_Rawf_200Hz_2s'

    # Segmenting settings
    FS = 200                  # sampling frequency (Hz)
    WIN_S = 0.5               # window (seconds)
    STEP_S = 0.25             # overlap (seconds)

    # Output directory depends on WIN_S
    OUTPUT_DIR = f'./data/Preprocessing/Segmented_{int(1000*WIN_S)}ms_sw'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith('.npy'):
            continue

        path_in = os.path.join(INPUT_DIR, fname)
        data = np.load(path_in)  # shape: (7, 40, 5, 62, 400)
        
        # Check if data has the expected shape
        if data.ndim != 5 or data.shape[-1] != 2 * FS:
            print(f"Skipping {fname}: unexpected shape {data.shape}")
            continue
        
        windows = seg_sliding_window(data, WIN_S, STEP_S, fs=FS)
        
        # Save segmented windows
        out_path = os.path.join(OUTPUT_DIR, fname)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(out_path, windows)

        print(f"Saved segmented windows for {fname} -> {windows.shape}")
        
        
