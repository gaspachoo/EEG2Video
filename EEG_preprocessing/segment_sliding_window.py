import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def seg_sliding_window(data, win_s, step_s, fs=200):    
    
    win_t = int(fs * win_s)   # points temporels par fenêtre (100)
    step_t = int(fs * step_s) # pas entre fenêtres (50)
    # Sliding window le long de l'axe temporel (-1)
    windows = sliding_window_view(data, window_shape=win_t, axis=-1)
    # windows.shape -> (7, 40, 5, 62, 301, 100)

    # Sous-échantillonner selon le pas STEP_T pour obtenir 7 fenêtres
    windows = windows[..., ::step_t, :]
    # windows.shape -> (7, 40, 5, 62, 7, 100)

    # Réorganiser pour avoir (7, 40, 5, 7, 62, 100)
    windows = windows.transpose(0, 1, 2, 4, 3, 5)

    return windows


if __name__ == "__main__":

    # Input directory
    INPUT_DIR = './data/Preprocessing/Segmented_Rawf_200Hz_2s'

    # Segmenting settings
    FS = 200                  # fréquence d'échantillonnage (Hz)
    WIN_S = 0.5               # fenêtre (secondes)
    STEP_S = 0.25             # recouvrement (secondes)

    # Output directory depends on WIN_S
    OUTPUT_DIR = f'./data/Preprocessing/Segmented_{int(1000*WIN_S)}ms_sw'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith('.npy'):
            continue

        path_in = os.path.join(INPUT_DIR, fname)
        data = np.load(path_in)  # shape: (7, 40, 5, 62, 400)
        
        # Vérification de la forme
        if data.ndim != 5 or data.shape[-1] != 2 * FS:
            print(f"Skipping {fname}: unexpected shape {data.shape}")
            continue
        
        windows = seg_sliding_window(data, WIN_S, STEP_S, fs=FS)
        
        # Enregistrement
        out_path = os.path.join(OUTPUT_DIR, fname)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(out_path, windows)

        print(f"Saved segmented windows for {fname} -> {windows.shape}")
        
        
