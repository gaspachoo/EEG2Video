import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# Dossiers d'entrée et de sortie
INPUT_DIR = './data/Preprocessing/Segmented_Rawf_200Hz_2s'


# Paramètres de découpage
FS = 200                  # fréquence d'échantillonnage (Hz)
WIN_S = 0.5               # fenêtre (secondes)
STEP_S = 0.25             # recouvrement (secondes)
WIN_T = int(FS * WIN_S)   # points temporels par fenêtre (100)
STEP_T = int(FS * STEP_S) # pas entre fenêtres (50)

OUTPUT_DIR = './data/Segmented_{int(1000*WIN_S)}ms_sw'

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

    # Sliding window le long de l'axe temporel (-1)
    windows = sliding_window_view(data, window_shape=WIN_T, axis=-1)
    # windows.shape -> (7, 40, 5, 62, 301, 100)

    # Sous-échantillonner selon le pas STEP_T pour obtenir 7 fenêtres
    windows = windows[..., ::STEP_T, :]
    # windows.shape -> (7, 40, 5, 62, 7, 100)

    # Réorganiser pour avoir (7, 40, 5, 7, 62, 100)
    windows = windows.transpose(0, 1, 2, 4, 3, 5)

    # Enregistrement
    out_path = os.path.join(OUTPUT_DIR, fname)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(out_path, windows)

    print(f"Saved segmented windows for {fname} -> {windows.shape}")
