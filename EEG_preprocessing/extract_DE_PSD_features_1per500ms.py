import os
import numpy as np
from DE_PSD import DE_PSD
from tqdm import tqdm
import argparse

# --- Parameters ---
FS = 200               # fréquence d'échantillonnage (Hz)
WIN_SEC = 0.5          # longueur de fenêtre en secondes
WIN_T = int(FS * WIN_SEC)  # points par fenêtre (100)

# --- Extraction DE/PSD sur fenêtres de 500 ms ---
def process_subject(raw_path, de_out_path, psd_out_path):
    # raw shape: (7, 40, 5, 7, 62, 100)
    raw = np.load(raw_path)
    # Pré-allocation
    DE_data  = np.zeros((7, 40, 5, 7, 62, 5), dtype=np.float32)
    PSD_data = np.zeros((7, 40, 5, 7, 62, 5), dtype=np.float32)

    for blk in range(raw.shape[0]):
        for cls in range(raw.shape[1]):
            for rep in range(raw.shape[2]):
                for win in range(raw.shape[3]):
                    segment = raw[blk, cls, rep, win, :, :]  # (62, 100)
                    de, psd = DE_PSD(segment, FS, WIN_SEC)
                    DE_data[blk, cls, rep, win]  = de  # (62,5)
                    PSD_data[blk, cls, rep, win] = psd # (62,5)

    # Sauvegarde
    np.save(de_out_path,  DE_data)
    np.save(psd_out_path, PSD_data)
    print(f"Saved DE/PSD: {os.path.basename(de_out_path)} / {os.path.basename(psd_out_path)}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = os.environ.get('HOME', os.environ.get('USERPROFILE')) + '/EEG2Video'
    parser.add_argument('--raw_dir',   default=f"{root}/data/Segmented_500ms_sw", help='dossier .npy fenêtré raw EEG')
    parser.add_argument('--de_dir',    default=f"{root}/data/DE_500ms_sw",    help='où sauvegarder DE')
    parser.add_argument('--psd_dir',   default=f"{root}/data/PSD_500ms_sw",   help='où sauvegarder PSD')
    parser.add_argument('--subs',      nargs='+', type=int, default=list(range(1,21)), help='numéro des sujets')
    args = parser.parse_args()

    os.makedirs(args.de_dir,  exist_ok=True)
    os.makedirs(args.psd_dir, exist_ok=True)

    for sub in args.subs:
        raw_path    = os.path.join(args.raw_dir, f'sub{sub}.npy')
        de_out_path = os.path.join(args.de_dir,  f'sub{sub}.npy')
        psd_out_path= os.path.join(args.psd_dir, f'sub{sub}.npy')
        print(f"Processing subject {sub}...")
        process_subject(raw_path, de_out_path, psd_out_path)
