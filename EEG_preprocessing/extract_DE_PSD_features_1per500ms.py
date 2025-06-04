import os
import numpy as np
from .DE_PSD import DE_PSD
from tqdm import tqdm
import argparse


# --- Extraction DE/PSD sur fenêtres de 500 ms ---
def extract_de_psd_sw(raw, fs,win_sec):
    # raw shape: (7, 40, 5, 7, 62, 100) if 7 blocks, 40 concepts, 5 repetitions, 7 windows, 62 channels, 100 samples per window
    
    # Pré-allocation
    DE_data  = np.zeros((raw.shape[0], raw.shape[1], raw.shape[2], raw.shape[3], raw.shape[4], 5), dtype=np.float32)
    PSD_data  = np.zeros((raw.shape[0], raw.shape[1], raw.shape[2], raw.shape[3], raw.shape[4], 5), dtype=np.float32)
    

    for blk in range(raw.shape[0]):
        for cls in range(raw.shape[1]):
            for rep in range(raw.shape[2]):
                for win in range(raw.shape[3]):
                    segment = raw[blk, cls, rep, win, :, :]  # (62, 100)
                    de, psd = DE_PSD(segment, fs, win_sec)
                    DE_data[blk, cls, rep, win]  = de  # (62,5)
                    PSD_data[blk, cls, rep, win] = psd # (62,5)

    return DE_data, PSD_data

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--raw_dir',   default="./data/Preprocessing/Segmented_500ms_sw", help='dossier .npy fenêtré raw EEG')
    parser.add_argument('--de_dir',    default="./data/Preprocessing/DE_500ms_sw",    help='où sauvegarder DE')
    parser.add_argument('--psd_dir',   default="./data/Preprocessing/PSD_500ms_sw",   help='où sauvegarder PSD')
    parser.add_argument('--subs',      nargs='+', type=int, default=list(range(1,21)), help='numéro des sujets')
    args = parser.parse_args()
    
    FS = 200               # fréquence d'échantillonnage (Hz)
    WIN_SEC = 0.5          # longueur de fenêtre en secondes

    os.makedirs(args.de_dir,  exist_ok=True)
    os.makedirs(args.psd_dir, exist_ok=True)

    for sub in args.subs:
        raw_path    = os.path.join(args.raw_dir, f'sub{sub}.npy')
        de_out_path = os.path.join(args.de_dir,  f'sub{sub}.npy')
        psd_out_path= os.path.join(args.psd_dir, f'sub{sub}.npy')
        print(f"Processing subject {sub}...")
        
        raw = np.load(raw_path)
        DE_data, PSD_data = extract_de_psd_sw(raw, FS, WIN_SEC)
        # Sauvegarde
        np.save(de_out_path,  DE_data)
        np.save(psd_out_path, PSD_data)
        print(f"Saved DE/PSD: {os.path.basename(de_out_path)} / {os.path.basename(psd_out_path)}")
