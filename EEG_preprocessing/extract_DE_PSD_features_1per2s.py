import numpy as np
from tqdm import tqdm
import os,sys

project_root = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from EEG_preprocessing.DE_PSD import DE_PSD

# Extract DE or PSD features with a 2-second window, that is, for each 2-second EEG segment, we extract a DE or PSD feature.
# Input the shape of (7 * 40 * 5 * 62 * 2s*fre), meaning 7 blocks, 40 concepts, 5 video clips, 62 channels, and 2s*fre time-points.
# Output the DE or PSD feature with (7 * 40 * 5 * 62 * 5), the last 5 indicates the frequency bands' number.

fre = 200

def extract_de_psd_raw(raw,fs=200):
    DE_data  = np.zeros((raw.shape[0], raw.shape[1], raw.shape[2], raw.shape[3], 5), dtype=np.float32)
    PSD_data  = np.zeros((raw.shape[0], raw.shape[1], raw.shape[2], raw.shape[3], 5), dtype=np.float32)

    for blk in range(raw.shape[0]):
        for cls in range(raw.shape[1]):
            for rep in range(raw.shape[2]):
                segment = raw[blk, cls, rep, :, :].reshape(raw.shape[3], 2*fre)
                de, psd = DE_PSD(segment, fs, 2, which="both")  # 2 sec
                DE_data[blk, cls, rep]  = de  # (62,5)
                PSD_data[blk, cls, rep] = psd # (62,5)
                
    return DE_data, PSD_data

if __name__ == "__main__":
    for subname in range(1,21):
        loaded_data = np.load('./data/Preprocessing/Segmented_Rawf_200Hz_2s/sub'+ str(subname) + '.npy')
        # (7 * 40 * 5 * 62 * 2*fre)
        print("Successfully loaded .npy file.")
        DE_data, PSD_data = extract_de_psd_raw(loaded_data,fre)

        os.makedirs("./data/Preprocessing/DE_1per2s", exist_ok=True)
        os.makedirs("./data/Preprocessing/PSD_1per2s", exist_ok=True)
        np.save("./data/Preprocessing/DE_1per2s/sub" + str(subname) +".npy", DE_data)
        np.save("./data/Preprocessing/PSD_1per2s/sub" + str(subname) + ".npy", PSD_data)
        print(f"Saved DE data in ./data/Preprocessing/DE_1per2s/sub{str(subname)}.npy")
        print(f"Saved PSD data in ./data/Preprocessing/PSD_1per2s/sub{str(subname)}.npy")