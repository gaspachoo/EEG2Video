import numpy as np, os
from tqdm import tqdm
from DE_PSD import DE_PSD

fre = 200                                     # Hz
input_folder  = './dataset/EEG_500ms_sw'   # peu importe le nom
output_psd    = './dataset/PSD_500ms_sw'
output_de     = './dataset/DE_500ms_sw'
os.makedirs(output_psd, exist_ok=True)
os.makedirs(output_de,  exist_ok=True)

for fname in sorted(f for f in os.listdir(input_folder) if f.endswith('.npz')):
    data = np.load(os.path.join(input_folder, fname))
    eeg   = data['eeg']          # (N, 62, L)
    labels, blocks = data['labels'], data['blocks']
    
    L = eeg.shape[2]             # longueur du segment en samples
    seg_dur = L / fre            # durée (s)   ← plus besoin de segment_duration fixe
    print(f"🔄 {fname} | seg_len {L} samples = {seg_dur:.3f}s, total {eeg.shape[0]} segments")

    de_list, psd_list = [], []
    for seg in tqdm(eeg, desc=f"{fname}"):
        de, psd = DE_PSD(seg, fre, seg_dur)   # passe la durée déduite
        de_list.append(de);  psd_list.append(psd)

    de_arr  = np.stack(de_list)   # (N, 62, 5)
    psd_arr = np.stack(psd_list)

    base = fname.replace('_segmented.npz', '').replace('.npz', '')
    np.savez(f"{output_psd}/{base}_features.npz",
             psd=psd_arr.astype(np.float32),
             labels=labels.astype(np.int64),
             blocks=blocks.astype(np.int64))

    np.savez(f"{output_de}/{base}_features.npz",
             de=de_arr.astype(np.float32),
             labels=labels.astype(np.int64),
             blocks=blocks.astype(np.int64))

print("✅ DE & PSD generation done.")
