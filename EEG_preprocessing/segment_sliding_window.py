import numpy as np
import os
from tqdm import tqdm

fre = 200
samples_per_concept = 5 * 2 * fre  # 5 videos of 2s
samples_hint = 3 * fre             # 3 seconds of hint
window_size = int(fre*0.5)               # 0.5 sec
step_size = int(fre*0.25)              # 0.25 sec

n_blocks = 7
n_concepts = 40
n_videos = 5

input_folder = './data/EEG'
label_file = './data/meta_info/All_video_label.npy'
output_folder = './data/EEG_500ms_sw'
os.makedirs(output_folder, exist_ok=True)

all_labels = np.load(label_file)  # shape: (7, 40)

for subj in range(1, 21):
    input_path = os.path.join(input_folder, f'sub{subj}.npy')
    eeg = np.load(input_path)  # shape: (7, 62, 104000)
    print(f"\n🧠 Processing subject {subj} - shape {eeg.shape}")

    segments = []
    labels = []
    blocks = []

    for block in range(n_blocks):
        eeg_block = eeg[block]  # (62, 104000)
        block_labels = all_labels[block]  # (40,)

        for concept in range(n_concepts):
            start = concept * (samples_hint + samples_per_concept)
            if start + samples_hint + n_videos * 2 * fre > eeg_block.shape[1]:
                print(f"⚠️ Skip: Block {block}, Concept {concept} - not enough space for 5 videos")
                break

            for vid in range(n_videos):
                seg_start = start + samples_hint + vid * 2 * fre
                seg_end = seg_start + 2 * fre  # 2s

                eeg_vid = eeg_block[:, seg_start:seg_end]  # (62, 400)
                if eeg_vid.shape[1] < 2 * fre:
                    print(f"⚠️ Skip: Block {block}, Concept {concept}, Vid {vid} - too short")
                    continue

                for win_start in range(0, eeg_vid.shape[1] - window_size + 1, step_size):
                    segment = eeg_vid[:, win_start:win_start + window_size]  # (62, window_size)
                    segments.append(segment)
                    labels.append(block_labels[concept]-1)
                    blocks.append(block)

    segments = np.stack(segments)  # (N, 62, window_size)
    labels = np.array(labels)      # (N,)
    blocks = np.array(blocks)      # (N,)

    output_path = os.path.join(output_folder, f"sub{subj}_segmented.npz")
    np.savez(output_path, eeg=segments.astype(np.float32),
             labels=labels.astype(np.int64),
             blocks=blocks.astype(np.int64))
    print(f"✅ Saved sub{subj} with shape {segments.shape} to {output_folder}")