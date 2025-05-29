import numpy as np
import matplotlib.pyplot as plt

# Load files
raw_data = np.load("./data/Preprocessing/Segmented_Rawf_200Hz_2s/sub1.npy")   # (7, 40, 5, 62, 400)
de_data = np.load("./data/Preprocessing/DE_1per2s/1.npy")                     # (1, 40, 5, 62, 5)
psd_data = np.load("./data/Preprocessing/PSD_1per2s/1.npy")                   # (1, 40, 5, 62, 5)

# Oz channel = index 59
channel_index = 59

# Time
time_raw = np.linspace(0, 2, 400)
time_feat = np.linspace(0, 2, 5)

# 5 concepts, 1st EEG of each concept
fig, axs = plt.subplots(3, 5, figsize=(20, 6), sharex=False)


for i in range(5):
    # RAW: bloc 0, concept i, eeg 0, channel Oz, all time points
    raw = raw_data[0, i, 0, channel_index, :]             # (400,)
    
    # DE/PSD: bloc 0, concept i, eeg 0, channel Oz, all 5 features
    de = de_data[0, i, 0, channel_index, :]               # (5,)
    psd = psd_data[0, i, 0, channel_index, :]             # (5,)

    axs[0, i].plot(time_raw, raw)
    
    axs[0, i].set_title(f"Concept {i+1}")
    
    axs[1, i].plot(time_feat, de, marker='o')
    axs[2, i].plot(time_feat, psd, marker='o')

# Label Y axes
axs[0, 0].set_ylabel("Raw EEG")
axs[1, 0].set_ylabel("DE")
axs[2, 0].set_ylabel("PSD")

# Label X
bands = ["delta", "theta", "alpha", "beta", "gamma"]
band_positions = np.arange(len(bands))/2  # [0, 1, 2, 3, 4]

for i in range(3):
    if i == 0:
        for ax in axs[i]:
            ax.set_xlabel("Time (s)")
            ax.grid(True)
    else:
        for ax in axs[i]:
            ax.set_xticks(band_positions,bands)  # Bands
            ax.grid(True)

plt.tight_layout()
plt.show()
