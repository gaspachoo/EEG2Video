import os
import matplotlib.pyplot as plt
import numpy as np

def load_all_eeg_data_by_subject(data_dir):
    data = {}
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            if "session" in name:
                subject = name.split("_")[0]  # sub1
            else:
                subject = name
            filepath = os.path.join(data_dir, file)
            eeg_array = np.load(filepath)  # (7, 62, 104000)
            if subject not in data:
                data[subject] = []
            data[subject].append(eeg_array)
    return data

data_dir = "./dataset/EEG"
eeg_data = load_all_eeg_data_by_subject(data_dir)

# To access sub 1 data
sub1_data = eeg_data['sub1']  # list of 2 tables (2 sessions)



def plot_eeg_block(eeg_data, subject='sub1', session_idx=0, block_idx=0, channels=[0, 1, 2, 3, 4], fs=200):
    """
    eeg_data: dict with subject -> list of sessions (each session is (7, 62, 104000))
    subject: subject id, like 'sub1'
    session_idx: 0 or 1 for sub1 (only 0 for others)
    block_idx: index from 0 to 6
    channels: list of channel indices to plot
    fs: sampling frequency (200 Hz)
    """
    session = eeg_data[subject][session_idx]  # shape (7, 62, 104000)
    block = session[block_idx]  # shape (62, 104000)

    time = np.arange(block.shape[1]) / fs  # in seconds

    plt.figure(figsize=(12, 8))
    for i, ch in enumerate(channels):
        plt.plot(time, block[ch] + i * 000, label=f'Channel {ch}')  #Offset for lisibility can be set

    plt.xlabel('Time (s)')
    plt.ylabel('EEG Signal (offset for display)')
    plt.title(f'{subject.upper()} – Session {session_idx+1} – Block {block_idx+1}')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_eeg_block(eeg_data, subject='sub7', session_idx=0, block_idx=0)
