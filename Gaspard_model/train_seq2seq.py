import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from models.models import Seq2SeqTransformer

# === Dataset ===
class EEGVideoDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        eeg = data['eeg']    # (7, 512)
        z0 = data['z0']      # (6, 256)
        return torch.tensor(eeg, dtype=torch.float32), torch.tensor(z0, dtype=torch.float32)

# === Entraînement ===
def train_model(data_dir, save_path, epochs=50, batch_size=64, lr=5e-4):
    dataset = EEGVideoDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Seq2SeqTransformer().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for eeg, z0 in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg, z0 = eeg.cuda(), z0.cuda()
            pred = model(eeg)  # (B, 6, 256)
            loss = loss_fn(pred, z0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), save_path)
    print("✅ Modèle Seq2Seq sauvegardé :", save_path)

# Exemple d'appel
if __name__ == "__main__":
    train_model(
        data_dir=os.path.expanduser("~/EEG2Video/data/EEG_Latent_pairs"),
        save_path=os.path.expanduser("~/EEG2Video_model/checkpoints/seq2seq.pt")
    )
