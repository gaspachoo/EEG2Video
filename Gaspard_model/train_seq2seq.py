import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Gaspard_model.old.eeg2video_dataset import EEG2VideoDataset
from models.my_autoregressive_transformer import myTransformer
from tqdm import tqdm

# === Config ===
data_dir = os.path.expanduser("~/EEG2Video/data/Pairs_latents_embeddings/")
batch_size = 32
lr = 5e-4
n_epochs = 100
save_path = os.path.expanduser("~/EEG2Video/Gaspard_model/checkpoints/seq2seq/")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Dataset ===
dataset = EEG2VideoDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model ===
model = myTransformer(d_model=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# === Training loop ===
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for eeg, z0 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        eeg = eeg.to(device)           # (B, 2, 512)
        z0 = z0.to(device)             # (B, 6, 9216)

        # Create padded tgt with 1 empty step (start token)
        b, t, dim = z0.shape
        z0_pad = torch.zeros((b, 1, dim), device=device)
        tgt = torch.cat([z0_pad, z0], dim=1)[:, :-1, :]  # input to decoder

        optimizer.zero_grad()
        _, pred = model(eeg, tgt)  # output: (B, 6, 9216)
        loss = criterion(pred, z0)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# === Save final model ===
torch.save(model.state_dict(), save_path+"seq2seq.pt")
print(f"✅ Model saved to {save_path}")
