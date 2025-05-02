import os
import numpy as np
import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
from train_glmnet_dual import GLMNet, OCCIPITAL_IDX, split_raw_2s_to_1s

# === Settings ===
emb_dim = 256 # be careful, embedding dimension/2 here
save_root = os.path.expanduser("~/EEG2Video")
raw_path = f"{save_root}/data/Segmented_Rawf_200Hz_2s/sub3.npy"
feat_path = f"{save_root}/data/DE_1per1s/sub3.npy"
video_root = f"{save_root}/data/Video_gifs/"
output_dir = f"{save_root}/data/Pairs_latents_embeddings/"
ckpt_dir = f"{save_root}/Gaspard_model/checkpoints/cv_glmnetv2/"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load blocks models ===
models = []
for fold in range(7):
    model = GLMNet(out_dim=40, emb_dim=emb_dim).to(device)
    ckpt_path = os.path.join(ckpt_dir, f"sub3_fold{fold}_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    models.append(model)

# === Load VAE Stable Diffusion ===
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval()
transform = transforms.ToTensor()

# === EEG Data ===
raw = np.load(raw_path)     # (7, 40, 5, 62, 400)
feat = np.load(feat_path)   # (7, 40, 5, 62, 5)
raw1s = split_raw_2s_to_1s(raw)  # (7, 40, 5, 2, 62, 200)

# === Generate pairs EEG - z0 ===
for block in range(7):
    model = models[block]
    video_dir = os.path.join(video_root, f"Block{block}")

    for tqdm(concept in range(40), desc=f"Block {block}", total=40):
        for rep in range(5):
            for w in range(2):  # 2 windows per block
                eeg_segments = []
                for seg in range(7):  # sliding windows 7 x 200
                    eeg = raw1s[block, concept, rep, w]  # (62, 200)
                    de = feat[block, concept, rep,w]       # (62, 5)

                    eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    de_tensor = torch.tensor(de[OCCIPITAL_IDX], dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        g = model.raw_global(eeg_tensor)  # (1, 256)
                        l = model.freq_local(de_tensor)   # (1, 256)
                        emb = torch.cat([g, l], dim=1).squeeze(0).cpu().numpy()  # (512,)
                        eeg_segments.append(emb)

                eeg_embedding = np.stack(eeg_segments, axis=0)  # (7, 512)

                gif_index = concept * 5 + rep + 1
                gif_path = os.path.join(video_dir, f"{gif_index}.gif")
                if not os.path.exists(gif_path):
                    print(f"❌ Missing video: {gif_path}")
                    continue

                frames = imageio.mimread(gif_path)
                frames = [transform(f) for f in frames[w*3:w*3+6]]  # select frames 0-5 or 3-8
                frames = torch.stack(frames).to(device)
                with torch.no_grad():
                    z = vae.encode(frames).latent_dist.mean.reshape(6, -1)

                z0 = z.cpu().numpy()  # (6, 256)

                save_name = f"sub3_b{block}_c{concept}_r{rep}_w{w}.npz"
                np.savez(os.path.join(output_dir, save_name), eeg=eeg_embedding, z0=z0)

print("✅ Fini. Fichiers EEG - latents enregistrés avec w0 et w1.")
