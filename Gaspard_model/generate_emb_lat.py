import os
import numpy as np
import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
from train_glmnet_dual import GLMNet, OCCIPITAL_IDX, split_raw_2s_to_1s

# === Parameters ===
emb_dim = 256 # be careful, embedding dimension/2 here
save_root = os.path.expanduser("~/EEG2Video")
raw_path = f"{save_root}/data/Segmented_Rawf_200Hz_2s/sub3.npy"
feat_path = f"{save_root}/data/DE_1per1s/sub3.npy"
video_root = f"{save_root}/data/Video_gifs/"
output_dir = f"{save_root}/data/Pairs_latents_embeddings/"
ckpt_dir = f"{save_root}/Gaspard_model/checkpoints/cv_glmnetv2/"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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


# === Generate EEG embeddings + video latents ===
for block in range(7):
    model = models[block]
    video_dir = os.path.join(video_root, f"Block{block}")

    for concept in tqdm(range(40), desc=f"Block {block}"):
        for rep in range(5):
            for w in range(2):  # Two 1-second windows per video
                eeg = raw1s[block, concept, rep, w]        # shape: (62, 200)
                de = feat[block, concept, rep, w]          # shape: (62, 5)

                # Format EEG and DE features as model inputs
                eeg_tensor = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 62, 200)
                de_tensor = torch.tensor(de[OCCIPITAL_IDX], dtype=torch.float32).unsqueeze(0).to(device)   # (1, 12, 5)

                # Compute EEG embedding (concatenated global + local features)
                with torch.no_grad():
                    g = model.raw_global(eeg_tensor)  # (1, 256)
                    l = model.freq_local(de_tensor)   # (1, 256)
                    emb = torch.cat([g, l], dim=1).squeeze(0).cpu().numpy()  # (512,)

                # Load corresponding video and extract latent for 6 frames
                gif_index = concept * 5 + rep + 1
                gif_path = os.path.join(video_dir, f"{gif_index}.gif")
                if not os.path.exists(gif_path):
                    print(f"❌ Missing video: {gif_path}")
                    continue

                frames = imageio.mimread(gif_path)
                #print(f"Loaded {len(frames)} frames from {gif_path}")
                if len(frames) != 6:
                    print(f"⚠️ GIF {gif_path} has {len(frames)} frames, expected 6.")
                    continue
                frames = [transform(f) for f in frames]  # use all 6 frames from the GIF
                frames = torch.stack(frames).to(device)
                #print("frames tensor shape:", frames.shape)  # (6, 3, 288, 512) expected
                with torch.no_grad():
                    z_latents = vae.encode(frames).latent_dist.mean
                    #print("z_latents shape:", z_latents.shape)  # should be (6, C, H, W)
                    z = z_latents.flatten(1)  # flatten (C, H, W) to a single vector → (6, 9216)
                z0 = z.cpu().numpy()  # (6, 9216)

                assert emb.shape == (512,), f"Invalid emb shape: {emb.shape}"
                assert z0.shape[0] == 6, f"Invalid z0 shape: {z0.shape}"

                # ⚠️ downstream models may project z0 from 9216 to 256 if needed
                save_name = f"sub3_b{block}_c{concept}_r{rep}_w{w}.npz"
                np.savez(os.path.join(output_dir, save_name), eeg=emb[np.newaxis, :], z0=z0)


print("Done!")
