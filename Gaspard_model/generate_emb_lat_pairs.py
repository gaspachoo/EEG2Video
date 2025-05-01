import os
import numpy as np
import torch
import imageio
from tqdm import tqdm
from torchvision import transforms
from diffusers.models import AutoencoderKL
from models.models import ShallowNetEncoder, MLPEncoder_feat, GLMNetFeatureExtractor

def extract_eeg_embedding(eeg_raw, de_feat, g_model, l_model):
    device = next(g_model.parameters()).device
    eeg_tensor = torch.tensor(eeg_raw, dtype=torch.float32).to(device)
    de_tensor = torch.tensor(de_feat, dtype=torch.float32).to(device)
    with torch.no_grad():
        embedding = GLMNetFeatureExtractor(g_model, l_model)(eeg_tensor, de_tensor)
    return embedding.cpu().numpy()

def extract_video_latent(gif_path, vae, transform):
    frames = imageio.mimread(gif_path)
    frames = [transform(f) for f in frames]
    frames = torch.stack(frames).to(vae.device)  # (6, 3, 288, 512)
    with torch.no_grad():
        z = vae.encode(frames).latent_dist.mean.view(6, -1)
    return z.cpu().numpy()

def main():
    root = os.environ["HOME"] + "/EEG2Video"
    eeg_dir = f"{root}/data/EEG_500ms_sw/"
    feat_dir = f"{root}/data/DE_500ms_sw/"
    video_root = f"{root}/data/Video_gifs/"
    output_dir = f"{root}/data/EEG_Latent_pairs/"
    g_ckpt = f"{root}/Gaspard_model/checkpoints/cv_shallownet/best_fold0.pth"
    l_ckpt = f"{root}/Gaspard_model/checkpoints/cv_mlp_DE/best_fold0.pt"

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load models
    g = ShallowNetEncoder(62, time_len=100).to(device)
    g.load_state_dict(torch.load(g_ckpt, map_location=device)["encoder"])
    g.eval()

    l = MLPEncoder_feat().to(device)
    l.load_state_dict(torch.load(l_ckpt, map_location=device), strict=False)
    l.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    transform = transforms.ToTensor()

    subj_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith(".npz")])

    for subj_file in tqdm(subj_files, desc="📦 Subjects"):
        subj_name = subj_file.replace(".npz", "")
        eeg_npz = np.load(os.path.join(eeg_dir, subj_file))
        feat_npz = np.load(os.path.join(feat_dir, subj_file.replace("segmented", "features")))

        eeg_data = eeg_npz["eeg"]     # (9800, 62, 100)
        print("EEG shape:", eeg_data.shape)  # <- must be (9800, 62, 100)
        feat_data = feat_npz["de"]    # (9800, 62, 5)
        print("DE shape:", feat_data.shape)

        assert eeg_data.shape[0] % 7 == 0

        for block in tqdm(range(7), leave=False, desc=f"📁 Blocks ({subj_name})"):
            video_dir = os.path.join(video_root, f"Block{block}")
            for concept in range(40):
                for rep in range(5):
                    eeg_segments = []
                    for window in range(7):
                        idx = ((block * 40 * 5 + concept * 5 + rep) * 7) + window
                        eeg = eeg_data[idx]  # (62, 100)
                        de = feat_data[idx]  # (62, 5)
                        
                        print(f"EEG shape: {eeg.shape}")  # <- must be (62, 100)
                        print(f"DE shape: {de.shape}")
                        
                        eeg_tensor = torch.tensor(eeg[np.newaxis], dtype=torch.float32).to(device)
                        de_tensor = torch.tensor(de[np.newaxis], dtype=torch.float32).to(device)
                        with torch.no_grad():
                            emb = GLMNetFeatureExtractor(g, l)(eeg_tensor, de_tensor)
                        eeg_segments.append(emb.squeeze(0).cpu().numpy())

                    if len(eeg_segments) != 7:
                        print(f"❌ Skipped: not enough EEG segments for b{block}, c{concept}, r{rep}")
                        continue

                    eeg_embedding = np.stack(eeg_segments, axis=0)  # (7, 512)

                    gif_index = concept * 5 + rep + 1
                    gif_path = os.path.join(video_dir, f"{gif_index}.gif")
                    if not os.path.exists(gif_path):
                        print(f"❌ Missing video: {gif_path}")
                        continue

                    z0 = extract_video_latent(gif_path, vae, transform)  # (6, 256)
                    print(f"eeg_embedding.shape = {np.array(eeg_segments).shape}")
                    print(f"z0.shape = {z0.shape}")
                    assert eeg_embedding.shape == (7, 512)
                    assert z0.shape == (6, 256)

                    save_name = f"{subj_name}_b{block}_c{concept}_r{rep}.npz"
                    np.savez(os.path.join(output_dir, save_name), eeg=eeg_embedding, z0=z0)

if __name__ == "__main__":
    main()
