import os, argparse
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import imageio
import matplotlib.pyplot as plt
from EEG_preprocessing.segment_raw_signals_200Hz import extract_2s_segment
from EEG_preprocessing.segment_sliding_window import seg_sliding_window
from EEG_preprocessing.extract_DE_PSD_features_1per500ms import extract_de_psd_sw
from EEG_preprocessing.extract_DE_PSD_features_1per2s import extract_de_psd_raw
from EEG2Video.GLMNet.inference_glmnet import (
    inf_glmnet,
    load_glmnet_from_checkpoint,
    load_scaler,
    load_raw_stats,
)
from EEG2Video.Seq2Seq.inference_seq2seq_v2 import inf_seq2seq, load_s2s_from_checkpoint
from EEG2Video.SemanticPredictor.inference_semantic import inf_semantic_predictor, load_semantic_predictor_from_checkpoint
from EEG2Video.TuneAVideo.inference_eeg2video import inf_tuneavideo, load_tuneavideo_from_checkpoint, load_pairs
from EEG2Video.TuneAVideo.tuneavideo.util_eeg2video import save_videos_grid



def parse_args():
    parser = argparse.ArgumentParser(description="Extract a single 2‑second EEG segment.")
    # Choose the EEG segment to extract
    parser.add_argument("--subject", type=int, required=True, help="Subject number (1‑20)")
    parser.add_argument("--block", type=int, required=True, help="Block index 0‑6")
    parser.add_argument("--concept", type=int, required=True, help="Concept index 0‑39")
    parser.add_argument("--rep", type=int, required=True, help="Repetition index 0‑4")
    parser.add_argument("--eeg_root", type=str, default="./data/EEG", help="Path to EEG folder")
    
    # Define model checkpoints to use
    parser.add_argument("--glmnet_path", type=str, default="./EEG2Video/checkpoints/glmnet/sub3_fold0_best.pt", help="Path to GLMNet model checkpoint")
    parser.add_argument("--glmnet_scaler_path", type=str, default="./EEG2Video/checkpoints/glmnet/sub3_fold0_scaler.pkl", help="Path to GLMNet StandardScaler")
    parser.add_argument(
        "--glmnet_stats_path",
        type=str,
        default="./EEG2Video/checkpoints/glmnet/sub3_fold0_rawnorm.npz",
        help="Path to GLMNet raw normalization stats",
    )
    parser.add_argument("--s2s_path", type=str, default="./EEG2Video/checkpoints/seq2seq/seq2seq_v2_classic.pth", help="Path to Seq2Seq model checkpoint")
    parser.add_argument("--sempred_path", type=str, default="./EEG2Video/checkpoints/semantic/eeg2text_clip.pt", help="Path to Semantic Predictor model checkpoint")
    parser.add_argument("--sempred_scaler_path", type=str, default="./EEG2Video/checkpoints/semantic/scaler.pkl", help="Path to Semantic Predictor StandardScaler")
    parser.add_argument("--tuneavideo_path", type=str,default="./EEG2Video/checkpoints/TuneAVideo/unet_ep89.pt",help="Path to TuneAVideo model checkpoint")
    
    # TuneAVideo parameters
    parser.add_argument("--diffusion_model_path", type=str, default="./stable-diffusion-v1-4", help="Chemin vers SD-v1-4 pré-entraîné")
    parser.add_argument("--output_dir", type=str, default="./data/EEG2Video-outputs", help="Répertoire de sortie pour les GIFs")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Nombre de pas de diffusion")
    parser.add_argument("--guidance_scale", type=float, default=12.5, help="Coefficient de guidance")
    
    return parser.parse_args()

    
if __name__ == "__main__":
   
    args = parse_args()
    
    seg = extract_2s_segment(
        subject=args.subject,
        block=args.block,
        concept=args.concept,
        repetition=args.rep,
        eeg_root=args.eeg_root,
    )[None, None, None, ...]  # (1,1,1,62, 400)
    
    
    FS = 200
    DEVICE = 'cuda'
    
    features_raw, _ = extract_de_psd_raw(seg,fs=FS)
    seven_sw = seg_sliding_window(seg, win_s=0.5, step_s=0.25, fs=FS)
    features_seven_sw, _ = extract_de_psd_sw(seven_sw, fs=FS, win_sec=0.5)
    
    print("Segment shape:", seg.shape)
    print("Segment features shape", features_raw.shape)
    print("Sliding window shape:", seven_sw.shape)
    print("Sliding window features shape:", features_seven_sw.shape)
    
    # GLMNet inference
    model_glmnet = load_glmnet_from_checkpoint(args.glmnet_path, device=DEVICE)
    scaler_glmnet = load_scaler(args.glmnet_scaler_path)
    stats_glmnet = load_raw_stats(args.glmnet_stats_path)
    eeg_embeddings = inf_glmnet(
        model_glmnet,
        scaler_glmnet,
        seven_sw,
        features_seven_sw,
        stats_glmnet,
        device=DEVICE,
    )[None, None, None, ...]
    print("EEG embeddings shape:", eeg_embeddings.shape)
    
    # Initialize random video latents instead of Seq2Seq inference
    vid_latents = np.random.rand(1, 6, 4, 36, 64)
    print("Video latents shape:", vid_latents.shape)
    
    # Semantic Predictor inference
    model_semantic = load_semantic_predictor_from_checkpoint(args.sempred_path, device=DEVICE)
    scaler_semantic = load_scaler(args.sempred_scaler_path)
    sem_embeddings = inf_semantic_predictor(model_semantic, scaler_semantic, features_raw[0], device=DEVICE)
    print("Semantic embeddings shape:", sem_embeddings.shape)

    # Load BLIP caption for current sample and compute BERT embeddings
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    blip_file = os.path.join("./data/BLIP", f"block{args.block}_10min.txt")
    with open(blip_file, "r") as f:
        captions = [line.strip() for line in f if line.strip()]
    idx = args.concept * 5 + args.rep
    caption = captions[idx]
    tokens = bert_tokenizer(
        caption,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = bert_model(
            input_ids=tokens.input_ids.to(DEVICE),
            attention_mask=tokens.attention_mask.to(DEVICE)
        )
    bert_embeddings = outputs.last_hidden_state.cpu().numpy()
    print("BERT embeddings shape:", bert_embeddings.shape)
    
    # TuneAvideo inference using BERT embeddings
    video_latents, semantic_embeddings = load_pairs(vid_latents, bert_embeddings, DEVICE)
    print("Video latents shape:", video_latents.shape)
    print("Semantic embeddings shape:", semantic_embeddings.shape)

    pipe = load_tuneavideo_from_checkpoint(args.tuneavideo_path,args.diffusion_model_path, DEVICE)
    pipe.scheduler.set_timesteps(args.num_inference_steps)

    z0 = video_latents[:1]
    emb = semantic_embeddings[:1]
    videos = inf_tuneavideo(pipe, emb, z0,args.num_inference_steps, args.guidance_scale, DEVICE)
    
    os.makedirs(os.path.join(args.output_dir, f'Block{args.block}'), exist_ok=True)

    # Sauvegarde sans double rescale (déjà normalisé)
    save_videos_grid(
        videos,
        os.path.join(args.output_dir,f'Block{args.block}', f'{5*args.concept + args.rep}.gif'),
        rescale=False
    )
    print(f"[INFO] {5*args.concept + args.rep}.gif saved in {args.output_dir}/Block{args.block}")

    # Visualisation EEG, ground truth et prédiction
    eeg_seg = np.squeeze(seg)  # (62, 400)

    gt_path = os.path.join(
        "./data/Seq2Seq/Video_gifs",
        f"Block{args.block}",
        f"{5*args.concept + args.rep}.gif",
    )
    pred_path = os.path.join(
        args.output_dir,
        f"Block{args.block}",
        f"{5*args.concept + args.rep}.gif",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(eeg_seg, aspect="auto", cmap="viridis")
    axes[0].set_title("EEG segment")
    axes[0].set_xlabel("Time (samples)")
    axes[0].set_ylabel("Channel")

    if os.path.exists(gt_path):
        gt_frames = imageio.mimread(gt_path)
        axes[1].imshow(gt_frames[0])
        axes[1].set_title("Ground truth")
        axes[1].axis("off")
    else:
        axes[1].text(0.5, 0.5, "GT not found", ha="center", va="center")
        axes[1].set_axis_off()
        axes[1].set_title("Ground truth")

    pred_frames = imageio.mimread(pred_path)
    axes[2].imshow(pred_frames[0])
    axes[2].set_title("Generated")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"Block{args.block}", f"{5*args.concept + args.rep}_comparison.png"))
