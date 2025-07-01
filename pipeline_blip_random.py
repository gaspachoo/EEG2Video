import os
import argparse
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL
import imageio
import matplotlib.pyplot as plt
from EEG_preprocessing.segment_raw_signals_200Hz import extract_2s_segment
from EEG_preprocessing.segment_sliding_window import seg_sliding_window
from EEG_preprocessing.extract_DE_PSD_features_1per500ms import extract_de_psd_sw
from EEG_preprocessing.extract_DE_PSD_features_1per2s import extract_de_psd_raw

from EEG2Video.TuneAVideo.tuneavideo.models.unet import UNet3DConditionModel
from EEG2Video.TuneAVideo.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from EEG2Video.TuneAVideo.tuneavideo.util_eeg2video import save_videos_grid


def load_tuneavideo_with_bert(ckpt_path: str, diffusion_model_path: str, device: torch.device,
                              bert_embeddings: torch.Tensor) -> TuneAVideoPipeline:
    """Load TuneAVideo pipeline and replace text encoding with pre-computed BERT embeddings."""

    vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device).half().eval()
    tokenizer = CLIPTokenizer.from_pretrained(diffusion_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(diffusion_model_path, subfolder="text_encoder").to(device).half().eval()
    unet = UNet3DConditionModel.from_pretrained_2d(ckpt_path).to(device).half().eval()
    scheduler = DDIMScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")

    pipe = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    pipe.custom_embeddings = bert_embeddings
    pipe.custom_negative = bert_embeddings.mean(dim=1, keepdim=True).expand(-1, bert_embeddings.size(1), -1)

    def _encode_prompt_override(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        embeds = self.custom_embeddings.to(device)
        embeds = embeds.repeat(1, num_videos_per_prompt, 1)
        embeds = embeds.view(embeds.shape[0] * num_videos_per_prompt, embeds.shape[1], embeds.shape[2])

        if do_classifier_free_guidance:
            uncond = self.custom_negative.to(device)
            uncond = uncond.repeat(1, num_videos_per_prompt, 1)
            uncond = uncond.view(embeds.shape[0], embeds.shape[1], embeds.shape[2])
            embeds = torch.cat([uncond, embeds])
        return embeds

    pipe._encode_prompt = _encode_prompt_override.__get__(pipe, TuneAVideoPipeline)
    return pipe



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
    parser.add_argument("--tuneavideo_path", type=str,default="./stable-diffusion-v1-4/unet",help="Path to TuneAVideo model checkpoint")
    
    # TuneAVideo parameters
    parser.add_argument("--diffusion_model_path", type=str, default="./stable-diffusion-v1-4", help="Chemin vers SD-v1-4 pré-entraîné")
    parser.add_argument("--output_dir", type=str, default="./data/EEG2Video-outputs", help="Répertoire de sortie pour les GIFs")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Nombre de pas de diffusion")
    parser.add_argument("--guidance_scale", type=float, default=12.5, help="Coefficient de guidance")
    
    return parser.parse_args()

    
if __name__ == "__main__":
    
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize random video latents instead of Seq2Seq inference
    vid_latents = np.random.randn(1, 4, 6, 36, 64).astype(np.float32)
    print("Video latents shape:", vid_latents.shape)

    # Load BLIP caption for current sample and compute BERT embeddings
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    name_suffix_list = ["st", "nd", "rd", "th", "th", "th", "th"]
    blip_file = os.path.join("./data/BLIP", f"{args.block}{name_suffix_list[args.block-1]}_10min.txt")
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
    bert_embeddings = outputs.last_hidden_state.to(DEVICE).half()
    print("BERT embeddings shape:", bert_embeddings.shape)

    video_latents = torch.from_numpy(vid_latents).to(DEVICE).half()

    pipe = load_tuneavideo_with_bert(
        args.tuneavideo_path,
        args.diffusion_model_path,
        DEVICE,
        bert_embeddings,
    )
    pipe.scheduler.set_timesteps(args.num_inference_steps)

    videos = pipe(
        "",
        video_length=6,
        height=288,
        width=512,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        latents=video_latents,
    ).videos
    
    os.makedirs(os.path.join(args.output_dir, f'Block{args.block}'), exist_ok=True)

    # Sauvegarde sans double rescale (déjà normalisé)
    save_videos_grid(
        videos,
        os.path.join(args.output_dir,f'Block{args.block}', f'{5*args.concept + args.rep}.gif'),
        rescale=False
    )
    print(f"[INFO] {5*args.concept + args.rep}.gif saved in {args.output_dir}/Block{args.block}")

    pred_path = os.path.join(
        args.output_dir,
        f"Block{args.block}",
        f"{5*args.concept + args.rep}.gif",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pred_frames = imageio.mimread(pred_path)
    axes[2].imshow(pred_frames[0])
    axes[2].set_title("Generated")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"Block{args.block}", f"{5*args.concept + args.rep}_comparison.png"))
