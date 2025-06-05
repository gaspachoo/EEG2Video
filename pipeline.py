from Gaspard.FullPipeline.segment_raw_signals_single import extract_2s_segment
from EEG_preprocessing.segment_sliding_window import seg_sliding_window
from EEG_preprocessing.extract_DE_PSD_features_1per500ms import extract_de_psd_sw
from EEG_preprocessing.extract_DE_PSD_features_1per2s import extract_de_psd_raw
from Gaspard.GLMNet.inference_glmnet import inf_glmnet, load_glmnet_from_checkpoint
from Gaspard.Seq2Seq.inference_seq2seq_v2 import inf_seq2seq, load_s2s_from_checkpoint
from Gaspard.SemanticPredictor.inference_semantic import inf_semantic_predictor, load_semantic_predictor_from_checkpoint

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a single 2‑second EEG segment.")
    # Choose the EEG segment to extract
    parser.add_argument("--subject", type=int, required=True, help="Subject number (1‑20)")
    parser.add_argument("--block", type=int, required=True, help="Block index 0‑6")
    parser.add_argument("--concept", type=int, required=True, help="Concept index 0‑39")
    parser.add_argument("--rep", type=int, required=True, help="Repetition index 0‑4")
    parser.add_argument("--eeg_root", type=str, default="./data/EEG", help="Path to EEG folder")
    
    # Define models to use
    parser.add_argument("--glmnet_path", type=str, default="./Gaspard/checkpoints/glmnet/sub3_fold0_best.pt", help="Path to GLMNet model checkpoint")
    parser.add_argument("--s2s_path", type=str, default="./Gaspard/checkpoints/seq2seq/seq2seq_v2_classic.pth", help="Path to Seq2Seq model checkpoint")
    parser.add_argument("--sempred_path", type=str, default="./Gaspard/checkpoints/semantic/eeg2text_clip.pt", help="Path to Semantic Predictor model checkpoint")
    
    args = parser.parse_args()

    seg = extract_2s_segment(
        subject=args.subject,
        block=args.block,
        concept=args.concept,
        repetition=args.rep,
        eeg_root=args.eeg_root,
    )
    
    FS = 200
    DEVICE = 'cuda'
    
    features_raw, _ = extract_de_psd_raw(seg,fs=FS)
    seven_sw = seg_sliding_window(seg, win_s=0.5, step_s=0.25, fs=FS)
    features_seven_sw, _ = extract_de_psd_sw(seven_sw, fs=FS, win_sec=0.5)
    
    
    print("Segment shape:", seg.shape)
    print("Segment features shape", features_raw.shape)
    print("Sliding window shape:", seven_sw.shape)
    print("Sliding window features shape:", features_seven_sw.shape)
    
    
    model_glmnet = load_glmnet_from_checkpoint(args.glmnet_path, device=DEVICE)
    eeg_embeddings = inf_glmnet(model_glmnet, seven_sw, features_seven_sw, device=DEVICE)[None, None, None, ...]
    print("EEG embeddings shape:", eeg_embeddings.shape)
    
    model_s2s = load_s2s_from_checkpoint(args.s2s_path, device=DEVICE)
    vid_latents = inf_seq2seq(model_s2s, eeg_embeddings, device=DEVICE)
    print("Video latents shape:", vid_latents.shape)
    
    model_semantic = load_semantic_predictor_from_checkpoint(args.sempred_path, device=DEVICE)
    sem_embeddings = inf_semantic_predictor(model_semantic, features_raw[0], device=DEVICE)
    print("Semantic embeddings shape:", sem_embeddings.shape)

    