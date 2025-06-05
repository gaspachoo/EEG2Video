from Gaspard.FullPipeline.segment_raw_signals_single import extract_2s_segment
from EEG_preprocessing.segment_sliding_window import seg_sliding_window
from EEG_preprocessing.extract_DE_PSD_features_1per500ms import extract_de_psd_sw
from Gaspard.GLMNet.inference_glmnet import inf_glmnet, load_glmnet_from_checkpoint
from Gaspard.Seq2Seq.inference_seq2seq_v2 import inf_seq2seq, load_s2s_model

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
    
    args = parser.parse_args()

    seg = extract_2s_segment(
        subject=args.subject,
        block=args.block,
        concept=args.concept,
        repetition=args.rep,
        eeg_root=args.eeg_root,
    )
    
    seven_sw = seg_sliding_window(seg, win_s=0.5, step_s=0.25, fs=200)
    features_seven_sw, _ = extract_de_psd_sw(seven_sw, fs=200, win_sec=0.5)
    
    print("Segment shape:", seg.shape)
    print("Sliding window shape:", seven_sw.shape)
    print("Features shape:", features_seven_sw.shape)
    
    model_glmnet = load_glmnet_from_checkpoint(args.glmnet_path, device='cuda')
    eeg_embeddings = inf_glmnet(model_glmnet, seven_sw, features_seven_sw, device='cuda')[None, None, None, ...]
    print("EEG embeddings shape:", eeg_embeddings.shape)
    
    model_s2s = load_s2s_model(args.s2s_path, device='cuda')
    eeg_embeddings = inf_glmnet(model_s2s, seven_sw, features_seven_sw, device='cuda')[None, None, None, ...]
    print("Video latents shape:", eeg_embeddings.shape)
    