# EEG2Video Pipeline 
## ✅ Modules Completed
### 1. Data Preprocessing
#### EEG data segmentation:

Raw EEG (62 channels, 200Hz, 8min40s) segmented into 2s windows (per video : 40 concepts, 5 repetitions per concept, remove hint sequences)

Shapes are : (block, concept, repetition, channel, time).

Script : `EEG_preprocessing/segment_raw_signals_200Hz.py`

Then, segment each 2s windows into 500ms windows using a 250 ms overlap (=7 windows)

Shapes adapted to (block, concept, repetition, window, channel, time).

Script : `EEG_preprocessing/segment_sliding_window.py`

#### DE/PSD Features detection on 5 frequency bands:
- Detect features on 1s windows without overlapping.

    Shapes adapted to (block, concept, repetition, window, channel, band).

    Script : `EEG_preprocessing/extract_DE_PSD_features_1per1s.py`

- Detect features on 1s windows without overlapping.

    Shapes adapted to (block, concept, repetition, window, channel, band).

    Script : `EEG_preprocessing/extract_DE_PSD_features_1per500ms.py`

#### Video alignment and GIF creation:

Segment 8min40s videos into 2s sliced into 200 2s clips (40 concepts × 5 repetitions) using cv2.

Downscale to (512, 288), extract 6 frames (3 FPS) and save to .gif format using imageio.

Script: `EEG2Video/extract_gif.py`


### 2. EEG Feature Encoding
We use a GLMNet which uses a ShallowNet on raw EEGs, and a MLP on DE/PSD features to extract features from EEGs.

One layer in Shallownet is modified compared to original : AvgPool2d -> AdaptiveAvgPool2d

Models path : `Gaspard_model/models/models.py` 

- Training :  We use 2s raw EEGs and 1s windows for DE/PSD features.

    Script : `Gaspard_model/train_glmnet.py`

- Inference : We generate EEgs embeddings from train GLMNet.

    We use 2s raw EEGs and 500ms windows for DE/PSD features.

    Script : `Gaspard_model/generate_eeg_emb_sw.py`

### 3. Align Video Latents with EEG embeddings
#### Generate latents from pretrained VAE:

A pre-trained VAE is used to convert 6-frame video GIFs (shape [n_frames, sample_f, height, width] = [6, 3, 288, 512]) into latent tensors [n_frames, d1, d2, d3] = [6, 4, 36, 64] where d1, d2, d3 are due to VAE model.

Script : `Gaspard_model/generate_latents_vae.py`

#### Use Seq2Seq model to align EEGs embeddings and video latents

The model 
- Training :

4. Transformer Seq2Seq
Seq2Seq Transformer implementation:

EEG sequences of shape [batch, 7, 512] used as src.

Video latents of shape [batch, 6, 9216] used as tgt.

Uses 2-layer encoder, 4-layer decoder.

Teacher forcing + autoregressive generation implemented in loop.

PositionalEncoding + Masking handled.

Training script complete:

Optimizer + Cosine LR scheduler.

Model outputs saved and evaluated with MSE.

5. Evaluation pipeline
Inference code for test block (block 7).

Outputs latent_out_block7_40_classes.npy, usable for diffusion conditioning.

🟡 Currently Ongoing
1. VAE Training on Custom Resolution
If using non-square resolution (e.g., 288×512), diffusers>=0.29 with updated AutoencoderKL(sample_size=(288,512)) is needed.

Training with small batch size (1–4) required due to mid-block attention memory cost.

Done in train_vae.py.

2. Environment stabilization (Enroot container)
CUDA 12.8 runtime container deployed using enroot.

Compatible stack:

PyTorch 2.2.2 (cu121)

diffusers 0.29+

accelerate 0.27+

decord 0.6.0

Finalizing conda/pip setup inside container for reproducibility.

🔴 Modules Yet To Be Implemented
1. Dynamic Attention & Semantic Conditioning
EEG2Video (NeurIPS 2024) proposes:

Semantic Predictor: predicts ê_t (textual embedding) from EEG.

Dynamic Predictor: estimates fast/slow motion to guide diffusion.

These are not yet included in the current codebase.

2. Video Generation via Diffusion
Conditioning diffusion model (e.g., UNet3DConditionModel) using:

EEG latents (ẑ₀) as input.

Semantic embeddings and dynamic motion as additional guidance.

Not yet implemented or integrated.

Will require:

Loading a pre-trained diffusion model (e.g., Tune-A-Video or similar).

Sampling with a scheduler (e.g., DDIM or PNDM).

3. Full Evaluation & Reconstruction
Generate actual .mp4 or .gif video outputs from predicted latents.

Compare reconstructed videos (FID, CLIP similarity, etc.).

Possibly evaluate semantic alignment (e.g., CLIPScore) between original and reconstructed clips.

🗂️ Optional Future Enhancements
Multimodal conditioning: combining EEG + text during generation.

Cross-subject generalization: model trained across participants.

Online visualization interface: demo page or notebook showing EEG → video generation.

📌 Current Commitment Summary
Module	Status
EEG segmentation + normalization	✅ Done
GIF slicing & alignment	✅ Done
EEG encoder (MyEEGNet)	✅ Done
VAE encoder training	🟡 Ongoing
Seq2Seq Transformer	✅ Done
Latent prediction	✅ Done
Environment setup (CUDA 12)	🟡 Ongoing
Diffusion generation	🔴 Not started
Semantic/dynamic predictors	🔴 Not started

