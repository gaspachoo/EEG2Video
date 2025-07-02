# EEG2Video Pipeline
Acronyms used:
- EEG = ElectroEncephaloGram
- DE = Differential Entropy
- PSD = Power Spectral Density

## 1. Data Preprocessing (~1 week)
### 1. EEG data segmentation:

**Purpose**: Segment raw 8min 40s EEG recordings into shorter recordings

- First step:

  Segment (62 channels, 200Hz, 8min40s) into 2s windows (per video: 40 concepts, 5 repetitions per concept, remove hint sequences)

  Shapes are: (block, concept, repetition, channel, time).

  Script: `EEG_preprocessing/segment_raw_signals_200Hz.py`
  (also exposes `extract_2s_segment` to grab a single 2-second window on the fly)

- Second step:

  Segment each 2s window into 500ms windows using a 250 ms overlap (=7 windows)

  Shapes adapted to (block, concept, repetition, window, channel, time).

  Script: `EEG_preprocessing/segment_sliding_window.py`

### 2. DE/PSD Features detection on 5 frequency bands:

**Purpose**: detect DE and PSD features on segmented EEGs, on 5 different frequency bands:

- Delta (1-4 Hz)
- Theta (4-8 Hz)
- Alpha (8-14 Hz)
- Beta (14-31 Hz)
- Gamma (31-99 Hz)

We detect on 3 types of windows:

- Detect features on 2s raw windows.

  Shapes adapted to (block, concept, repetition, channel, band).

  Script: `EEG_preprocessing/extract_DE_PSD_features_1per2s.py`

- Detect features on 500ms windows with a 250 ms overlap.

  Shapes adapted to (block, concept, repetition, window, channel, band).

  Script: `EEG_preprocessing/extract_DE_PSD_features_1per500ms.py`

### 3. Video alignment and GIF creation:

This repository does not include the script for extracting GIFs. Videos should be segmented externally into 2s GIFs before using the downstream models.

## 2. EEG Feature Encoding (~2 weeks)
### Purpose:
We use a GLMNet which uses a ShallowNet on raw EEGs, and a MLP on DE/PSD features to extract features from EEGs.
One layer in ShallowNet is modified compared to the original: AvgPool2d -> AdaptiveAvgPool2d
Models path: `EEGtoVideo/GLMNet/modules/models_paper.py`

### Training:
Training uses 2s raw EEGs split into 500 ms sliding windows for DE/PSD features.
The pre-segmented data is stored in `Segmented_500ms_sw` and `DE_500ms_sw`.
Script: `EEGtoVideo/GLMNet/train_glmnet.py`
An alternative, lighter model relying only on spectral features can be trained with `EEGtoVideo/GLMNet/train_glfnet_mlp.py`.
- Raw EEGs are normalized per channel using the training split statistics.
- Both trainers accept a `--scheduler` argument (`steplr`, `reducelronplateau`, `cosine`) and a `--min_lr` value to set the learning rate floor.
- The best checkpoint is saved as `<subject>_<category>_best.pt` and the ShallowNet weights are also stored in `<subject>_<category>_shallownet.pt`.

### Inference:
We generate EEG embeddings from a trained GLMNet.
## 3. Training pipeline
The project includes three sequential passes:
- **P0**: pre-training the GLMNet (run `make p0`).
- **P1**: training the Transformer while the VAE and the diffusion model stay frozen (run `make p1`).
- **P2**: end-to-end fine tuning with a learning rate of 1e-5 for one or two epochs (run `make p2`).
You can pass extra options to each step with `ARGS`.

We use 2s raw EEGs and 500ms windows for DE/PSD features.
The same normalization parameters are loaded to preprocess raw EEGs at inference time.
Script: `EEGtoVideo/GLMNet/inference_glmnet.py`
Embeddings can also be produced with the features-only model using `EEGtoVideo/GLMNet/inference_glfnet_mlp.py`.

## 3. Video Latent Extraction
Video clips are converted into 2-second GIFs and encoded with the VAE from Stable Diffusion. Each clip becomes a latent tensor stored alongside the EEG embeddings.

## 4. Pair Creation
EEG embeddings and video latents share the same block/concept/repetition identifiers. We serialize these pairs in `npz` archives to feed them into the Transformer.

## 5. Transformer Training
The Transformer takes EEG embeddings as input and predicts the corresponding video latent. Training scripts build upon the `EEGtoVideo/GLMNet` utilities and stream the paired data.

## 6. Video Generation with Diffusion
At inference time, the Transformer predicts a latent from unseen EEG signals. This latent is then decoded by Stable Diffusion to produce the final video.

# Further work
Investigate on the reason why generated videos lack of contrast.

## License
EEG2Video is released under the MIT License. See the [LICENSE](LICENSE) file for details.
