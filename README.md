# EEG2Video Pipeline 
Acronyms used:
-   EEG = ElectroEncephaloGram
-   DE = Differential Entropy
-   PSD = Power Spectral Density

## 1. Data Preprocessing  ($\approx$ 1 week)
### 1. EEG data segmentation:

**Purpose** : Segment raw  8min 40s EEG recordings into shorter recordings 

- First step :

    Segment (62 channels, 200Hz, 8min40s)  into 2s windows (per video : 40 concepts, 5 repetitions per concept, remove hint sequences)

    Shapes are : (block, concept, repetition, channel, time).

    Script : `EEG_preprocessing/segment_raw_signals_200Hz.py`
    (also exposes `extract_2s_segment` to grab a single 2‑second window on the fly)

- Second step :
    
    Segment each 2s windows into 500ms windows using a 250 ms overlap (=7 windows)

    Shapes adapted to (block, concept, repetition, window, channel, time).

    Script : `EEG_preprocessing/segment_sliding_window.py`

### 2. DE/PSD Features detection on 5 frequency bands:

**Purpose**: detect DE and PSD features on segmented EEGs, on 5 different frequency bands:

- Delta (1-4 Hz)
- Theta (4-8 Hz)
- Alpha (8-14 Hz)
- Beta (14-31 Hz)
- Gamma (31-99 Hz)

We detect on 3 types of windows :

- Detect features on 2s raw windows.

    Shapes adapted to (block, concept, repetition, channel, band).

    Script : `EEG_preprocessing/extract_DE_PSD_features_1per2s.py`

- Detect features on 1s windows without overlapping.

    Shapes adapted to (block, concept, repetition, window, channel, band).

    Script : `EEG_preprocessing/extract_DE_PSD_features_1per1s.py`

- Detect features on 1s windows without overlapping.

    Shapes adapted to (block, concept, repetition, window, channel, band).

    Script : `EEG_preprocessing/extract_DE_PSD_features_1per500ms.py`

### 3. Video alignment and GIF creation:

**Purpose** : Segment raw  8min 40s Video recordings into shorter downscaled GIFs

- Segment 8 min 40s videos into 200 (40 concepts × 5 repetitions) 2s clips  using cv2.

    Downscale to (512, 288), extract 6 frames (3 FPS) and save to .gif format using imageio.

    Script: `EEG2Video/extract_gif.py`




## 2. EEG Feature Encoding ($\approx$ 2 weeks)

### Purpose :

We use a GLMNet which uses a ShallowNet on raw EEGs, and a MLP on DE/PSD features to extract features from EEGs.

One layer in Shallownet is modified compared to original : AvgPool2d -> AdaptiveAvgPool2d

Models path : `Gaspard/GLMNet/models.py` 

### Training :

We use 2s raw EEGs and 1s windows for DE/PSD features.

Script : `Gaspard/GLMNet/train_glmnet.py`

### Inference :
    
We generate EEgs embeddings from train GLMNet.

We use 2s raw EEGs and 500ms windows for DE/PSD features.

Script : `Gaspard/GLMNet/inference_glmnet.py`




## 3. Seq2Seq Transformer ($\approx$ 1 week)

### Purpose :

We align EEG embeddings with videos

The model is just a rewriting of original model : `Gaspard/Seq2Seq/models/transformer.py`

### Training :

#### 1. Generate latents from pretrained VAE:

- A pre-trained VAE is used to convert 6-frame video GIFs (shape [n_frames, sample_f, height, width] = [6, 3, 288, 512]) into latent tensors [n_frames, d1, d2, d3] = [6, 4, 36, 64] where d1, d2, d3 are due to VAE model.

    Script : `Gaspard/Seq2Seq/generate_video_latents.py`

#### 2. Use Seq2Seq model to align EEGs embeddings and video latents

- We use the generated EEG embeddings from part  as source(shape : [batch, 7, 512]) and the generated latents from part 3.1 as source (shape : [batch, 6, 9216]) to train the model.

    Script : `Gaspard/Seq2Seq/train_seq2seq.py`

 ### Inference : 

We use the generated EEG embeddings from part 2 to generate predicted latents.

Script : `Gaspard/Seq2Seq/inference_seq2seq_v2.py`





## 4. Semantic Predictor ($\approx$ 2 days)

### Purpose :
We use a semantic predictor to align EEG features with BLIP captions

Model path : `Gaspard/SemanticPredictor/models.py`

### Training :

#### 1. Generate text embeddings

- We process the BLIP captions into pretrained CLIP model to generate text embeddings.

    Script : `Gaspard/SemanticPredictor/generate_text_emb.py`

#### 2. Align EEG features with BLIP captions

    
- We use the DE/PSD features of part 2 as source and the text embeddings from part 4.1 as target.

    Script : `Gaspard/SemanticPredictor/train_semantic.py`

### Inference : 

- We used the generated text embeddings from part 4.1 to generate semantic embeddings.

    Script : `Gaspard/SemanticPredictor/inference_semantic.py`




## 5. TuneAVideo pipeline ($\approx$ 3 weeks)

### Purpose : We use the TuneAVideo pipeline to improve the quality of the video latents from part 3 in adding context thanks to semantic predictor.

### Training :

- We use the predicted latents of part 3.2 as source and semantic embeddings of part 4.2 as a target to finetune the TuneAVideo pipeline.

    Script : `Gaspard/TuneAVideo/train_tuneavideo_v7.py --mixed_precision --batch_size 7 --use_xformers --num_workers 2 --pin_memory --use_empty_cache --use_channels_last`

### Inference :
    
- We generate precise video latents, and respective videos from predicted latents of part 3.2.

    Script : `Gaspard/TuneAVideo/inference_tuneavideo.py`


# Further work
Investigate on the reason why generated videos lack of contrast.