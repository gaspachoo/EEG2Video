# EEG2Video

This project aims to convert EEG (electroencephalographic) signals into video sequences using deep learning models, including Transformer-based models, 3D UNets, and diffusion pipelines.

## 📁 Project Structure

```
EEG2Video/
│
├── EEG-VP/                         # EEG classification module
│   ├── EEG_VP_train_test.py       # EEG model training and testing
│   └── models.py                  # Neural network architectures for EEG
│
├── EEG2Video/                     # Main EEG-to-Video model components
│   ├── 40_class_run_metrics.py   # Evaluation metrics (MSE, SSIM, CLIP, etc.)
│   ├── inference_eeg2video.py    # Inference script to generate video from EEG
│   ├── train_finetune_videodiffusion.py # Pipeline fine-tuning entry point
│   ├── models/                   # Models for video generation
│   │   ├── attention.py
│   │   ├── models.py
│   │   ├── resnet.py
│   │   ├── train_semantic_predictor.py
│   │   ├── unet.py
│   │   └── unet_blocks.py
│   └── pipelines/
│       ├── pipeline_tuneavideo.py
│       └── pipeline_tuneeeg2video.py
│
├── EEG_preprocessing/            # EEG feature extraction scripts
│   ├── DE_PSD.py
│   ├── extract_DE_PSD_features_1per1s.py
│   ├── extract_DE_PSD_features_1per2s.py
│   └── segment_raw_signals_200Hz.py
│
└── project/                      # Data manipulation utilities
    ├── import.py
    └── segment_data.py
```

## 🔍 Key Components

### EEG-VP
- `EEG_VP_train_test.py`: Data loading, accuracy metrics, training for EEG models.
- `models.py`: Contains architectures like `shallownet`, `eegnet`, `conformer`, `glfnet`, `mlpnet`, etc.

### EEG2Video
- `train_finetune_videodiffusion.py`: Training entry point for the video generation pipeline.
- `inference_eeg2video.py`: Generates video sequences from EEG inputs.
- `models/`:
  - `attention.py`: 3D transformer blocks.
  - `unet.py` & `unet_blocks.py`: 3D conditional UNet with cross-attention.
  - `train_semantic_predictor.py`: CLIP-based semantic predictor module.
- `pipelines/`:
  - `pipeline_tuneavideo.py`: Video generation pipeline.
  - `pipeline_tuneeeg2video.py`: EEG-adapted video generation pipeline.

### EEG_preprocessing
- `DE_PSD.py`: DE/PSD feature extraction.
- `extract_DE_PSD_features_1per1s.py`, `1per2s.py`: Feature extraction over time windows.
- `segment_raw_signals_200Hz.py`: Segments raw EEG signals into time windows.

### project
- `import.py`: Load and visualize EEG blocks.
- `segment_data.py`: Custom segmentation of EEG data for training.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/EEG2Video.git
   cd EEG2Video
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. Run training or inference:
   ```bash
   python EEG2Video/train_finetune_videodiffusion.py
   ```

## 📊 Evaluation

Use `40_class_run_metrics.py` to compute:
- Top-k accuracy
- CLIP Score
- MSE / SSIM between generated and target videos

## 📌 Notes

- This project relies on PyTorch, diffusers, Transformers, and related libraries.
- Some model definitions appear multiple times (`models.py`), which could be refactored for clarity.

