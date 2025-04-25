# EEG2Video

Transform electroencephalographic (EEG) activity into coherent video sequences using state‑of‑the‑art deep‑learning techniques: Conformers, Transformer‑encoders, 3‑D UNets with cross‑attention, and diffusion generative models. The repository also contains classical EEG‑classification baselines and extensive preprocessing utilities.

---

## 📑 Table of Contents

1. [Project Goals](#project-goals)
2. [Directory Layout](#directory-layout)
3. [Main Workflows](#main-workflows)
4. [Key APIs & Scripts](#key-apis--scripts)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Evaluation](#evaluation)
8. [Notes & Roadmap](#notes--roadmap)

---

## Project Goals

* **EEG‑to‑Video** – learn a mapping from multichannel EEG sequences to short video clips.
* **EEG‑VP baselines** – benchmark shallow/deep CNNs, Conformers and MLPs on motor‑imagery classification.
* **Modular Pipeline** – decouple preprocessing, feature engineering, model training, and evaluation for reproducibility.

---

## Directory Layout

```text
EEG2Video/
│
├── analyse_tree.py                # Static‑analysis helper (lists functions/classes)
│
├── EEG-VP/                        # EEG classification baselines
│   ├── EEG_VP_train_test.py       # Training / test loop & helpers
│   └── models.py                  # ShallowNet, DeepNet, EEGNet, Conformer, …
│
├── Gaspard_preprocess/            # Personal preprocessing utilities
│   ├── import.py                  # Load & plot raw blocks
│   ├── yaml_gen.py                # YAML metadata generator
│   ├── process_video.py           # Extract 2 s clips, down‑sample videos
│   └── plot_data.py               # Quick visual checks
│
├── Gaspard_model/                 # Training scripts & custom models
│   ├── train_glmnet_cv.py         # Cross‑validated GLMNet trainer
│   ├── train_model_comparison.py  # ShallowNet vs Deep baselines
│   ├── train_shallownet_{cv,paper}.py # ShallowNet experiments
│   ├── train_mlp_cv.py            # MLP on PSD/DE features
│   └── models/                    # Encoders, Transformers, etc.
│       ├── encoders.py            # CLIP, GLMNetEncoder, MLPEncoder, …
│       ├── transformers.py        # EEG2VideoTransformer
│       └── models.py              # Video & EEG backbones (shared)
│
├── EEG_preprocessing/             # Signal segmentation & feature extraction
│   ├── segment_raw_signals_200Hz.py
│   ├── segment_sliding_window.py
│   ├── DE_PSD.py                  # Differential Entropy & Power Spectral Density
│   ├── extract_DE_PSD_features_{1per1s,1per2s}.py
│   └── gen_features_from_sw_data.py
│
├── EEG2Video/                     # Core EEG‑to‑Video pipeline
│   ├── inference_eeg2video.py     # Zero‑shot / fine‑tuned inference
│   ├── train_finetune_videodiffusion.py
│   ├── 40_class_run_metrics.py    # MSE, SSIM, CLIP, Top‑k metrics
│   ├── models/                    # Diffusers‑style latent models
│   ├── pipelines/                 # Tune‑a‑Video and EEG‑conditioned versions
│   └── models/… (resnet, unet, attention, …)
│
└── EEG2Video_New/                 # Experimental v2 pipeline (modularised)
    └── … (mirrors the structure above)
```

> **ℹ︎ Tip:** duplicate model definitions in `models/models.py` are kept for backward compatibility and will be consolidated in a future refactor.

---

## Main Workflows

| Stage | Script / Entry‑point | Description |
|-------|----------------------|-------------|
| **1. Pre‑processing** | `EEG_preprocessing/segment_raw_signals_200Hz.py`<br>`EEG_preprocessing/extract_DE_PSD_features_*.py` | Slice raw `.npy` recordings into windows (200 Hz) and compute DE/PSD features. |
| **2. Feature Engineering** | `EEG_preprocessing/gen_features_from_sw_data.py` | Aggregate sliding‑window features for downstream tasks. |
| **3. EEG Baselines** | `EEG-VP/EEG_VP_train_test.py` | Train ShallowNet / EEGNet / Conformer baselines on classification. |
| **4. GLMNet & MLP** | `Gaspard_model/train_glmnet_cv.py`<br>`Gaspard_model/train_mlp_cv.py` | Cross‑validated training on spectral features. |
| **5. EEG‑to‑Video** | `EEG2Video/train_finetune_videodiffusion.py` | Fine‑tune latent‑diffusion pipeline conditioned on EEG embeddings. |
| **6. Inference** | `EEG2Video/inference_eeg2video.py` | Generate video clips from unseen EEG segments. |
| **7. Evaluation** | `EEG2Video/40_class_run_metrics.py` | Compute clip/video accuracy, CLIP Score, MSE, SSIM, PSNR, etc. |

---

## Key APIs & Scripts

Below is a non‑exhaustive registry of public classes & utilities (auto‑generated via `analyse_tree.py`). Use it as a quick reference when importing:

### Core Helpers

- **`analyse_tree.py`** – `list_functions_and_classes`, `scan_project`
- **`Gaspard_preprocess/import.py`** – `load_all_eeg_data_by_subject`, `plot_eeg_block`

### Representative Models

| Path | Classes |
|------|---------|
| `Gaspard_model/models/encoders.py` | `CLIP`, `GLMNetEncoder`, `MLPEncoder`, `ShallowNetEncoder`, `MLPEncoder_feat` |
| `Gaspard_model/models/transformers.py` | `EEG2VideoTransformer` |
| `EEG2Video/models/unet.py` | `UNet3DConditionModel`, `UNet3DConditionOutput` |
| `EEG2Video/models/DANA_module.py` | `Diffusion` |

*(Expand the full list with `analyse_tree.py` when developing new components.)*

---

## Installation

```bash
# 1. Clone
$ git clone https://github.com/your‑username/EEG2Video.git
$ cd EEG2Video

# 2. Environment
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate
$ pip install -r requirements.txt
```

CUDA 11.8 + PyTorch 2.2 are recommended for 3‑D diffusion training.

---

## Quick Start

```bash
# Finetune diffusion on preprocessed EEG
python EEG2Video/train_finetune_videodiffusion.py \
       --config configs/finetune.yaml

# Generate video from a saved EEG feature file
python EEG2Video/inference_eeg2video.py \
       --eeg ./samples/example.npy --output ./out/
```

---

## Evaluation

Run the comprehensive metrics suite:

```bash
python EEG2Video/40_class_run_metrics.py \
       --pred_dir ./out/ --gt_dir ./ground_truth/ \
       --metrics clip mse ssim topk
```

Outputs include per‑video JSON logs and an aggregated CSV summary.

---

## Notes & Roadmap

- [ ] **Model consolidation** – unify duplicate `models/models.py` across sub‑packages.
- [ ] **Lightning migration** – port training scripts to PyTorch Lightning for cleaner checkpoints.
- [ ] **Web demo** – stream generated clips via Gradio.

Contributions via pull requests or issues are welcome! Feel free to open a discussion for feature requests or questions.

