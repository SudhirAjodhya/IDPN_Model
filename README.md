# Improving the Detection of Small Pulmonary Nodules in Chest X-Rays using Learnable Downsampling Techniques

This repository contains the research framework and benchmarking pipeline for evaluating learnable downsampling techniques against standard static downsamplers for small pulmonary nodule detection in Chest X-Rays (CXRs).

---

## Project Overview

In medical imaging, Chest X-Rays are high-resolution images (often 2048×2048). Training Deep Neural Networks (DNNs) directly at native resolution is computationally expensive. Standard approaches downsample CXRs via bilinear interpolation or max/average pooling, which frequently degrades subtle semantic details and edge contrast critical for detecting small nodules.

This project implements a modular, config-driven framework to systematically compare:
* **Static Baselines:** Bilinear interpolation, BlurPool, strided pooling.
* **Learnable Downsampling Modules:** Talebi resizers (paper-faithful & zero-init variants), Learned Kernel Resizers, Learned Interpolation (LIP), Weighted Pooling, and Residual Resizers.
* **Detection & Classification:** CenterNet detection heads with ConvNeXt-FPN and ResNet-FPN backbones, evaluated with FROC, CPM, and size-stratified sensitivity metrics.

---

## Repository Structure

```
Model/
├── configs/                       # Declarative YAML experiment configs
│   ├── base.yaml
│   └── experiments/
├── scripts/                       # CLI entrypoints & execution scripts
│   ├── train.py                   # Primary nodule detection training entrypoint
│   ├── train_talebi_table4.py     # Talebi Table 4 classification benchmark runner
│   ├── train_talebi_reproduce.py  # Reproduction runner for Talebi et al. (ICCV 2021)
│   ├── experiments/               # PowerShell experiment & ablation batch runners
│   │   ├── sweep.ps1              # Multi-variant parameter sweep runner
│   │   ├── run_multiseed_ablation.ps1
│   │   ├── run_talebi_table4.ps1
│   │   ├── run_frozen_backbone_control.ps1
│   │   ├── run_v2_zeroinit_and_attn.ps1
│   │   └── ...
│   ├── data/                      # Offline preprocessing, caching & mask generation
│   │   ├── cache_node21_npy.py
│   │   ├── stratify_nodules.py
│   │   └── generate_lung_masks.py
│   ├── eval/                      # Evaluation, metric aggregation & diagnostics
│   │   ├── eval.py
│   │   ├── eval_size_stratified.py
│   │   ├── analyze_results.py
│   │   ├── verify_stratification.py
│   │   └── per_nodule_analysis.py
│   └── diagnostics/               # Hardware checks, smoke tests & mechanism checks
│       ├── check_cuda.py
│       ├── smoke_test_node21_pipeline.py
│       ├── overfit_single_batch.py
│       ├── talebi_residual_diagnostic.py
│       └── view_dicom.py
├── src/                           # Core package (`idpn-model`)
│   ├── core/                      # Config loader, logging, registry, seeding
│   ├── data/                      # Datasets (JSRT, NODE21, Imagenette2), splits, transforms
│   ├── diagnostics/               # Mechanism diagnostics (residual stats, gradient flow)
│   ├── evaluation/                # FROC, CPM, size-stratified evaluation metrics
│   ├── experiments/               # Experiment helper pipelines
│   ├── losses/                    # Focal Loss (v5), MSE, QFL, TV, Auxiliary
│   ├── models/
│   │   ├── backbones/             # ConvNeXt-FPN, ResNet-FPN
│   │   ├── classifiers/           # ResizerBackbone classifier wrapper
│   │   ├── downsamplers/          # Bilinear, BlurPool, LIP, Talebi, Residual, WeightedPool
│   │   ├── heads/                 # CenterNet detection head
│   │   └── detector.py            # End-to-End Nodule Detector
│   ├── training/                  # Trainer & ClassificationTrainer classes
│   ├── utils/                     # FLOPS counter, formatting, summary generators
│   └── visualization/             # FROC curves, classification curves, overlays
├── outputs/                       # Experiment outputs (managed & git-ignored)
│   ├── mlflow/                    # Centralized MLflow SQLite database (`mlruns.db`)
│   ├── figures/                   # Exported publication plots & visualizations
│   ├── results/                   # Final summary benchmark metrics & tables
│   ├── talebi_table4/             # Reports (HTML/PDF/MD) and benchmark logs
│   └── v8/                        # Date-stamped run outputs and checkpoints
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Getting Started

### 1. Environment Setup

Create a virtual environment and install PyTorch with CUDA support:

```bash
# 1. Create virtual environment
python -m venv torch_env

# 2. Activate virtual environment
# Windows (PowerShell):
.\torch_env\Scripts\Activate.ps1
# Windows (Git Bash):
source torch_env/Scripts/activate
# Linux/macOS:
source torch_env/bin/activate

# 3. Install PyTorch with CUDA (adjust CUDA version if needed, e.g. cu128 / cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install repository dependencies
pip install -r requirements.txt

# 5. Install the local package in editable mode
pip install -e .
```

### 2. Verify GPU / Hardware

Run the CUDA diagnostic script to verify GPU detection and memory allocation:

```bash
python scripts/diagnostics/check_cuda.py
```
*(Or run `make test`)*

---

## Datasets Setup

Place datasets in `data_sets/` with the following structure:

### JSRT Dataset
Download from [Kaggle JSRT Dataset](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt):
```
data_sets/
└── JSRT/
    ├── jsrt_metadata.csv
    └── images/
        └── images/
            └── *.png
```

### NODE21 Dataset
Download from [Kaggle NODE21 Dataset](https://www.kaggle.com/datasets/pshikk/node-21-dataset-untampered):
```
data_sets/
└── node21/
    └── cxr_images/
        └── original_data/
            ├── metadata.csv
            └── images/
                └── *.mha
```

---

## Usage

### 1. Training Single Experiments

Train the nodule detection model with different downsamplers:

```bash
# Bilinear baseline at 224x224
python scripts/train.py --variant bilinear --backbone convnext_fpn --seed 42 --epochs 50

# Talebi learnable resizer (paper-faithful)
python scripts/train.py --variant node21_talebi --backbone convnext_fpn --seed 42 --epochs 50

# Zero-initialized Talebi resizer with Cosine LR schedule
python scripts/train.py --variant talebi --backbone convnext_fpn --seed 42 --cosine-lr --epochs 50
```

### 2. Running Experiment Sweeps (PowerShell)

Batch sweeps and multi-seed ablations are located in `scripts/experiments/`:

```powershell
# Run multi-seed ablation matrix (Talebi vs Bilinear across seeds 42, 7, 123)
.\scripts\experiments\run_multiseed_ablation.ps1

# Run Talebi Table 4 classification benchmarks
.\scripts\experiments\run_talebi_table4.ps1

# Run custom parameter sweep
.\scripts\experiments\sweep.ps1 -RunGroup "saliency-ablation" -Variants "bilinear,talebi,lip" -Seeds "42,7" -Epochs 50
```

### 3. Offline Preprocessing & Caching

```bash
# Fast NPY caching for NODE21 (dramatically speeds up DataLoader throughput)
python scripts/data/cache_node21_npy.py --workers 6

# Generate nodule size stratification bins
python scripts/data/stratify_nodules.py

# Generate binary lung masks with PSPNet
python scripts/data/generate_lung_masks.py
```

### 4. Evaluation & Diagnostics

```bash
# Evaluate size-stratified sensitivity across nodule size bins
python scripts/eval/eval_size_stratified.py

# Perform per-nodule sensitivity comparisons across model checkpoints
python scripts/eval/per_nodule_analysis.py

# Test residual magnitude and spatial steering diagnostics
python scripts/diagnostics/talebi_residual_diagnostic.py
```

---

## Experiment Tracking (MLflow)

All training runs log hyper-parameters, epoch metrics (loss, sensitivity, FROC), diagnostic metrics, and output checkpoints to the centralized SQLite database at `outputs/mlflow/mlruns.db`.

To launch the MLflow UI and view/compare experiment runs:

```bash
python -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlruns.db
```

Open your browser at `http://localhost:5000` to inspect:
* **Experiments:** Filter by `nodule_v8`, `talebi_table4`, etc.
* **Parameters:** Compare downsampler types, learning rates, seeds, and backbones.
* **Curves & Metrics:** Live loss curves, validation sensitivity, and FROC metrics.
