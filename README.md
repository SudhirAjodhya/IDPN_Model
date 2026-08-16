# Improving the Detection of Small Pulmonary Nodules in Chest X-Rays using Learnable Downsampling Techniques

This repository contains the research framework and benchmarking pipeline for evaluating learnable downsampling techniques against standard static downsamplers for small pulmonary nodule detection in Chest X-Rays (CXRs).

---

## Project Overview

In medical imaging, Chest X-Rays are high-resolution images (often 2048x2048). Training Deep Neural Networks (DNNs) directly at native resolution is computationally expensive. Standard approaches downsample CXRs via bilinear interpolation or max/average pooling, which frequently degrades subtle semantic details and edge contrast critical for detecting small nodules.

This project implements a modular, config-driven framework to systematically compare:
* **Static Baselines:** Bilinear interpolation, BlurPool, strided pooling.
* **Learnable Downsampling Modules:** Talebi resizers (paper-faithful & zero-init variants), Learned Kernel Resizers, Learned Interpolation (LIP), Weighted Pooling, and Residual Resizers.
* **Detection & Classification:** CenterNet detection heads with ConvNeXt-FPN and ResNet-FPN backbones, evaluated with FROC, CPM, and size-stratified sensitivity metrics.

---

## Repository Structure

```bash
Model/
├── configs/                       # Declarative YAML experiment configs
│   ├── base.yaml
│   └── experiments/               # One YAML per experiment/ablation
├── scripts/                       # Thin CLI entrypoints & execution scripts
│   ├── train.py                   # Primary nodule detection training entrypoint
│   ├── train_talebi_table4.py     # Talebi Table 4 classification benchmark runner
│   ├── train_talebi_reproduce.py  # Reproduction runner for Talebi et al. (ICCV 2021)
│   ├── experiments/               # PowerShell experiment & ablation batch runners
│   │   ├── sweep.ps1              # Multi-variant parameter sweep runner
│   │   ├── run_multiseed_ablation.ps1
│   │   ├── run_talebi_table4.ps1
│   │   ├── run_frozen_backbone_control.ps1
│   │   ├── run_v2_zeroinit.ps1
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

## Architectural Deep Dive

The framework is organized into decoupled, modular components registered dynamically through decorators:

```
Input CXR (B, 1, 2048, 2048)
            │
            ▼
┌───────────────────────────────────────┐
│     Learnable Downsampler (LDM)       │   e.g. TalebiZeroInit / LIP / Bilinear
│  (B, 1, 2048, 2048) -> (B, 1, 224, 224)│
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│     Feature Backbone + FPN Neck       │   e.g. ConvNeXt-Tiny + FPN / ResNet50-FPN
│   Multi-scale feature pyramid maps    │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│       CenterNet Detection Head        │   Outputs:
│   - Nodule Heatmap (B, 1, 56, 56)     │   - Peak locations (subtlety-weighted)
│   - Offsets & Sizes (B, 2, 56, 56)    │   - Continuous bounding box coordinates
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│     Loss Formulation & Evaluation     │   - Focal Loss v5 (Gaussian-filtered peaks)
│   - FROC / CPM / Size Stratification  │   - Regression L1 Loss + Optional TV Regularization
└───────────────────────────────────────┘
```

### 1. Downsamplers & The Registry Pattern

All downsamplers inherit from `nn.Module` and register themselves using `@DOWNSAMPLER_REGISTRY.register("variant_name")`. 

Below is the implementation pattern used in our zero-initialized Talebi resizer (`src/models/downsamplers/talebi.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import DOWNSAMPLER_REGISTRY


class _TalebiResBlock(nn.Module):
    """
    Residual block with LeakyReLU and BatchNorm used inside Talebi resizers.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.block(tensor)


@DOWNSAMPLER_REGISTRY.register("node21_talebi")
class TalebiPaperResizer(nn.Module):
    """
    Paper-faithful Talebi resizer (Talebi & Milanfar, ICCV 2021).
    Applies residual filtering over standard bilinear interpolation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 16,
        n_blocks: int = 2,
        target_size: int = 224,
    ):
        super().__init__()

        self.target_size = target_size

        # Entry convolutions at native resolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=7, padding=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(n_filters),
        )

        # Residual blocks operating at target downsampled resolution
        self.res_blocks = nn.Sequential(
            *[_TalebiResBlock(n_filters) for _ in range(n_blocks)]
        )

        # Post-residual convolution
        self.post_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
        )

        # Final projection layer generating the residual map
        self.output_conv = nn.Conv2d(n_filters, in_channels, kernel_size=7, padding=3)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # image_tensor: (B, 1, H_in, W_in)

        # Baseline bilinear downsampling branch
        baseline_downsampled = F.interpolate(
            image_tensor,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Learnable residual branch
        feature_map = self.input_conv(image_tensor)
        feature_map = F.interpolate(
            feature_map,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        skip_connection = feature_map
        feature_map = self.res_blocks(feature_map)
        feature_map = self.post_conv(feature_map) + skip_connection

        residual = self.output_conv(feature_map)

        return baseline_downsampled + residual


@DOWNSAMPLER_REGISTRY.register("node21_talebi_zeroinit")
class TalebiPaperResizerZeroInit(TalebiPaperResizer):
    """
    Talebi resizer initialized with near-zero residual projection.
    Starts with output strictly matching bilinear interpolation while preserving upstream gradients.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 16,
        n_blocks: int = 2,
        target_size: int = 224,
    ):
        super().__init__(
            in_channels=in_channels,
            n_filters=n_filters,
            n_blocks=n_blocks,
            target_size=target_size,
        )

        # Near-zero scaling prevents initial perturbation while maintaining backpropagation flow
        nn.init.kaiming_normal_(self.output_conv.weight, a=0.2)
        self.output_conv.weight.data *= 0.01
        nn.init.zeros_(self.output_conv.bias)
```

### 2. End-to-End Detector Assembly (`src/models/detector.py`)

The `NoduleDetector` coordinates the downsampler, backbone, Feature Pyramid Network (FPN), and CenterNet head:

```python
from src.core.registry import BACKBONE_REGISTRY, DOWNSAMPLER_REGISTRY


class NoduleDetector(nn.Module):
    """
    End-to-End nodule detection network integrating learnable downsamplers with FPN detectors.
    """

    def __init__(
        self,
        variant: str = "bilinear",
        backbone_name: str = "convnext_fpn",
        target_size: int = 224,
    ):
        super().__init__()

        # Instantiate downsampler module via registry
        self.downsampler = DOWNSAMPLER_REGISTRY.build(
            variant,
            target_size=target_size,
        )

        # Instantiate backbone and detection head
        self.backbone = BACKBONE_REGISTRY.build(backbone_name)
        self.head = CenterNetHead(in_channels=self.backbone.out_channels)

    def forward(self, native_cxr: torch.Tensor) -> dict[str, torch.Tensor]:
        # native_cxr: (B, 1, 2048, 2048) or (B, 1, 1024, 1024)

        downsampled_image = self.downsampler(native_cxr)
        # downsampled_image: (B, 1, 224, 224)

        features = self.backbone(downsampled_image)
        predictions = self.head(features)

        return {
            "downsampled_image": downsampled_image,
            "heatmap": predictions["heatmap"],
            "size_offsets": predictions["size_offsets"],
        }
```

### 3. Loss Functions & Optimization

* **Focal Loss v5 (`src/losses/focal.py`):** Modified Gaussian focal loss targeting nodule centers with adaptive radius proportional to physical diameter (mm).
* **Offset & Size Regression (`src/losses/`):** Smooth L1 loss for sub-pixel peak alignment and bounding box dimensions.
* **Auxiliary Loss:** Optional Total Variation (`src/losses/total_variation.py`) and MSE constraints on residual maps.

### 4. Evaluation Protocol (`src/evaluation/`)

* **FROC & CPM (`src/evaluation/froc.py`):** Free-Response ROC curve computed at 7 operational false-positive points: `[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]` false positives per scan. Competitive Performance Metric (CPM) is the average sensitivity across these 7 points.
* **Size Stratification (`src/evaluation/size_stratified.py`):** Breakdown of sensitivity into three clinically meaningful bins:
  * Small: $< 10\text{ mm}$
  * Medium: $10 - 20\text{ mm}$
  * Large: $> 20\text{ mm}$

---

## Configuration System (YAML)

All experiments are driven by structured YAML files in `configs/`, loaded and validated by `src/core/config.py`:

```yaml
# configs/base.yaml
experiment:
  name: "node21_ablation"
  seed: 42
  device: "cuda"

data:
  dataset: "node21"
  input_size: 1024
  target_size: 224
  batch_size: 16
  num_workers: 4

model:
  variant: "node21_talebi_zeroinit"
  backbone: "convnext_fpn"

training:
  epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 0.05
  cosine_lr: true
```

CLI arguments automatically override YAML settings:
```bash
python scripts/train.py --variant node21_talebi_zeroinit --backbone convnext_fpn --seed 42 --epochs 50
```

---

## PowerShell Automation Scripts

Batch sweeps and multi-seed matrices are located in `scripts/experiments/`. They resolve repository root paths automatically and log runs to disk:

```powershell
# Run the V2 Zero-Init multi-seed evaluation
.\scripts\experiments\run_v2_zeroinit.ps1

# Run multi-seed ablation matrix (seeds 42, 7, 123)
.\scripts\experiments\run_multiseed_ablation.ps1

# Run general parameter sweep
.\scripts\experiments\sweep.ps1 -RunGroup "talebi_sweep" -Variants "bilinear,node21_talebi_zeroinit" -Seeds "42,7" -Epochs 50
```

---

## Output Organization & MLflow Tracking

Every artifact generated during training is automatically organized within `outputs/`:

```
outputs/
├── figures/                       # Exported plots and heatmaps
├── results/                       # JSON/CSV metric summaries
├── mlflow/                        # Centralized SQLite database
│   ├── mlruns.db
│   └── mlruns/
├── talebi_table4/                 # Table 4 benchmark reports & logs
└── v8/                            # Date-stamped run checkpoints and outputs
```

### Viewing Results in MLflow UI

To launch the MLflow UI and view all metrics, parameters, and loss curves:

```bash
python -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlruns.db
```

Open your browser at `http://localhost:5000`.

---

## Getting Started

### 1. Environment Setup

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

# 3. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Verify GPU / Hardware

```bash
python scripts/diagnostics/check_cuda.py
```

### 3. Datasets Setup

Place datasets in `data_sets/`:

```
data_sets/
├── JSRT/
│   ├── jsrt_metadata.csv
│   └── images/images/*.png
└── node21/
    └── cxr_images/original_data/
        ├── metadata.csv
        └── images/*.mha
```

### 4. Offline Preprocessing & Caching

```bash
# Cache NODE21 .mha files to .npy for high-speed training
python scripts/data/cache_node21_npy.py --workers 6

# Compute size stratification bins
python scripts/data/stratify_nodules.py

# Generate binary lung masks
python scripts/data/generate_lung_masks.py
```
