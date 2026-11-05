# Improving the Detection of Small Pulmonary Nodules in Chest X-Rays using Learnable Downsampling Techniques

Code for the honours report. It compares a small learned resizer, adapted from
Talebi and Milanfar (ICCV 2021), against bilinear interpolation as the downsampling step
in front of an otherwise identical chest X-ray model.

To reproduce the report's tables and figures, see **[REPRODUCE.md](REPRODUCE.md)**.

---

## The experiment

Chest X-rays are large (1024 to 2048 pixels per side) and are usually shrunk with
bilinear interpolation before a CNN sees them, which can discard the fine detail small
findings depend on. The report replaces that step with a learned resizer and holds
everything else fixed: backbone, head, data, augmentation and training schedule.

```
Input CXR (B, 1, S_in, S_in)        S_in = 1024 (NODE21) or 2048 (VinDr-CXR)
        |
        v
Resizer: bilinear, or learned (bilinear + zero-initialised residual), 13,649 parameters
        |   (B, 1, S_out, S_out)    S_out in {224, 256, 384, 512, 768, 1024}
        v
ConvNeXt-T (ImageNet weights) + FPN
        |   stride-4 level, 256 channels
        v
FCOS head: classification, box regression, centerness
        |
        +--> NODE21: nodule boxes, scored by FROC (sensitivity at 0.25 FP/img), CPM, AP
        +--> VinDr-CXR: 14-class image labels, scored by micro/macro AUC, macro-AP, F1
```

Both datasets are run at six output resolutions, with both resizers and three seeds
(42, 7, 123): 72 training runs in total.

| Resizer | Variant name in the code |
|---|---|
| Bilinear | `bilinear` |
| Learned resizer | `node21_talebi_zeroinit` |

---

## Repository structure

Only the parts the report uses are listed.

```
Model/
├── REPRODUCE.md                    # How to reproduce the report
├── configs/report/reproduce.yml    # Every run, table and figure in the report
├── scripts/
│   ├── reproduce.py                # Runs reproduce.yml
│   ├── train.py                    # NODE21 detection training
│   ├── train_vindr.py              # VinDr-CXR multi-label training
│   ├── warm_vindr_cache.py         # Builds the VinDr image cache
│   ├── data/cache_node21_npy.py    # Caches NODE21 .mha files as .npy
│   └── eval/
│       ├── report_tables.py        # Tables I, II, III and their paired tests
│       ├── frequency_bands.py      # NODE21 frequency-band table
│       ├── frequency_bands_vindr.py    # VinDr frequency-band table
│       ├── score_vindr_checkpoints.py  # Caches VinDr test scores
│       ├── plot_report_results.py  # FROC, VinDr delta and per-class figures
│       ├── plot_decimation_cost.py # Fig. 1
│       └── compare_resizer_outputs.py  # Qualitative cases figure
├── src/
│   ├── data/                       # NODE21 and VinDr datasets, splits, augmentation (CLAHE)
│   ├── models/
│   │   ├── downsamplers/           # bilinear.py, talebi.py (the learned resizer)
│   │   ├── backbones/              # convnext_fpn.py
│   │   ├── heads/                  # fcos.py
│   │   └── detector.py             # Resizer + backbone + head
│   ├── losses/fcos.py              # Focal, GIoU and centerness losses
│   ├── evaluation/                 # FROC, CPM, AP, VinDr multi-label metrics
│   └── training/                   # Training loop
└── outputs/                        # Run outputs, checkpoints and figures (git-ignored)
    ├── v8/                         # NODE21 runs
    ├── vindr/                      # VinDr runs
    └── report_figures/             # Figures generated for the report
```

The repository also holds downsamplers, backbones, heads and losses from earlier
experiments (LIP, BlurPool, weighted pooling, CenterNet, ResNet-FPN and others). The
report does not use them.

---

## Getting started

Setup, data download and the expected folder layout are in
[REPRODUCE.md](REPRODUCE.md#setup). Run everything from `Model/`.

Check the GPU:

```bash
python scripts/diagnostics/check_cuda.py
```

Train a single run, for example the learned resizer at 224 pixels, seed 42:

```bash
# NODE21
python -u scripts/train.py --backbone convnext_fpn --head fcos --variant node21_talebi_zeroinit --seed 42 --datasets node21_vindr --input-size 1024 --target-size 224 --vindr-input-size 2048 --vindr-negative-ratio 1.0 --batch-size 4 --grad-accum 2 --lr 2e-4 --epochs 50 --patience 15 --select-on val_sens_matched --aug-profile v3 --val-frac 0.20 --run-group my_runs --augment --froc --attribution

# VinDr-CXR
python -u scripts/train_vindr.py --variant node21_talebi_zeroinit --backbone convnext_fpn --seed 42 --split-seed 0 --input-size 2048 --target-size 224 --batch-size 8 --grad-accum 1 --lr 2e-4 --epochs 100 --patience 15 --select-smoothing 5 --min-delta 0.0005 --aug-profile v3 --num-workers 6 --run-group my_runs --augment --oversample
```

These are the exact settings used for the report. At outputs of 768 and 1024, VinDr
uses `--batch-size 4 --grad-accum 2`. Each run writes `metrics.json` and
`best_model.pth` to a date-stamped folder under `outputs/`.

Training is also logged to MLflow. To browse runs:

```bash
python -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlruns.db
```

Then open `http://localhost:5000`.
