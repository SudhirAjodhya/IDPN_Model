# Reproducing the report

Every table, figure and quoted statistic in the report comes from one config,
[`configs/report/reproduce.yml`](configs/report/reproduce.yml), run by
[`scripts/reproduce.py`](scripts/reproduce.py). Run everything from `Model/`.

```bash
python -m scripts.reproduce --dry-run  # list every command; runs nothing
python -m scripts.reproduce --stage node21 vindr --smoke 1   # 1-epoch check of setup
python -m scripts.reproduce   # everything, in order
```

Training cells whose `metrics.json` already exists are skipped, so an interrupted run
resumes where it stopped. Each command's output is also appended to
`outputs/reproduce/<stage>.log`.

## What produces what

| Report element | Stage | Script | Output |
|---|---|---|---|
| Table I, parameter counts | `tables` | `scripts/eval/report_tables.py` | printed |
| Table II, NODE21 sensitivity at 0.25 FP/img | `node21`, then `tables` | `report_tables.py` | printed LaTeX rows |
| Table III, VinDr micro-AUC and macro-AP, paired tests | `vindr`, `score`, `tables` | `report_tables.py` | printed LaTeX rows |
| VinDr frequency-band table (three seeds) | `tables` | `scripts/eval/frequency_bands_vindr.py` | printed LaTeX rows |
| NODE21 frequency-band table (seed 42) | `tables` | `scripts/eval/frequency_bands.py` | printed LaTeX rows |
| Fig. 1, decimation cost on n0189 | `figures` | `scripts/eval/plot_decimation_cost.py` | `fig01_decimation_cost_node21_n0189.png` |
| FROC, VinDr delta and per-class figures | `figures` | `scripts/eval/plot_report_results.py` | `results_*.png` |
| Qualitative cases figure | `figures` | `scripts/eval/compare_resizer_outputs.py` | `qualitative_resizer_cases.png` |
| Per-class gains quoted in the Discussion | `figures` | `plot_vindr_sweep.py` | printed |
| All figures | `figures` | see `reproduce.yml` | `outputs/report_figures/report/*.png` |

The `figures` stage copies its outputs to the file names used in the report's
`images/` directory, so that folder can be replaced wholesale.

## Generating the tables and figures yourself

Once training has finished, the tables and figures can be regenerated without
retraining:

```bash
python -m scripts.reproduce --stage score tables figures
```

Or run each command on its own, from `Model/`, in this order. Scoring must come first,
because Table III reads the cached VinDr test scores.

```bash
# VinDr test scores, needed by Table III
python -m scripts.eval.score_vindr_checkpoints --run-group v23_vindr_3seed

# Tables I, II and III, with the paired statistics quoted in the Results
python -m scripts.eval.report_tables

# Frequency-band tables
python -m scripts.eval.frequency_bands --films 20 --seed 42
python -m scripts.eval.frequency_bands_vindr --films 100

# Fig. 1, decimation cost
python -m scripts.eval.plot_decimation_cost --film n0189 --out-dir outputs/report_figures

# FROC curves, VinDr difference against decimation, VinDr per-class difference
python -m scripts.eval.plot_report_results --out-dir outputs/report_figures

# Per-class gains quoted in the Discussion
python -m scripts.eval.plot_vindr_sweep --run-group v23_vindr_3seed --out-dir outputs/report_figures
```

The qualitative figure compares the seed-42, 224-pixel run of each resizer, so it needs
their run directories:

```bash
python scripts/eval/compare_resizer_outputs.py \
  --a outputs/v8/<date>_v15_selection_and_scale/bilinear_fcos_focal_v5_s42_i1024_t224_vindr_vin2048_vnr1 \
  --b outputs/v8/<date>_v15_selection_and_scale/node21_talebi_zeroinit_fcos_focal_v5_s42_i1024_t224_vindr_vin2048_vnr1 \
  --films n1076 n1032 n0757 n0637 --out outputs/report_figures
```

Replace `<date>` with the folder each run was written to.
`python -m scripts.reproduce --stage figures` finds both directories automatically.

## Stages

| Stage | Runs | Measured time on one RTX 5070 Ti (16 GB) |
|---|---|---|
| `prepare` | builds the VinDr cache at input 2048 | about an hour, once (8,788 films at ~0.37 s each, single worker) |
| `node21` | 36 training runs: 6 resolutions x 2 resizers x 3 seeds | 178 GPU-hours (2.5 to 11.9 h per run) |
| `vindr` | 36 training runs, same grid | 240 GPU-hours (2.9 to 15.8 h per run) |
| `score` | forward pass of each VinDr checkpoint over the test split | minutes |
| `tables`, `figures` | read results from `outputs/` | minutes |

Disk: 161 GB for the VinDr DICOMs plus 69 GB for the cache `prepare` builds.

## Setup

Python 3.13 with CUDA 12.8. Exact versions are in
[`requirements-report.txt`](requirements-report.txt):

```bash
python -m venv torch_env
torch_env/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
torch_env/Scripts/python -m pip install -r requirements-report.txt
```

## Data

**NODE21.** Download the
[NODE21 dataset from Kaggle](https://www.kaggle.com/datasets/pshikk/node-21-dataset-untampered)
and place it so that this layout exists:

```
Model/data_sets/node21/cxr_images/original_data/metadata.csv
Model/data_sets/node21/cxr_images/original_data/images/*.mha     (4,882 films)
```

**VinDr-CXR.** Download the
[VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)
competition data. Only `train.csv` and `train/*.dicom` are used:

```
Model/data_sets/vindr/train.csv
Model/data_sets/vindr/train/*.dicom   (15,000 films)
```

The DICOMs take 161 GB. To keep them on another drive, set `VINDR_ROOT` to the folder
that holds `train.csv` instead:

```bash
export VINDR_ROOT=/mnt/big/vinbigdata-chest-xray-abnormalities-detection
```

The `prepare` stage builds `train_png16_cache_i2048/` inside that directory and checks
it holds exactly **8,788** films. That number matters: the VinDr experiment takes its
film population from the cache contents, so warming the cache with any other
`--negative-ratio` silently changes the experiment. If the check fails, empty the cache
directory and run `prepare` again.

JSRT is not needed.

