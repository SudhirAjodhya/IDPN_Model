import argparse
import json
import math
import os
import socket
import warnings
from datetime import date
from pathlib import Path

import torch
from torch.amp import GradScaler
import mlflow

import sys
from pathlib import Path

# Add root directory to path to allow importing from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splits import build_three_way_split
from src.models.detector import NoduleDetector
from src.losses import focal_loss_heatmap_v5, focal_loss_heatmap, mse_heatmap_loss, quality_focal_loss
from src.training.trainer import train_loop
from src.evaluation.detection_metrics import compute_detection_metrics_v4
from src.evaluation.froc import compute_froc_data_v5
from src.visualization.training_curves import save_training_curves
from src.visualization.ldm_comparison import save_ldm_comparison
from src.visualization.froc_curve import save_froc_curve
from src.utils.misc import set_seed
from src.utils.formatting import print_banner, print_final_summary
from src.diagnostics.runner import snapshot_saliency_params

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data_sets"
JSRT_IMAGE_DIR = DATA_DIR / "JSRT" / "images" / "images"
JSRT_CSV_PATH = DATA_DIR / "JSRT" / "jsrt_metadata.csv"
NODE21_IMAGE_DIR = DATA_DIR / "node21" / "cxr_images" / "original_data" / "images"
NODE21_CSV_PATH = DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv"
# VinDr-CXR lives outside the repo data_sets tree because the PNG16 archive is ~125 GB.
# Written by scripts/preprocess_vindr.py from the original DICOM download.
VINDR_ROOT = Path("D:/Downloads/vinbigdata-chest-xray-abnormalities-detection")
VINDR_IMAGE_DIR = VINDR_ROOT / "train_png16"
VINDR_DICOM_DIR = VINDR_ROOT / "train"          # fallback when the PNG archive is absent
VINDR_CSV_PATH = VINDR_ROOT / "train.csv"
LUNG_MASK_DIR = ROOT_DIR / "data" / "lung_masks"
OUTPUT_DIR = ROOT_DIR / "outputs" / "v8"

def init_dataloader_worker(worker_id):
    """
    Pin each dataloader worker to a single thread for OpenCV and torch.

    Both libraries default to one thread per logical core INSIDE EVERY WORKER PROCESS, so
    6 workers on a 16-thread machine request 96 OpenCV threads. OpenCV's parallel_for_
    pool spins while waiting for work, so the surplus threads burn CPU without producing
    throughput. Measured symptom: 85% CPU against a 10% GPU, every worker pegged at a full
    core for hours, while the per-film cost is only ~12 ms.

    The per-film work (a resize, a warpAffine, CLAHE) is far too small to benefit from
    intra-op parallelism. Parallelism ACROSS workers is what matters and is already
    num_workers-wide.

    MUST be module-level: worker_init_fn is pickled to each worker, and Windows spawns
    rather than forks, so a nested function would fail to pickle.

    Worker RNG seeding is deliberately NOT done here. The augmentations in
    NODE21Dataset._augment (shared by VinDrDataset) draw from the numpy GLOBAL RNG, which
    looks exactly like the classic spawned-worker reproducibility bug, but PyTorch's
    _worker_loop already seeds numpy and random per worker from the parent generator.
    Verified by scripts/eval/check_worker_seeding.py - run that before "fixing" this.

    What DOES change the augmentation stream is num_workers itself: worker count partitions
    the sample indices, and each worker holds its own RNG stream, so the same seed at a
    different worker count augments different films. Runs are only comparable at equal
    worker counts, and num_workers is not currently recorded in metrics.json.
    """
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)


MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'outputs' / 'mlflow' / 'mlruns.db').as_posix()}"
MLFLOW_EXPERIMENT = "nodule_v8"

LOSS_FNS = {
    "focal_v5": focal_loss_heatmap_v5,
    "focal":    focal_loss_heatmap,
    "mse":      mse_heatmap_loss,
    "qfl":      quality_focal_loss,
}

# Read from the registry rather than restated here. A hand-maintained copy drifts the
# moment a variant is added: on 2026-09-06 four new resizers were registered and this list
# was not updated, so a 12-run sweep died at run 1 with "invalid choice" after passing the
# smoke test - argparse rejected keys the model layer had accepted all along. Deriving the
# choices makes registering a variant sufficient to run it.
#
# NOTE ON WARP VARIANTS (node21_ldm_warp*, 2026-08-23 offset-bound sweep): run
# scripts/diagnostics/warp_label_displacement.py before reading their detection numbers.
# The linear target rule in fcos_targets.py is only valid while label displacement stays
# small relative to nodule size, and it scales with max_offset.
import src.models.downsamplers                                       # noqa: F401  (registers)
from src.core.registry import DOWNSAMPLER_REGISTRY

# Accepted by the CLI but not a registry key: NoduleDetector maps it to "talebi" before the
# lookup (see detector.py). Deriving the list would otherwise silently drop it.
VARIANT_ALIASES = ["talebi_aux"]

VARIANTS = sorted(DOWNSAMPLER_REGISTRY.list_registered() + VARIANT_ALIASES)

def parse_args():
    p = argparse.ArgumentParser(description="LDM Training Script")
    p.add_argument("--variant", choices=VARIANTS, default="bilinear")
    p.add_argument("--loss", choices=list(LOSS_FNS), default="focal_v5")
    p.add_argument("--head", choices=["centernet", "centernet_hr", "centernet_det", "fcos"], default="centernet",
                   help="centernet_det restores CenterNet's size and offset branches, so the model "
                        "predicts boxes and IoU/AP become computable. fcos supervises a box at EVERY "
                        "cell inside the ground-truth box rather than at the centre only (6x more "
                        "supervised cells at target=224, 32x at 512). Both imply per-image sampling and "
                        "the box-based NODE21 metric; numbers are NOT comparable to the heatmap heads.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--run-group", type=str, default="default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-seed", type=int, default=0, help="Seed for test set split (should be fixed)")
    p.add_argument("--cosine-lr", action="store_true", help="Use CosineAnnealingLR")
    p.add_argument("--froc", action="store_true", help="Compute FROC curve at the end")
    p.add_argument("--heatmap-sigma-min", type=float, default=2.5)
    p.add_argument("--backbone-lr-factor", type=float, default=0.1)
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                   help="Hold the pretrained backbone fixed (weights AND BatchNorm stats) for this "
                        "many epochs so the randomly-initialised head can warm up without writing "
                        "noise into ImageNet features. 0 disables, matching all runs before 2026-08-20.")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--aux-lambda", type=float, default=0.1, help="Weight for auxiliary lung mask loss (talebi_aux only)")
    p.add_argument("--input-size", type=int, default=2048)
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--augment", action="store_true", help="Enable data augmentation for training splits")
    p.add_argument("--letterbox", action="store_true",
                   help="Resize with aspect ratio preserved and centre-pad to square. 39%% of NODE21 "
                        "films are non-square natively, so the default squash distorts nodules by a "
                        "per-image factor. Off by default: it changes every image and so invalidates "
                        "comparison with runs before 2026-08-21.")
    p.add_argument("--attribution", action="store_true",
                   help="After training, measure gradient and residual enrichment at nodules against "
                        "matched control boxes, and record both in metrics.json. This is the mechanism "
                        "readout - it says whether a change fixed the actual deficit, independently of "
                        "whether detection sensitivity moved.")
    p.add_argument("--aug-profile", choices=["v3", "v4"], default="v3",
                   help="Augmentation strength for the NODE21 train split. v3 is the setting all "
                        "results up to 2026-08-17 used; v4 is the stronger 2026-08-18 jitter.")
    p.add_argument("--select-on", choices=["val_sens", "val_loss", "val_sens_matched"], default="val_sens",
                   help="Metric driving checkpoint selection and early stopping. "
                        "val_sens (default, and what every run to date used) maximises "
                        "sensitivity at a FIXED score threshold of 0.3 with no constraint on "
                        "false positives, so it saves the most liberal epoch of the run - the "
                        "worst val loss and highest FP of a run can be the saved checkpoint. "
                        "val_sens_matched instead reads validation sensitivity at 0.25 FP/img, "
                        "the budget results are reported at, so selection agrees with "
                        "reporting. val_loss is smoother but trap 4.6 measured it selecting "
                        "the most conservative model rather than the best detector. "
                        "Runs are only comparable to others using the SAME criterion.")
    p.add_argument("--pos-frac", type=float, default=None,
                   help="Target fraction of nodule-bearing samples per epoch via weighted resampling "
                        "(e.g. 0.45). Omit for the natural ~28%% balance.")
    p.add_argument("--val-frac", type=float, default=None,
                   help="Val fraction of non-test remainder (both JSRT and NODE21). Default: legacy (0.111/0.125). v2 uses 0.20.")
    p.add_argument("--backbone", type=str, default="convnext_fpn",
                   choices=["convnext_fpn", "convnext_fpn_scratch", "resnet50_fpn", "resnet50_fpn_scratch",
                            "densenet121_cxr_fpn", "densenet121_cxr_nih_fpn", "densenet121_imagenet_fpn"],
                   help="Backbone name (registry key). _scratch variants skip ImageNet init. "
                        "densenet121_cxr_* are CHEST X-RAY pretrained via torchxrayvision and take "
                        "single-channel input, so the 3-channel replicate and ImageNet normalisation "
                        "in detector.py are skipped for them. densenet121_imagenet_fpn is the control "
                        "that separates architecture from pretraining domain.")
    p.add_argument("--datasets", choices=["both", "node21", "node21_vindr"], default="both",
                   help="'both' (legacy JSRT+NODE21), 'node21' (Node21-only, leakage-free), or "
                        "'node21_vindr' (Node21 train plus VinDr-CXR boxes as generic foreground; "
                        "val and test stay Node21-only so metrics remain comparable)")
    p.add_argument("--vindr-input-size", type=int, default=None,
                   help="Input size for VinDr films only, defaulting to --input-size. Set it "
                        "higher (VinDr is 2128-3408 px natively) to hand the resizer a ~12x "
                        "decimation to target instead of 4.6x. Batches then become "
                        "single-source, since default_collate cannot stack mixed image sizes. "
                        "ConvNeXt only: resnet50_fpn's 53 BatchNorm layers make "
                        "domain-homogeneous batches unsafe.")
    p.add_argument("--vindr-negative-ratio", type=float, default=None,
                   help="Keep this many box-free VinDr films per box-bearing one. 10,606 of "
                        "VinDr's 15,000 films carry NO boxes and so add nothing to the box "
                        "supervision while costing full epoch time. 1.0 keeps all 18,231 "
                        "boxes on 41%% fewer films. Default None keeps every negative.")
    p.add_argument("--num-workers", type=int, default=6,
                   help="DataLoader TRAIN workers. Do NOT raise this casually on Windows: train "
                        "and val each hold their own persistent worker pool, so the live process "
                        "count is num_workers + val workers, and every one is a fresh process "
                        "importing torch (~1.5 GB of commit each). 12 produced 33 live processes "
                        "holding 56 GB of a 90 GB commit limit and killed a run with WinError "
                        "1455. Once the VinDr cache is warm a film loads in 14 ms, so 6 workers "
                        "already keep the GPU fed and raising this buys nothing.")

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_str = date.today().strftime("%Y-%m-%d")
    # Backbone tag suffix: keep legacy dir naming when backbone is the default (convnext_fpn)
    # so v1 outputs aren't shadowed; append _<backbone> otherwise to prevent collisions across the matrix.
    _bb_tag  = "" if args.backbone == "convnext_fpn" else f"_{args.backbone}"
    # Resolution tag: only appended when input!=2048 or target!=512 (defaults). Keeps legacy
    # naming for v1/v2 runs, prevents collisions in v3 resolution sweep where multiple runs
    # share (variant, seed, backbone) but differ in input/target size.
    _res_tag = "" if (args.input_size == 2048 and args.target_size == 512) else f"_i{args.input_size}_t{args.target_size}"

    # Recipe tag: any knob that changes the run but is not otherwise in the path. Without
    # it, two runs differing only by --freeze-backbone-epochs (or --aug-profile, or
    # --backbone-lr-factor) resolve to the SAME directory and the second silently
    # overwrites the first. That destroyed half of the 2026-08-20 Arm V sweep. Only
    # non-default values are appended, so every existing run keeps its current path.
    _recipe_bits = []
    if args.freeze_backbone_epochs: _recipe_bits.append(f"fz{args.freeze_backbone_epochs}")
    if args.aug_profile != "v3":    _recipe_bits.append(f"aug{args.aug_profile}")
    if args.backbone_lr_factor != 0.1: _recipe_bits.append(f"bblr{args.backbone_lr_factor:g}")
    if args.pos_frac is not None:   _recipe_bits.append(f"pf{args.pos_frac:g}")
    if args.letterbox:              _recipe_bits.append("lb")
    # Pooled-data knobs. Without these, a node21-only run, a matched-resolution pooled run
    # and a native-resolution pooled run all collapse onto the same directory name.
    if args.datasets == "node21_vindr": _recipe_bits.append("vindr")
    if args.vindr_input_size and args.vindr_input_size != args.input_size:
        _recipe_bits.append(f"vin{args.vindr_input_size}")
    if args.vindr_negative_ratio is not None:
        _recipe_bits.append(f"vnr{args.vindr_negative_ratio:g}")
    _recipe_tag = ("_" + "_".join(_recipe_bits)) if _recipe_bits else ""

    output_dir = (OUTPUT_DIR / f"{date_str}_{args.run_group}"
                  / f"{args.variant}_{args.head}_{args.loss}_s{args.seed}{_bb_tag}{_res_tag}{_recipe_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lung_mask_dir = LUNG_MASK_DIR if args.variant == "talebi_aux" else None
    
    train_loader, val_loader, test_loaders, split_info = build_three_way_split(
        jsrt_csv_path=JSRT_CSV_PATH,
        jsrt_image_dir=JSRT_IMAGE_DIR,
        node21_csv_path=NODE21_CSV_PATH,
        node21_image_dir=NODE21_IMAGE_DIR,
        seed=args.seed,
        test_seed=args.test_seed,
        lung_mask_dir=lung_mask_dir,
        target_size=args.target_size,
        augment_train=args.augment,
        datasets=args.datasets,
        vindr_csv_path=VINDR_CSV_PATH,
        vindr_image_dir=VINDR_IMAGE_DIR,
        vindr_dicom_dir=VINDR_DICOM_DIR,
        vindr_input_size=args.vindr_input_size,
        vindr_negative_ratio=args.vindr_negative_ratio,
        input_size=args.input_size,
        val_frac=args.val_frac,
        aug_profile=args.aug_profile,
        group_by_image=(args.head in ("centernet_det", "fcos")),
        letterbox=args.letterbox,
    )

    from torch.utils.data import DataLoader
    # Node21 loader is I/O bound (SimpleITK read per __getitem__, no cache).
    # persistent_workers keeps them alive across epochs (skip re-init cost),
    # pin_memory speeds CPU->GPU copies, prefetch_factor deepens the queue.
    dl_kwargs = dict(num_workers=args.num_workers, persistent_workers=True, pin_memory=True,
                     prefetch_factor=2, worker_init_fn=init_dataloader_worker)

    # These two are mutually exclusive: a WeightedRandomSampler draws across the whole
    # pooled set and would build batches that mix the two image sizes, which fails deep
    # inside default_collate with an unhelpful shape error. Fail here instead.
    if args.pos_frac is not None and args.datasets == "node21_vindr":
        raise SystemExit("--pos-frac cannot be combined with --datasets node21_vindr: "
                         "oversampling draws across both datasets, but pooled batches must "
                         "be single-source (differing box padding width, and differing image "
                         "size when --vindr-input-size is set).")

    # Positive oversampling. Natural NODE21 train balance is ~28% nodule-bearing rows
    # (1476 annotations vs 3748 negatives). --pos-frac draws positives more often so each
    # epoch sees roughly that share instead. Sampling is with replacement, and augmentation
    # runs inside __getitem__, so a repeated film comes back with a different rotation,
    # shift and noise draw each time rather than as an identical duplicate.
    if args.pos_frac is not None:
        from torch.utils.data import WeightedRandomSampler
        from src.data.splits import collect_positive_flags

        positive_flags = collect_positive_flags(train_loader)
        num_positive = sum(positive_flags)
        num_negative = len(positive_flags) - num_positive

        # Per-sample weight = target share of its class, divided evenly within the class
        weight_positive = args.pos_frac / num_positive
        weight_negative = (1.0 - args.pos_frac) / num_negative
        sample_weights = [weight_positive if is_positive else weight_negative for is_positive in positive_flags]

        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(positive_flags), replacement=True)
        train_dl = DataLoader(train_loader, batch_size=args.batch_size, sampler=train_sampler, **dl_kwargs)

        print(f"[sampler] pos_frac={args.pos_frac:.2f} | {num_positive} positive / {num_negative} negative rows "
              f"| natural share {num_positive / len(positive_flags):.3f}")

    elif args.datasets == "node21_vindr":
        # Pooled batches MUST be single-source, for two independent reasons. Either one
        # alone breaks default_collate:
        #   1. Box padding width differs. NODE21 pads to MAX_OBJECTS=4 (it has at most 3
        #      nodules per film); VinDr pads to 16 (median 3 boxes, p99 15). A mixed batch
        #      tries to stack (4,4) with (16,4).
        #   2. Image size differs whenever --vindr-input-size is set, e.g. stacking
        #      (1,1,1024,1024) with (1,1,2700,2700).
        # Reason 1 applies even at matched resolution, so this branch is keyed on the
        # dataset mode rather than on mixed_input_sizes. Batch ORDER is still interleaved,
        # so the optimizer alternates between domains rather than seeing a curriculum.
        from src.data.pooled_batch_sampler import SingleSourceBatchSampler

        assert args.backbone.startswith("convnext"), (
            f"--datasets node21_vindr requires a BatchNorm-free backbone; {args.backbone} "
            "would get a BN-statistics confound from domain-homogeneous batches")

        cumulative = train_loader.cumulative_sizes
        component_sizes = [b - a for a, b in zip([0] + cumulative[:-1], cumulative)]

        batch_sampler = SingleSourceBatchSampler(component_sizes, batch_size=args.batch_size,
                                                 seed=args.seed)
        train_dl = DataLoader(train_loader, batch_sampler=batch_sampler, **dl_kwargs)

        print(f"[sampler] single-source batches | node21 @ {args.input_size}px, "
              f"vindr @ {split_info['vindr_input_size']}px | components {component_sizes} "
              f"| {len(batch_sampler)} batches/epoch "
              f"| mixed_sizes={split_info.get('mixed_input_sizes')}")

    else:
        train_dl = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True,  **dl_kwargs)
    # Validation and test get their OWN, smaller worker pools. They previously reused
    # dl_kwargs, so each held as many persistent workers as training did - on Windows every
    # worker is a separate process importing torch, and the pools coexist for the whole run.
    # Val is 924 films against training's 12,379, so it saturates on 2 workers, and this
    # cuts the live process count by roughly two thirds.
    eval_kwargs = dict(dl_kwargs, num_workers=min(2, args.num_workers))
    val_dl   = DataLoader(val_loader,   batch_size=args.batch_size, shuffle=False, **eval_kwargs)
    test_dls = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, **eval_kwargs) for k, v in test_loaders.items()}

    model = NoduleDetector(variant=args.variant, head=args.head, target_size=args.target_size, backbone=args.backbone).to(device)

    # sigma_min scaling MUST match matrix_v1: scale linearly with heatmap ratio
    # relative to the base 128. matrix_v1 ran at target=224 → hm=56 → sigma=1.09
    # and learned fine. My previous "fix" (unscaled sigma=2.5) diverged from the
    # working recipe and destabilised training.
    base_heatmap_size = 128
    effective_sigma_min = args.heatmap_sigma_min * (model.heatmap_size / base_heatmap_size)
    args.effective_sigma_min = effective_sigma_min
    args.effective_sigma_min = effective_sigma_min

    print_banner(args, split_info, model, device, input_size=args.input_size)

    backbone_params = list(model.backbone_fpn.features.parameters())
    other_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_factor},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=0.05)   # matches matrix_v1 - 1e-4 was too weak and let params drift

    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = GradScaler(device=device.type, enabled=False)

    NO_PARAM_VARIANTS = {"bilinear", "blurpool"}
    initial_params = ({} if args.variant in NO_PARAM_VARIANTS else snapshot_saliency_params(model, args.variant))

    # centernet_det supplies its own composite loss inside the trainer; loss_fn stays
    # bound for the heatmap heads and for the diagnostics that expect it.
    loss_fn = LOSS_FNS[args.loss]
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = f"{args.run_group}/{args.variant}_{args.head}_{args.loss}_s{args.seed}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "variant": args.variant,
            "loss": args.loss,
            "head": args.head,
            "seed": args.seed,
            "test_seed": args.test_seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr_factor": args.backbone_lr_factor,
            "select_on": args.select_on,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "aug_profile": args.aug_profile,
            "letterbox": args.letterbox,
            "pos_frac": args.pos_frac,
            "heatmap_sigma_min": args.heatmap_sigma_min,
            "effective_sigma_min": effective_sigma_min,
            "heatmap_size": model.heatmap_size,
            "grad_accum": args.grad_accum,
            "num_workers": args.num_workers,   # changes the augmentation stream - see below
            "cosine_lr": args.cosine_lr,
            "augment": args.augment,
            "run_group": args.run_group,
            "total_params": total_params,
            "trainable_params": trainable,
            "aux_lambda": args.aux_lambda,
        })
        mlflow.set_tags({
            "variant": args.variant,
            "loss": args.loss,
            "head": args.head,
            "seed": str(args.seed),
            "run_group": args.run_group,
            "dataset": {"node21": "node21", "node21_vindr": "node21+vindr"}.get(args.datasets, "jsrt+node21"),
            "hostname": socket.gethostname(),
        })

        history, diagnostics_history = train_loop(
            model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            train_loader=train_dl, val_loader=val_dl, variant=args.variant,
            epochs=args.epochs, patience=args.patience, device=device,
            grad_accum=args.grad_accum, aux_lambda=args.aux_lambda,
            initial_params=initial_params, output_dir=output_dir,
            loss_fn=loss_fn, effective_sigma_min=effective_sigma_min,
            input_size=args.input_size, select_on=args.select_on,
            freeze_backbone_epochs=args.freeze_backbone_epochs
        )

        best_ckpt = output_dir / "best_model.pth"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

        if args.head in ("centernet_det", "fcos"):
            from src.evaluation.node21_detection_metrics import compute_box_detection_metrics, froc_auc

            # Report both a permissive and the COCO-standard IoU so the choice is visible
            box_metrics = {}
            for iou in (0.25, 0.5):
                m = compute_box_detection_metrics(model, test_dls["combined"], device=device,
                                                  input_size=args.input_size, iou_threshold=iou)
                m["froc_auc"] = froc_auc(m["froc"])
                box_metrics[f"iou{iou}"] = m

            primary = box_metrics["iou0.25"]
            flat_test = {
                "combined": primary, "node21": primary, "jsrt": {}, "jsrt_by_size": {},
                "box_metrics": box_metrics,
            }

        else:
            combined_metrics = compute_detection_metrics_v4(model, test_dls["combined"], device=device, input_size=args.input_size)
            flat_test = {
                "combined": combined_metrics.get("all", {}),
                "jsrt": combined_metrics.get("jsrt", {}),
                "node21": combined_metrics.get("node21", {}),
                "jsrt_by_size": combined_metrics.get("jsrt_by_size", {}),
            }

        print_final_summary(history, flat_test, output_dir)

        combined = flat_test["combined"]
        jsrt     = flat_test.get("jsrt", {})       # empty in node21-only mode
        node21   = flat_test.get("node21", {})
        by_size  = flat_test.get("jsrt_by_size", {})
        metrics = {
            "test_sensitivity_combined": combined.get("sensitivity", 0),
            "test_sensitivity_node21":   node21.get("sensitivity", 0),
            "test_fp_combined":          combined.get("fp_per_image", 0),
            "test_fp_node21":            node21.get("fp_per_image", 0),
            "best_val_sensitivity": history["best_val_sensitivity"],
            "best_epoch":           history["best_val_sensitivity_epoch"],
            "peak_sensitivity":     history["peak_sensitivity"],
            "epochs_trained":       len(history["train_loss"]),
        }
        if jsrt:                                  # only log JSRT metrics when JSRT is in the split
            metrics.update({
                "test_sensitivity_jsrt":        jsrt.get("sensitivity", 0),
                "test_fp_jsrt":                 jsrt.get("fp_per_image", 0),
                "test_sensitivity_jsrt_small":  by_size.get("small",  {}).get("sensitivity", 0),
                "test_sensitivity_jsrt_medium": by_size.get("medium", {}).get("sensitivity", 0),
                "test_sensitivity_jsrt_large":  by_size.get("large",  {}).get("sensitivity", 0),
            })
        mlflow.log_metrics(metrics)

        save_training_curves(history, output_dir)

        # Release training dataloaders' worker processes + shared memory before eval loaders
        # spin up their own. Successive persistent-worker DataLoaders on Windows exhaust
        # the shared file mapping pool otherwise (error 1455).
        del train_dl, val_dl, test_dls
        import gc; gc.collect()

        # Fresh eval loaders with num_workers=0: no throughput need, no shared-mem accumulation.
        eval_dls = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
                    for k, v in test_loaders.items()}
        # Optional extras are wrapped: a failure here must never cost the run its metrics.
        # save_ldm_comparison, compute_froc_data_v5 and the attribution pass have each
        # destroyed a completed run's metrics.json by raising before it was written.
        try:
            save_ldm_comparison(model, eval_dls, output_dir, device, input_size=args.input_size, target_size=args.target_size)
        except Exception as exc:
            print(f"  [warn] ldm_comparison failed, continuing: {exc}")

        try:
          if args.froc:
            if args.head in ("centernet_det", "fcos"):
                # Box FROC was already swept during test evaluation; reuse it rather than
                # re-running the model under the centre-distance criterion.
                curve = flat_test["box_metrics"]["iou0.25"]["froc"]
                froc_data = {
                    "thresholds": [p["threshold"] for p in curve],
                    "all": curve, "node21": curve, "jsrt": [],
                    "criterion": "box_iou_0.25",
                }
            else:
                froc_data = compute_froc_data_v5(model, eval_dls["combined"], device=device, input_size=args.input_size)
                save_froc_curve(froc_data, output_dir, args.variant)

            with open(output_dir / "froc_data.json", "w") as f:
                json.dump(froc_data, f, indent=4)
        except Exception as exc:
            print(f"  [warn] FROC failed, continuing: {exc}")

        # Mechanism readout: does the residual concentrate on nodules? Reported alongside
        # detection metrics so an intervention can be judged on whether it fixed the
        # measured deficit, not only on whether sensitivity moved.
        attribution = None

        try:
          if args.attribution and args.head in ("centernet_det", "fcos"):
            from src.evaluation.residual_attribution_hook import run_attribution
            attribution = run_attribution(model, eval_dls["node21"], device,
                                          input_size=args.input_size,
                                          target_size=args.target_size, head=args.head)
            def _fmt(value, places):
                # bilinear has no residual, so its residual enrichment is legitimately
                # undefined - every control-box ratio is skipped and None comes back
                return "n/a" if value is None else f"{value:.{places}f}"

            print(f"  [attribution] gradient enrichment {_fmt(attribution['gradient_enrichment'], 1)} "
                  f"| residual enrichment {_fmt(attribution['residual_enrichment'], 3)} "
                  f"(n={attribution['n_nodules']} nodules)")
        except Exception as exc:
            print(f"  [warn] attribution failed, continuing: {exc}")
            attribution = None

        metrics_payload = {
            "attribution": attribution,
            "variant": args.variant,
            "loss": args.loss,
            "head": args.head,
            "heatmap_size": model.heatmap_size,
            "input_size":  args.input_size,                        # v3+ res sweep needs this; earlier runs default to 2048
            "target_size": args.target_size,
            "backbone":    args.backbone,
            "box_metrics": flat_test.get("box_metrics"),        # centernet_det only
            "effective_sigma_min": effective_sigma_min,
            "run_group": args.run_group,
            "seed": args.seed,
            "test_seed": args.test_seed,
            "epochs_trained": len(history["train_loss"]),
            "best_val_sensitivity": history["best_val_sensitivity"],
            "best_val_sensitivity_epoch": history["best_val_sensitivity_epoch"],
            "best_val_sens_matched": history.get("best_val_sens_matched"),
            "best_val_sens_matched_epoch": history.get("best_val_sens_matched_epoch"),
            "patience": args.patience,
            "heatmap_sigma_min": args.heatmap_sigma_min,
            "backbone_lr_factor": args.backbone_lr_factor,
            "select_on": args.select_on,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "aug_profile": args.aug_profile,
            "letterbox": args.letterbox,
            "pos_frac": args.pos_frac,
            "grad_accum": args.grad_accum,
            # Load-bearing despite looking like a throughput knob: worker count partitions
            # the sample indices and each worker holds its own RNG stream, so the same seed
            # at a different worker count augments different films. Two runs are only
            # comparable at equal num_workers. Untracked until 2026-09-02, which is why the
            # v14/v15 divergence could not be diagnosed from the stored metrics.
            "num_workers": args.num_workers,
            "cosine_lr": args.cosine_lr,
            "augment": args.augment,
            "aux_lambda": args.aux_lambda,
            "split_info": split_info,
            "test_results": flat_test,
            "peak_sensitivity": history["peak_sensitivity"],
            "peak_sensitivity_epoch": history["peak_sensitivity_epoch"],
            "history": {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "sensitivity": history["sensitivity"],
                "fp_per_image": history["fp_per_image"],
                # Validation sensitivity at the budget results are reported at. `sensitivity`
                # above is at a fixed score threshold of 0.3 and is not the reported number.
                "sens_matched": history.get("sens_matched", []),
                "lr": history["lr"],
            },
            "diagnostics_history": diagnostics_history,
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=4)

        mlflow.log_artifacts(str(output_dir))

if __name__ == "__main__":
    main()
