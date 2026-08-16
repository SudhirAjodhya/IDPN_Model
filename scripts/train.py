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
LUNG_MASK_DIR = ROOT_DIR / "data" / "lung_masks"
OUTPUT_DIR = ROOT_DIR / "outputs" / "v8"

MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'outputs' / 'mlflow' / 'mlruns.db').as_posix()}"
MLFLOW_EXPERIMENT = "nodule_v8"

LOSS_FNS = {
    "focal_v5": focal_loss_heatmap_v5,
    "focal":    focal_loss_heatmap,
    "mse":      mse_heatmap_loss,
    "qfl":      quality_focal_loss,
}

VARIANTS = [
    "bilinear", "alpha", "gamma", "delta", "lip", "blurpool", "talebi",
    "saliency_talebi", "inverse_bone_talebi", "talebi_aux", "node21_talebi",
    "node21_talebi_zeroinit", "node21_talebi_zeroinit_attn",
]

def parse_args():
    p = argparse.ArgumentParser(description="LDM Training Script")
    p.add_argument("--variant", choices=VARIANTS, default="bilinear")
    p.add_argument("--loss", choices=list(LOSS_FNS), default="focal_v5")
    p.add_argument("--head", choices=["centernet", "centernet_hr"], default="centernet")
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
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--aux-lambda", type=float, default=0.1, help="Weight for auxiliary lung mask loss (talebi_aux only)")
    p.add_argument("--input-size", type=int, default=2048)
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--augment", action="store_true", help="Enable data augmentation for training splits")
    p.add_argument("--val-frac", type=float, default=None,
                   help="Val fraction of non-test remainder (both JSRT and NODE21). Default: legacy (0.111/0.125). v2 uses 0.20.")
    p.add_argument("--backbone", type=str, default="convnext_fpn",
                   choices=["convnext_fpn", "convnext_fpn_scratch", "resnet50_fpn", "resnet50_fpn_scratch"],
                   help="Backbone name (registry key). _scratch variants skip ImageNet init.")
    p.add_argument("--datasets", choices=["both", "node21"], default="both",
                   help="'both' (legacy JSRT+NODE21) or 'node21' (Node21-only, leakage-free)")

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date_str = date.today().strftime("%Y-%m-%d")
    # Backbone tag suffix: keep legacy dir naming when backbone is the default (convnext_fpn)
    # so v1 outputs aren't shadowed; append _<backbone> otherwise to prevent collisions across the matrix.
    _bb_tag = "" if args.backbone == "convnext_fpn" else f"_{args.backbone}"
    output_dir = (OUTPUT_DIR / f"{date_str}_{args.run_group}"
                  / f"{args.variant}_{args.head}_{args.loss}_s{args.seed}{_bb_tag}")
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
        input_size=args.input_size,
        val_frac=args.val_frac,
    )

    from torch.utils.data import DataLoader
    # Node21 loader is I/O bound (SimpleITK read per __getitem__, no cache).
    # persistent_workers keeps them alive across epochs (skip re-init cost),
    # pin_memory speeds CPU->GPU copies, prefetch_factor deepens the queue.
    dl_kwargs = dict(num_workers=6, persistent_workers=True, pin_memory=True, prefetch_factor=2)
    train_dl = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True,  **dl_kwargs)
    val_dl   = DataLoader(val_loader,   batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    test_dls = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, **dl_kwargs) for k, v in test_loaders.items()}

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
    ], weight_decay=0.05)   # matches matrix_v1 — 1e-4 was too weak and let params drift

    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = GradScaler(device=device.type, enabled=False)

    NO_PARAM_VARIANTS = {"bilinear", "blurpool"}
    initial_params = ({} if args.variant in NO_PARAM_VARIANTS else snapshot_saliency_params(model, args.variant))

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
            "heatmap_sigma_min": args.heatmap_sigma_min,
            "effective_sigma_min": effective_sigma_min,
            "heatmap_size": model.heatmap_size,
            "grad_accum": args.grad_accum,
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
            "dataset": "node21" if args.datasets == "node21" else "jsrt+node21",
            "hostname": socket.gethostname(),
        })

        history, diagnostics_history = train_loop(
            model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            train_loader=train_dl, val_loader=val_dl, variant=args.variant,
            epochs=args.epochs, patience=args.patience, device=device,
            grad_accum=args.grad_accum, aux_lambda=args.aux_lambda,
            initial_params=initial_params, output_dir=output_dir,
            loss_fn=loss_fn, effective_sigma_min=effective_sigma_min,
            input_size=args.input_size
        )

        best_ckpt = output_dir / "best_model.pth"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

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
        save_ldm_comparison(model, eval_dls, output_dir, device, input_size=args.input_size, target_size=args.target_size)

        if args.froc:
            froc_data = compute_froc_data_v5(model, eval_dls["combined"], device=device, input_size=args.input_size)
            save_froc_curve(froc_data, output_dir, args.variant)
            with open(output_dir / "froc_data.json", "w") as f:
                json.dump(froc_data, f, indent=4)

        metrics_payload = {
            "variant": args.variant,
            "loss": args.loss,
            "head": args.head,
            "heatmap_size": model.heatmap_size,
            "effective_sigma_min": effective_sigma_min,
            "run_group": args.run_group,
            "seed": args.seed,
            "test_seed": args.test_seed,
            "epochs_trained": len(history["train_loss"]),
            "best_val_sensitivity": history["best_val_sensitivity"],
            "best_val_sensitivity_epoch": history["best_val_sensitivity_epoch"],
            "patience": args.patience,
            "heatmap_sigma_min": args.heatmap_sigma_min,
            "backbone_lr_factor": args.backbone_lr_factor,
            "grad_accum": args.grad_accum,
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
                "lr": history["lr"],
            },
            "diagnostics_history": diagnostics_history,
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=4)

        mlflow.log_artifacts(str(output_dir))

if __name__ == "__main__":
    main()
