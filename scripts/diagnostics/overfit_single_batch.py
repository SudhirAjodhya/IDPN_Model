"""Overfit-a-single-batch diagnostic.

Trains on a FIXED small batch of Node21 positive samples with augmentation OFF,
validation OFF, no early stopping. Logs train loss + train sensitivity per epoch.

Interpretation:
  * Loss -> 0 (< 0.05) AND train sens -> 1.0 within ~100 epochs
    => model + loss + head + gradient path all work.
    Generalization is a data / regularization / capacity issue.
  * Loss plateaus above ~0.5 OR train sens stays below 0.8
    => model literally cannot memorize a handful of samples;
    something structural is wrong (loss doesn't backprop cleanly,
    head capacity is too small, backbone can't adapt, etc.)
  * Loss NaN / diverges
    => numerical stability bug (fp16 overflow, exploding gradient,
    bad init combo).

Runs on CPU-friendly settings: batch=8 positive samples, 200 epochs, no dataloader
workers (all in RAM). ~10-15 min on GPU. If this passes, the full pipeline's
plateau is definitively a generalization problem, not a broken model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import build_three_way_split
from src.evaluation.heatmap import make_gt_heatmap
from src.evaluation.detection_metrics import compute_detection_metrics_v4
from src.core.registry import LOSS_REGISTRY
import src.losses.focal              # trigger @LOSS_REGISTRY.register decorators
from src.models.detector import NoduleDetector

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data_sets"
DEFAULTS = {
    "jsrt_csv":      DATA_DIR / "JSRT" / "jsrt_metadata.csv",
    "jsrt_images":   DATA_DIR / "JSRT" / "images" / "images",
    "node21_csv":    DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv",
    "node21_images": DATA_DIR / "node21" / "cxr_images" / "original_data" / "images",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",     default="bilinear",
                    choices=["bilinear", "talebi", "node21_talebi"])
    ap.add_argument("--n-samples",   type=int, default=8,
                    help="Positive Node21 samples to overfit on")
    ap.add_argument("--epochs",      type=int, default=200)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--backbone-lr-factor", type=float, default=0.1)
    ap.add_argument("--weight-decay",       type=float, default=0.05)
    ap.add_argument("--target-size", type=int, default=224)
    ap.add_argument("--input-size",  type=int, default=2048)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grab N positive samples out of Node21 training set. Augment off.
    train_ds, _, _, _ = build_three_way_split(
        jsrt_csv_path=DEFAULTS["jsrt_csv"], jsrt_image_dir=DEFAULTS["jsrt_images"],
        node21_csv_path=DEFAULTS["node21_csv"], node21_image_dir=DEFAULTS["node21_images"],
        seed=args.seed, test_seed=0, datasets="node21",
        input_size=args.input_size, augment_train=False,
    )
    positives = [i for i in range(len(train_ds)) if bool(train_ds[i][1]["has_nodule"])]
    picked = positives[: args.n_samples]
    print(f"Picked {len(picked)} positive samples out of {len(positives)} available.")

    # Materialize the batch once, keep it on GPU. No dataloader, no workers.
    imgs, targets = [], {"has_nodule": [], "center_2048": ([], []),
                         "radius_px": [], "filename": []}
    for i in picked:
        img, tgt = train_ds[i]
        imgs.append(img)
        targets["has_nodule"].append(bool(tgt["has_nodule"]))
        targets["center_2048"][0].append(float(tgt["center_2048"][0]))
        targets["center_2048"][1].append(float(tgt["center_2048"][1]))
        targets["radius_px"].append(float(tgt["radius_px"]))
        targets["filename"].append(tgt.get("filename", f"idx_{i}"))
    images = torch.stack(imgs).to(device)
    batched_targets = {
        "has_nodule":  torch.tensor(targets["has_nodule"]),
        "center_2048": (torch.tensor(targets["center_2048"][0]),
                        torch.tensor(targets["center_2048"][1])),
        "radius_px":   torch.tensor(targets["radius_px"]),
        "filename":    targets["filename"],
    }

    model = NoduleDetector(variant=args.variant, head="centernet",
                           target_size=args.target_size).to(device)
    heatmap_size = model.heatmap_size
    sigma_min    = 2.5 * (heatmap_size / 128)   # matches trainer

    # Standard optimizer + loss, no scheduler.
    backbone_params = list(model.backbone_fpn.parameters())
    other_params    = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_factor},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scaler  = GradScaler(device=device.type, enabled=False)
    loss_fn = LOSS_REGISTRY.get("focal_v5")

    print(f"\nOverfit target: variant={args.variant}  n={len(picked)}  "
          f"heatmap={heatmap_size}  sigma_min={sigma_min:.2f}  "
          f"lr={args.lr}  backbone_lr={args.lr * args.backbone_lr_factor:.1e}")
    print(f"Success = loss -> ~0 AND train sens -> 1.0 within {args.epochs} epochs.\n")
    print(f"{'Ep':>4} | {'Loss':>8} | {'Train Sens':>10} | {'FP':>5}")
    print("-" * 40)

    # Dummy loader-shaped iterable for compute_detection_metrics_v4.
    single_batch = [(images.cpu(), batched_targets)]

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=torch.bfloat16,
                      enabled=device.type == "cuda"):
            hm_pred = model(images)
            hm_gt   = make_gt_heatmap(batched_targets, images.shape[0],
                                       heatmap_size, sigma_min=sigma_min,
                                       input_size=args.input_size, device=device)
            loss = loss_fn(hm_pred, hm_gt)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            metrics = compute_detection_metrics_v4(model, single_batch, device=device,
                                                   input_size=args.input_size)
            r = metrics.get("all", {}) or metrics.get("node21", {})
            sens = r.get("sensitivity", 0.0)
            fp   = r.get("fp_per_image", 0.0)
            marker = "  <-- SUCCESS" if loss.item() < 0.05 and sens >= 0.95 else ""
            print(f"{epoch:>4} | {loss.item():>8.4f} | {sens:>10.3f} | {fp:>5.2f}{marker}")
            model.train()

    print("\nInterpretation:")
    print("  * loss -> ~0 AND sens -> 1.0  => model works; plateau is generalization")
    print("  * loss stuck > 0.5           => structural bug (loss / gradient / capacity)")
    print("  * loss NaN                    => numerical stability bug")


if __name__ == "__main__":
    main()
