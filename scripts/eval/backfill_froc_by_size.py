"""
Backfill `froc_by_size` into existing metrics.json files.

`by_size` was only ever recorded at a fixed score_threshold=0.3, which is trap 4.1 in
docs/context/HANDOVER.md: runs sit anywhere from 0.2 to 1.1 FP/img at that threshold, so
per-bin sensitivities are read at different operating points and are not comparable. The
per-bin FROC curve added to src/evaluation/node21_detection_metrics.py fixes that, but it
only exists for runs evaluated after the change. This script recomputes it for runs already
on disk, from their saved checkpoints.

MERGE ONLY. Every other key in metrics.json - history, best_val_sensitivity, attribution,
split_info - is read, preserved and written back untouched. Nothing is retrained and no
existing number is overwritten. Unlike recover_metrics.py, which rebuilds metrics.json from
scratch and is only safe on runs that have none.

As a free regression check, the recomputed pooled sensitivity is compared against the stored
one and nothing is written on a mismatch.

--batch-size MUST match what the run trained at, which is 8 for every fcos run in this
project (confirmed from outputs/mlflow/mlruns.db). Evaluation runs under bfloat16 autocast,
so a different batch size changes accumulation order and can flip a single borderline
nodule. This is not covered by trap 4.7: "deterministic at fixed seed" was established for
full training reruns under identical config, and says nothing about batch-size invariance
of the eval pass. The first attempt here ran at 4 and shifted 5 of 29 runs by exactly 1/147.

Usage:
    python scripts/eval/backfill_froc_by_size.py --dry-run
    python scripts/eval/backfill_froc_by_size.py
    python scripts/eval/backfill_froc_by_size.py --head centernet_det --batch-size 8 --force
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.data.splits import build_three_way_split
from src.evaluation.node21_detection_metrics import compute_box_detection_metrics, froc_auc
from src.models.detector import NoduleDetector

# Mirrors scripts/train.py lines 39-44. Duplicated rather than imported because train.py
# pulls in mlflow at module scope, which this read-only script has no use for.
ROOT_DIR         = Path(__file__).resolve().parents[2]
DATA_DIR         = ROOT_DIR / "data_sets"
JSRT_IMAGE_DIR   = DATA_DIR / "JSRT" / "images" / "images"
JSRT_CSV_PATH    = DATA_DIR / "JSRT" / "jsrt_metadata.csv"
NODE21_IMAGE_DIR = DATA_DIR / "node21" / "cxr_images" / "original_data" / "images"
NODE21_CSV_PATH  = DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv"

# Recomputation is deterministic, so anything above float noise is a real behaviour change
SENSITIVITY_TOLERANCE = 1e-6


def needs_backfill(run, force):
    """True when this run has box_metrics but no per-bin FROC in them."""
    box_metrics = run.get("box_metrics")

    if not box_metrics: return False
    if force: return True

    return any("froc_by_size" not in m for m in box_metrics.values())


def rebuild(run, run_dir, device, batch_size, num_workers):
    """
    Recompute box_metrics from the saved checkpoint. Returns the new dict.

    `batch_size` must match the one the run was originally evaluated with. train.py:309
    evaluates through test_dls["combined"], built at args.batch_size, and
    compute_box_detection_metrics runs under bfloat16 autocast, where batch composition
    changes accumulation order. That is enough to move a box across the IoU or score
    boundary and flip a single nodule. Recomputing 29 fcos runs at 4 instead of their
    actual 8 reproduced 24 exactly and shifted 5 by exactly 1/147.
    """
    model = NoduleDetector(variant=run["variant"], head=run["head"],
                           target_size=run["target_size"],
                           backbone=run.get("backbone", "convnext_fpn")).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device))
    model.eval()

    # Test split is pinned by test_seed=0 and does not depend on the training seed,
    # but the loader still needs the run's own resolution and letterbox setting
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=JSRT_CSV_PATH, jsrt_image_dir=JSRT_IMAGE_DIR,
        node21_csv_path=NODE21_CSV_PATH, node21_image_dir=NODE21_IMAGE_DIR,
        seed=run["seed"], test_seed=run.get("test_seed", 0),
        target_size=run["target_size"], augment_train=False, datasets="node21",
        input_size=run["input_size"], val_frac=0.20,
        aug_profile=run.get("aug_profile", "v3"), group_by_image=True,
        letterbox=bool(run.get("letterbox")))

    # Workers are safe to raise: shuffle=False with a fixed batch size means they prefetch
    # in order and batch composition is unchanged, so no number moves. Batch size is NOT
    # safe to raise - see the docstring above.
    loader = DataLoader(test_loaders["node21"], batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)

    rebuilt = {}

    for iou in (0.25, 0.5):
        metrics = compute_box_detection_metrics(model, loader, device=device,
                                                input_size=run["input_size"], iou_threshold=iou)
        metrics["froc_auc"] = froc_auc(metrics["froc"])
        rebuilt[f"iou{iou}"] = metrics

    return rebuilt


def process(metrics_path, device, force, dry_run, batch_size, num_workers):
    run_dir = os.path.dirname(metrics_path)
    name = os.path.basename(run_dir)

    run = json.load(open(metrics_path))

    if not os.path.exists(os.path.join(run_dir, "best_model.pth")):
        return f"SKIP  {name}: no best_model.pth, cannot recompute"

    if dry_run:
        return f"WOULD BACKFILL  {name}: {run['variant']} s{run['seed']} " \
               f"{run['input_size']}->{run['target_size']}"

    rebuilt = rebuild(run, run_dir, device, batch_size, num_workers)

    # Regression check before writing anything: deterministic pipeline must reproduce
    for key, metrics in rebuilt.items():
        stored = run["box_metrics"].get(key)

        if stored is None: continue

        drift = abs(metrics["sensitivity"] - stored["sensitivity"])

        if drift > SENSITIVITY_TOLERANCE:
            return (f"FAILURE  {name}: {key} sensitivity moved {stored['sensitivity']:.6f} -> "
                    f"{metrics['sensitivity']:.6f} (drift {drift:.2e}, "
                    f"{round(drift * 147):+d} nodules) at batch {batch_size}. NOTHING "
                    f"WRITTEN, so any froc_by_size already present is KEPT AND MAY BE "
                    f"STALE - check its froc_by_size_batch_size. This run's group is "
                    f"{run.get('run_group')}; v7_recipe_and_fcos trained at batch 4, "
                    f"v8/v9 at 8.")

    # Merge only the new key, leaving every stored number and the training history intact.
    # The batch size is stamped alongside because a curve is only valid for the batch that
    # produced it, and a refusal leaves the previous curve in place with no other marker.
    for key, metrics in rebuilt.items():
        if key in run["box_metrics"]:
            run["box_metrics"][key]["froc_by_size"] = metrics["froc_by_size"]
            run["box_metrics"][key]["froc_by_size_batch_size"] = batch_size
        else:
            metrics["froc_by_size_batch_size"] = batch_size
            run["box_metrics"][key] = metrics

    with open(metrics_path, "w") as f:
        json.dump(run, f, indent=4)

    curves = run["box_metrics"]["iou0.25"]["froc_by_size"]
    populated = [b for b, c in curves.items() if c]

    return f"OK    {name}: froc_by_size written for bins {populated}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head",      type=str, default="fcos",
                        help="Restrict to one head. Pass 'all' for every head.")
    parser.add_argument("--run-group", type=str, default=None,
                        help="Restrict to runs whose metrics.json records this run_group.")
    parser.add_argument("--only",      type=str, default=None,
                        help="Comma-separated EXACT run directory names. Exact, not substring: "
                             "a substring like _s42_i1024_t384 matches every variant sharing "
                             "that suffix, which has already caused eight runs to be recomputed "
                             "at the wrong batch size.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Evaluation batch size. MUST match the run's training batch "
                             "size or bf16 accumulation order shifts borderline boxes. "
                             "Every fcos run in this project used 8.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Safe to raise (does not change numerics); "
                             "4 suits an 8-core CPU. Use 0 if Windows worker spawn misbehaves.")
    parser.add_argument("--force",     action="store_true",
                        help="Recompute even where froc_by_size already exists.")
    parser.add_argument("--dry-run",   action="store_true")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = []

    for metrics_path in sorted(glob.glob("outputs/**/metrics.json", recursive=True)):
        try:
            run = json.load(open(metrics_path))
        except (json.JSONDecodeError, OSError):
            continue

        if arguments.head != "all" and run.get("head") != arguments.head: continue
        if arguments.run_group and run.get("run_group") != arguments.run_group: continue
        if arguments.only and os.path.basename(os.path.dirname(metrics_path)) not in                 [name.strip() for name in arguments.only.split(",")]: continue
        if not needs_backfill(run, arguments.force): continue

        candidates.append(metrics_path)

    if not candidates:
        print(f"nothing to backfill (head={arguments.head}) - every matching run already "
              f"has froc_by_size")
        return

    print(f"{len(candidates)} run(s) to backfill on {device}, roughly 40s each "
          f"(~{len(candidates) * 40 / 60:.0f} min)\n")

    failures = 0
    started = time.time()

    for i, metrics_path in enumerate(candidates, 1):
        result = process(metrics_path, device, arguments.force, arguments.dry_run,
                         arguments.batch_size, arguments.num_workers)

        if result.startswith("FAILURE"): failures += 1

        print(f"  [{i}/{len(candidates)}] {result}")

    print(f"\ndone in {(time.time() - started) / 60:.1f} min, {failures} failure(s)")

    if failures:
        print("A failure means recomputation did not reproduce the stored sensitivity.")
        print("Check --batch-size first: it must match what the run trained at (8 for every")
        print("fcos run here). Under bf16 autocast a different batch size shifts accumulation")
        print("order and can flip one borderline nodule. If the batch size already matches,")
        print("the drift is real and worth investigating before anything is written.")


if __name__ == "__main__":
    main()
