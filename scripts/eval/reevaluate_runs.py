"""
Re-evaluate finished runs from their checkpoints, refreshing box_metrics and froc_data.

WHY THIS EXISTS
    The FROC sampling was fixed on 2026-08-31: curves are now built on a fixed threshold
    grid instead of 25 points spaced evenly by RANK, which had left exactly one sample
    below 1 FP/img and made every number reported at 0.25 FP/img a chord across a gap.
    Runs evaluated before that date still carry the sparse curves and understate
    sensitivity by roughly 2x. Their checkpoints are fine, so nothing needs retraining -
    only the evaluation pass has to be redone.

WHY NOT recover_metrics.py
    That script reconstructs configuration by PARSING THE DIRECTORY NAME and writes a
    fresh metrics.json, which destroys `history` (trap 13.4). The training curves are
    exactly what the loss figures need. This script instead reads configuration from the
    existing metrics.json, which is authoritative, and merges the new results back so
    history, attribution, split_info and every recipe field survive untouched.

    The original metrics.json and froc_data.json are copied to *.pre_froc_fix.bak on first
    run, so nothing is destroyed.

NOTE ON BATCH SIZE
    Evaluation runs at a fixed batch of 8. Trap 13.1: bf16 evaluation is batch-size
    dependent, because accumulation order changes and can flip a borderline box. The
    measured magnitude is at most 0.002 at 0.25 FP/img, one twentieth of the noise floor,
    so a run trained at micro-batch 4 (Arm N) is not meaningfully affected.

Usage:
    python -m scripts.eval.reevaluate_runs --run-group v14 --dry-run
    python -m scripts.eval.reevaluate_runs --run-group v14
"""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.splits import build_three_way_split                                   # noqa: E402
from src.evaluation.node21_detection_metrics import (                               # noqa: E402
    compute_box_detection_metrics, froc_auc)
from src.models.detector import NoduleDetector                                      # noqa: E402
import scripts.train as T                                                           # noqa: E402

EVAL_BATCH = 8


def test_loader_for(input_size, batch_size=EVAL_BATCH):
    """NODE21-only test split at this input size, in fixed order."""
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=input_size, seed=42, test_seed=0,
        val_frac=0.20, group_by_image=True)

    return DataLoader(test_loaders["node21"], batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)


def reevaluate(run_dir: Path, loader, device):
    """
    Recompute box_metrics and froc_data for one run, preserving everything else.

    Args:
        run_dir: directory holding metrics.json and best_model.pth
        loader: NODE21 test DataLoader at the run's input size
        device: torch device

    Returns:
        (old sensitivity at 0.25 FP interpolated, new one) for reporting.
    """
    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path))

    model = NoduleDetector(variant=metrics["variant"], head=metrics["head"],
                           target_size=metrics["target_size"],
                           backbone=metrics.get("backbone", "convnext_fpn")).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    model.eval()

    # Same two IoU thresholds train.py reports, so the file shape is unchanged.
    box_metrics = {}

    for iou in (0.25, 0.5):
        computed = compute_box_detection_metrics(model, loader, device=device,
                                                 input_size=metrics["input_size"],
                                                 iou_threshold=iou)
        computed["froc_auc"] = froc_auc(computed["froc"])
        box_metrics[f"iou{iou}"] = computed

    del model
    torch.cuda.empty_cache()

    # Back up once, then merge. Only box_metrics and test_results are replaced; history,
    # attribution, split_info and every recipe field are left exactly as they were.
    backup = run_dir / "metrics.pre_froc_fix.bak"

    if not backup.exists():
        shutil.copy2(metrics_path, backup)

    old_curve = (metrics.get("box_metrics") or {}).get("iou0.25", {}).get("froc", [])
    metrics["box_metrics"] = box_metrics
    metrics["froc_sampling"] = "fixed_threshold_grid_2026-08-31"

    primary = box_metrics["iou0.25"]

    if isinstance(metrics.get("test_results"), dict):
        for key in ("combined", "node21"):
            if key in metrics["test_results"]:
                metrics["test_results"][key] = primary

    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=4)

    # froc_data.json mirrors the shape train.py writes for box heads.
    froc_path = run_dir / "froc_data.json"
    froc_backup = run_dir / "froc_data.pre_froc_fix.bak"

    if froc_path.exists() and not froc_backup.exists():
        shutil.copy2(froc_path, froc_backup)

    curve = primary["froc"]

    with open(froc_path, "w") as handle:
        json.dump({"thresholds": [p["threshold"] for p in curve],
                   "all": curve, "node21": curve, "jsrt": [],
                   "criterion": "box_iou_0.25"}, handle, indent=4)

    return old_curve, curve


def sensitivity_at(curve, budget=0.25):
    """Interpolated sensitivity at a budget, for before/after reporting only."""
    import numpy as np

    if not curve: return float("nan")

    fp = np.array([p["fp_per_image"] for p in curve])
    se = np.array([p["sensitivity"] for p in curve])
    order = np.argsort(fp)

    return float(np.interp(budget, fp[order], se[order]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-group", required=True,
                        help="substring of run_group to re-evaluate, e.g. v14")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = []

    for path in sorted(glob.glob("outputs/**/metrics.json", recursive=True)):
        try:
            metrics = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if arguments.run_group not in str(metrics.get("run_group", "")): continue
        if not (Path(path).parent / "best_model.pth").exists(): continue

        candidates.append((Path(path).parent, metrics))

    print(f"\n{len(candidates)} run(s) matching run_group~{arguments.run_group}\n")

    if arguments.dry_run:
        for run_dir, metrics in candidates:
            print(f"  would re-evaluate {run_dir.name}  "
                  f"(input {metrics['input_size']}, target {metrics['target_size']})")
        return

    loaders = {}

    for run_dir, metrics in candidates:
        input_size = metrics["input_size"]

        if input_size not in loaders:
            loaders[input_size] = test_loader_for(input_size)

        old_curve, new_curve = reevaluate(run_dir, loaders[input_size], device)

        old_below = sum(1 for p in old_curve if p["fp_per_image"] < 1.0)
        new_below = sum(1 for p in new_curve if p["fp_per_image"] < 1.0)

        print(f"  {run_dir.name}")
        print(f"      points below 1 FP/img : {old_below:3d} -> {new_below:3d}")
        print(f"      sens @0.25 FP         : {sensitivity_at(old_curve):.3f} -> "
              f"{sensitivity_at(new_curve):.3f}")

    print(f"\nOriginals preserved as metrics.pre_froc_fix.bak / froc_data.pre_froc_fix.bak\n")


if __name__ == "__main__":
    main()
