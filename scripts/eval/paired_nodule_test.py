"""
Paired per-nodule comparison of two runs at a matched false-positive budget.

WHY THIS EXISTS
    Every comparison in this project has been judged against the binomial standard error
    of a SINGLE proportion, +-0.041 at n=147. That is the wrong yardstick. Both arms are
    evaluated on the SAME 147 nodules, so the comparison is PAIRED, and for paired binary
    outcomes the variance of the difference depends only on the DISCORDANT nodules - those
    one model finds and the other misses. Concordant nodules carry no information about
    which model is better.

    Using the marginal SE on a paired difference is over-conservative and can hide a real
    effect. McNemar's exact test is the correct instrument.

WHAT IT DOES NOT FIX
    This addresses test-set sampling variance only. Seed-to-seed variance from the
    train/val split and initialisation is a SEPARATE component (~0.020 on fcos) and is not
    captured here. A significant McNemar result on one seed still needs seed replication
    before it means the architecture is better.

METHOD
    Each model is evaluated at ITS OWN score threshold that yields the target FP budget,
    read from the run's froc_data.json. Comparing at a fixed threshold instead would
    reintroduce trap 4.1, where a variant looks better only because it sits at a different
    operating point.

    Matching reuses `_match` and `_truth_boxes_for` from the production metrics module, so
    the hit/miss decision is identical to the one behind the reported sensitivities rather
    than a second implementation that could disagree.

Usage:
    python -m scripts.eval.paired_nodule_test --a <run_dir_a> --b <run_dir_b>
    python -m scripts.eval.paired_nodule_test --a ... --b ... --budget 0.25
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.splits import build_three_way_split                      # noqa: E402
from src.evaluation.fcos_targets import decode_fcos                    # noqa: E402
from src.evaluation.centernet_targets import decode_detections         # noqa: E402
from src.evaluation.node21_detection_metrics import _match, _truth_boxes_for  # noqa: E402
from src.models.detector import NoduleDetector                         # noqa: E402

import scripts.train as T                                              # noqa: E402


def evaluate_at_threshold(cached, score_threshold: float) -> tuple[np.ndarray, float]:
    """
    Re-match cached predictions at one score threshold.

    Args:
        cached: list of (predicted_boxes, predicted_scores, truth_boxes) per image
        score_threshold: score floor for a prediction to count

    Returns:
        (per-nodule hit vector, false positives per image)
    """
    hits, false_positives = [], 0

    for boxes, scores, truth in cached:
        claimed, n_false = _match(boxes, scores, truth,
                                  iou_threshold=0.25, score_threshold=score_threshold)
        false_positives += n_false
        hits.extend(j in claimed for j in range(len(truth)))

    return np.array(hits, dtype=bool), false_positives / len(cached)


def threshold_for_budget(cached, budget: float, tolerance: float = 0.002) -> float:
    """
    Find, by bisection, the score threshold that actually yields `budget` FP/img.

    For any run evaluated BEFORE 2026-08-31 this MUST be measured rather than interpolated
    from the stored FROC curve. Those curves sampled 25 points by RANK, which left them
    desperately sparse at the operating point: on the v14 runs the curve jumps straight
    from (threshold 0.874, 0.000 FP/img) to (threshold 0.303, 0.652 FP/img) with NOTHING
    between, and the 0.25 budget falls inside that gap. Interpolating there invents a
    number, exactly as trap 13.5 describes for precision.

    Runs evaluated after that date use a fixed threshold grid and are dense enough to read
    directly, but bisection is kept as the default because it is correct either way and
    costs nothing once predictions are cached.

    Args:
        cached: list of (predicted_boxes, predicted_scores, truth_boxes) per image
        budget: target false positives per image
        tolerance: acceptable FP/img error before stopping

    Returns:
        The score threshold achieving the budget.
    """
    low, high = 0.0, 1.0

    for _ in range(40):
        middle = (low + high) / 2
        _, fp_per_image = evaluate_at_threshold(cached, middle)

        if abs(fp_per_image - budget) < tolerance:
            return middle

        # Higher threshold means fewer predictions, so fewer false positives.
        if fp_per_image > budget:
            low = middle
        else:
            high = middle

    return (low + high) / 2


@torch.no_grad()
def nodule_hits(run_dir: Path, loader, device, budget: float) -> tuple[np.ndarray, float]:
    """
    Per-nodule hit vector for one run, evaluated at its matched-budget threshold.

    Args:
        run_dir: run directory holding metrics.json and best_model.pth
        loader: NODE21 test DataLoader
        device: torch device
        budget: FP/img to match at

    Returns:
        (hits, threshold) where hits is a bool array over every GT nodule in loader order.
    """
    metrics = json.load(open(run_dir / "metrics.json"))
    input_size = metrics["input_size"]

    model = NoduleDetector(variant=metrics["variant"], head=metrics["head"],
                           target_size=metrics["target_size"],
                           backbone=metrics.get("backbone", "convnext_fpn")).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    model.eval()

    # Inference runs ONCE; the threshold search then re-matches these cached predictions,
    # so bisection costs no extra GPU work. Every image is kept, including nodule-free
    # ones, because they contribute false positives and so determine the budget.
    cached = []

    for images, targets in loader:
        images = images.to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16,
                      enabled=device.type == "cuda"):
            output = model(images)

        if getattr(model, "is_fcos", False):
            boxes_batch, scores_batch = decode_fcos(
                output["classification"].float(), output["ltrb"].float(),
                output["centerness"].float(), input_size=input_size, top_k=20)
        else:
            boxes_batch, scores_batch = decode_detections(
                output["heatmap"].float(), output["size_wh"].float(),
                output["offset"].float(), input_size=input_size, top_k=20)

        boxes_batch, scores_batch = boxes_batch.cpu(), scores_batch.cpu()

        for i in range(images.shape[0]):
            truth, _ = _truth_boxes_for(targets, i, input_size)
            cached.append((boxes_batch[i], scores_batch[i], truth))

    del model
    torch.cuda.empty_cache()

    score_threshold = threshold_for_budget(cached, budget)
    hits, achieved = evaluate_at_threshold(cached, score_threshold)

    return hits, score_threshold, achieved


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """
    Two-sided exact McNemar p-value on the discordant counts.

    Under the null the discordant nodules split 50/50, so the count is Binomial(n, 0.5).
    The exact test is used rather than the chi-square approximation because discordant
    counts here are small (single digits to low tens), where the approximation is poor.

    Args:
        only_a: nodules found by A and missed by B
        only_b: nodules found by B and missed by A

    Returns:
        Two-sided p-value.
    """
    n = only_a + only_b

    if n == 0:
        return 1.0

    smaller = min(only_a, only_b)
    tail = sum(comb(n, k) for k in range(smaller + 1)) / (2 ** n)

    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, type=Path, help="first run directory")
    parser.add_argument("--b", required=True, type=Path, help="second run directory")
    parser.add_argument("--budget", type=float, default=0.25, help="FP/img to match at")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="keep at the training value; trap 13.1 makes bf16 evaluation "
                             "batch-size dependent")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = json.load(open(arguments.a / "metrics.json"))["input_size"]

    # NODE21-only test split, identical for every arm. shuffle=False so the two runs see
    # nodules in the same order and the hit vectors line up elementwise.
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=input_size, seed=42, test_seed=0,
        val_frac=0.20, group_by_image=True)

    loader = DataLoader(test_loaders["node21"], batch_size=arguments.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    hits_a, threshold_a, fp_a = nodule_hits(arguments.a, loader, device, arguments.budget)
    hits_b, threshold_b, fp_b = nodule_hits(arguments.b, loader, device, arguments.budget)

    assert len(hits_a) == len(hits_b), "hit vectors must align nodule-for-nodule"

    n = len(hits_a)
    only_a = int((hits_a & ~hits_b).sum())
    only_b = int((hits_b & ~hits_a).sum())
    both = int((hits_a & hits_b).sum())
    neither = int((~hits_a & ~hits_b).sum())

    print(f"\nPaired per-nodule comparison at {arguments.budget} FP/img, n = {n} nodules")
    print(f"  A: {arguments.a.name}")
    print(f"     threshold {threshold_a:.4f} -> {fp_a:.3f} FP/img (measured, not interpolated)")
    print(f"  B: {arguments.b.name}")
    print(f"     threshold {threshold_b:.4f} -> {fp_b:.3f} FP/img (measured, not interpolated)")
    print(f"\n  sensitivity A {hits_a.mean():.3f}   B {hits_b.mean():.3f}   "
          f"difference {hits_b.mean() - hits_a.mean():+.3f}")

    print(f"\n  contingency          B hit   B miss")
    print(f"    A hit             {both:6d}   {only_a:6d}")
    print(f"    A miss            {only_b:6d}   {neither:6d}")

    discordant = only_a + only_b
    p_value = mcnemar_exact(only_a, only_b)

    print(f"\n  discordant nodules: {discordant}  ({only_b} favour B, {only_a} favour A)")
    print(f"  McNemar exact two-sided p = {p_value:.4f}")

    # The paired SE, for contrast with the marginal +-0.041 the project has been quoting.
    paired_se = (discordant ** 0.5) / n if discordant else 0.0
    marginal_se = (hits_a.mean() * (1 - hits_a.mean()) / n) ** 0.5

    print(f"\n  paired SE of the difference : +-{paired_se:.4f}   <- correct for this comparison")
    print(f"  marginal SE of one arm      : +-{marginal_se:.4f}   <- what has been quoted")

    verdict = "SIGNIFICANT at 0.05" if p_value < 0.05 else "not significant at 0.05"
    print(f"\n  verdict: {verdict}")
    print("  NOTE: this covers test-set sampling only. Seed variance (~0.020 on fcos) is a")
    print("        separate component and still requires replication.\n")


if __name__ == "__main__":
    main()
