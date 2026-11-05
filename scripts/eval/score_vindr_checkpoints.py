"""
Re-score banked VinDr checkpoints and cache the RAW per-film scores.

WHY THIS EXISTS. metrics.json keeps only summary numbers: micro/macro AUC and F1, and
per-class AUC, F1, recall and precision at the single threshold fitted on validation. That
covers sensitivity (recall is the same quantity) but it cannot answer anything that needs
the underlying scores - specificity, any other operating point, PR or ROC curves, average
precision, or calibration. Those were never recoverable from what was written.

A forward pass over the 1,764-film test split is cheap (no backward, no optimiser), so the
fix is to run each checkpoint once and cache the (films x 14) score matrix. After that
every derived metric is a numpy operation and nothing needs the GPU again.

WHAT IS WRITTEN, per run, next to its metrics.json:
    test_scores.npz   scores (N, 14) float32, labels (N, 14) float32, film order fixed

The film order is the dataset's own, which is deterministic for a fixed split seed, so
matrices from different runs align row for row and can be compared or ensembled.

THE THRESHOLD IS NOT REFITTED HERE. Any operating point derived later must still be fitted
on validation and applied to test, exactly as the training script does. Caching test
scores makes it easy to accidentally tune on them; the derived-metric helper below takes a
threshold as an argument rather than searching for the best one.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.vindr import VinDrDataset, NUM_VINDR_CLASSES, VINDR_CLASSES, VINDR_ROOT
from src.data.vindr_split import build_vindr_multilabel_split
from src.models.detector import NoduleDetector

ANNOTATIONS = VINDR_ROOT / "train.csv"
IMAGES = VINDR_ROOT / "train_png16"


@torch.no_grad()
def score_split(model, loader, device):
    """Per-film, per-class confidence, the same quantity training selected on."""
    scores, labels = [], []

    for batch in loader:
        images, target = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) \
            else (batch["image"], batch)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images.to(device))

        confidence = (torch.sigmoid(output["classification"].float())
                      * torch.sigmoid(output["centerness"].float())).sqrt()

        scores.append(confidence.amax(dim=(2, 3)).cpu().numpy())
        labels.append(target["image_labels"].numpy())

    return np.concatenate(scores).astype(np.float32), np.concatenate(labels).astype(np.float32)


def derived_metrics(scores, labels, threshold):
    """
    Everything the stored summaries could not answer, at a GIVEN threshold.

    The threshold is an argument, never searched here - searching it on test scores would
    be reading the answer off the test set.

    Returns:
        dict with micro specificity, per-class specificity, average precision, and the
        counts the rest can be rebuilt from.
    """
    predicted = scores >= threshold

    true_positive = (predicted & (labels == 1)).sum(axis=0)
    false_positive = (predicted & (labels == 0)).sum(axis=0)
    true_negative = (~predicted & (labels == 0)).sum(axis=0)
    false_negative = (~predicted & (labels == 1)).sum(axis=0)

    def average_precision(class_scores, class_labels):
        """AP by the step-wise sum over recall increments, no interpolation."""
        order = np.argsort(-class_scores)
        hits = class_labels[order]
        positives = hits.sum()

        if positives == 0:
            return float("nan")

        cumulative = np.cumsum(hits)
        precision = cumulative / np.arange(1, len(hits) + 1)

        return float((precision * hits).sum() / positives)

    per_class = []

    for index, name in enumerate(VINDR_CLASSES):
        tp, fp = int(true_positive[index]), int(false_positive[index])
        tn, fn = int(true_negative[index]), int(false_negative[index])

        per_class.append({
            "name": name,
            "sensitivity": tp / max(tp + fn, 1),          # same quantity as recall
            "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
            "average_precision": average_precision(scores[:, index], labels[:, index]),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    tp, fp = int(true_positive.sum()), int(false_positive.sum())
    tn, fn = int(true_negative.sum()), int(false_negative.sum())

    return {
        "threshold": float(threshold),
        "micro_sensitivity": tp / max(tp + fn, 1),
        "micro_specificity": tn / max(tn + fp, 1),
        "micro_precision": tp / max(tp + fp, 1),
        "macro_average_precision": float(np.nanmean([c["average_precision"] for c in per_class])),
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-group", default="v23_vindr_3seed")
    parser.add_argument("--force", action="store_true",
                        help="re-score runs that already have test_scores.npz")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = []

    for path in (ROOT / "outputs" / "vindr").rglob("metrics.json"):
        metrics = json.load(open(path))

        if metrics.get("run_group") in arguments.run_group.split(","):
            run_dirs.append((path.parent, metrics))

    run_dirs.sort(key=lambda pair: (pair[1]["target_size"], pair[1]["variant"], pair[1]["seed"]))
    print("{} runs in {}".format(len(run_dirs), arguments.run_group))

    # One loader per input size, reused across every checkpoint that shares it
    loaders = {}

    for directory, metrics in run_dirs:
        output_path = directory / "test_scores.npz"

        if output_path.exists() and not arguments.force:
            print("  skip {} (already scored)".format(directory.name))
            continue

        input_size = metrics["input_size"]

        if input_size not in loaders:
            cache = VINDR_ROOT / "train_png16_cache_i{}".format(input_size)
            available = sorted(p.stem for p in cache.glob("*.npy"))
            splits = build_vindr_multilabel_split(ANNOTATIONS, seed=metrics["split_seed"],
                                                  available_ids=available)

            dataset = VinDrDataset(annotations_csv=ANNOTATIONS, images_root=IMAGES,
                                   image_ids=splits["test"], augment=False,
                                   input_size=input_size, clahe=True, multiclass=True,
                                   cache_dir=cache)

            loaders[input_size] = DataLoader(dataset, batch_size=arguments.batch_size,
                                             shuffle=False, num_workers=arguments.num_workers,
                                             pin_memory=True)

        model = NoduleDetector(variant=metrics["variant"], head="fcos",
                               backbone=metrics.get("backbone", "convnext_fpn"),
                               target_size=metrics["target_size"],
                               num_classes=NUM_VINDR_CLASSES).to(device)
        model.load_state_dict(torch.load(directory / "best_model.pth", map_location=device))
        model.eval()

        scores, labels = score_split(model, loaders[input_size], device)

        # Sanity: the recomputed micro-AUC must match what the run reported, or the
        # checkpoint, the split or the scoring path has drifted since training
        from src.evaluation.vindr_multilabel_metrics import evaluate
        recomputed = evaluate(scores, labels, threshold=metrics["fitted_threshold"])
        stored = metrics["test_results"]["micro_auc"]
        drift = abs(recomputed["micro_auc"] - stored)

        np.savez_compressed(output_path, scores=scores, labels=labels)

        flag = "" if drift < 1e-4 else "   <-- DRIFT {:.5f}, investigate".format(drift)
        print("  {:<52} microAUC {:.4f} vs stored {:.4f}{}".format(
            directory.name, recomputed["micro_auc"], stored, flag))

        del model
        torch.cuda.empty_cache()

    print("\nWrote test_scores.npz beside each metrics.json.")
    print("Derived metrics now need no GPU - call derived_metrics(scores, labels, threshold).")


if __name__ == "__main__":
    main()
