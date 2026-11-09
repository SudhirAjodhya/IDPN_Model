"""
Re-scores the VinDr sweep with image labels taken from all fused boxes, instead of only
the 16 boxes per film the dataset keeps, using the cached test scores of each run.

Usage (from Model/):
    python -m scripts.eval.rescore_vindr_uncapped
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval.score_vindr_checkpoints import derived_metrics                     # noqa: E402
from src.data.vindr import MAX_OBJECTS, NUM_VINDR_CLASSES, VINDR_CLASSES, VINDR_ROOT  # noqa: E402
from src.data.vindr import VinDrDataset                                              # noqa: E402
from src.data.vindr_split import build_vindr_multilabel_split                        # noqa: E402
from src.evaluation.vindr_multilabel_metrics import evaluate                          # noqa: E402

RUN_GROUP = "v23_vindr_3seed"
ANNOTATIONS = VINDR_ROOT / "train.csv"
IMAGES = VINDR_ROOT / "train_png16"
OUTPUT_PATH = ROOT / "outputs" / "rescore" / "vindr_uncapped.json"

BILINEAR = "bilinear"
LEARNED = "node21_talebi_zeroinit"
METRICS = ("micro_auc", "macro_auc", "macro_ap", "micro_f1")
LABEL_SETS = ("capped", "uncapped")


def find_runs():
    """
    Collect every run in the VinDr sweep.

    Returns:
        dict mapping (target size, variant, seed) to (run directory, metrics dict)
    """
    runs = {}

    for metrics_path in (ROOT / "outputs" / "vindr").rglob("metrics.json"):
        metrics = json.load(open(metrics_path))

        if metrics.get("run_group") != RUN_GROUP: continue

        key = (int(metrics["target_size"]), metrics["variant"], int(metrics["seed"]))
        runs[key] = (metrics_path.parent, metrics)

    return runs


def build_label_matrices(input_size, split_seed):
    """
    Build the capped and uncapped labels for the test split, in dataset order.

    Args:
        input_size: input resolution, whose cache decides which films are available
        split_seed: the split seed the runs were trained with

    Returns:
        (capped, uncapped), each a (films, 14) float32 multi-hot array
    """
    cache_dir = VINDR_ROOT / "train_png16_cache_i{}".format(input_size)
    available_films = sorted(path.stem for path in cache_dir.glob("*.npy"))
    splits = build_vindr_multilabel_split(ANNOTATIONS, seed=split_seed, available_ids=available_films)

    # Building the dataset only reads annotations; no pixels are loaded here
    dataset = VinDrDataset(annotations_csv=ANNOTATIONS, images_root=IMAGES, image_ids=splits["test"],
                           augment=False, input_size=input_size, clahe=True, multiclass=True,
                           cache_dir=cache_dir)

    capped = np.zeros((len(dataset.records), NUM_VINDR_CLASSES), dtype=np.float32)
    uncapped = np.zeros_like(capped)

    # Each record holds every fused box; the dataset keeps only the first MAX_OBJECTS of them
    for row, (_, _, box_classes) in enumerate(dataset.records):
        uncapped[row, box_classes] = 1.0
        capped[row, box_classes[:MAX_OBJECTS]] = 1.0

    return capped, uncapped


def score_run(scores, labels, threshold):
    """
    The four report metrics for one run against one label set.

    Returns:
        dict with micro_auc, macro_auc, macro_ap, micro_f1 and per-class AUC
    """
    summary = evaluate(scores, labels, threshold=threshold)
    macro_ap = derived_metrics(scores, labels, threshold)["macro_average_precision"]

    return {
        "micro_auc": summary["micro_auc"],
        "macro_auc": summary["macro_auc"],
        "macro_ap": macro_ap,
        "micro_f1": summary["micro_f1"],
        "per_class_auc": {entry["name"]: entry["auc"] for entry in summary["per_class"]},
    }


def summarise_label_set(results, label_set):
    """
    Per-target means for both arms, the paired difference, and the paired tests.

    Args:
        results: dict mapping (target, variant, seed) to {label set: metrics}
        label_set: "capped" or "uncapped"

    Returns:
        dict mapping each metric to its per-target table and paired statistics
    """
    targets = sorted({target for target, _, _ in results})
    seeds = sorted({seed for _, _, seed in results})
    summary = {}

    for metric in METRICS:
        per_target = {}
        differences = []
        decimations = []

        for target in targets:
            bilinear = [results[(target, BILINEAR, seed)][label_set][metric] for seed in seeds]
            learned = [results[(target, LEARNED, seed)][label_set][metric] for seed in seeds]
            paired = [learned_value - bilinear_value for learned_value, bilinear_value in zip(learned, bilinear)]

            per_target[target] = {
                "bilinear": statistics.mean(bilinear),
                "learned": statistics.mean(learned),
                "difference": statistics.mean(paired),
            }

            differences.extend(paired)
            decimations.extend([2048 / target] * len(paired))

        t_test = stats.ttest_1samp(differences, 0.0)
        slope = stats.linregress(decimations, differences)

        summary[metric] = {
            "per_target": per_target,
            "mean_difference": statistics.mean(differences),
            "t": float(t_test.statistic),
            "p": float(t_test.pvalue),
            "slope": float(slope.slope),
            "slope_p": float(slope.pvalue),
        }

    return summary


def print_label_changes(capped, uncapped):
    """Report how many test films, and which classes, gain a label without the cap."""
    changed_films = int((capped != uncapped).any(axis=1).sum())
    print("test films {}, films whose labels change without the cap {}".format(len(capped), changed_films))

    for class_index, class_name in enumerate(VINDR_CLASSES):
        gained = int(uncapped[:, class_index].sum() - capped[:, class_index].sum())

        if gained: print("   {:<22} +{} positive test films".format(class_name, gained))

    return changed_films


def print_comparison(report):
    """Print each metric's capped and uncapped results side by side."""
    for metric in METRICS:
        old = report["capped"][metric]
        new = report["uncapped"][metric]

        print("\n{}".format(metric))
        print("   paired difference: capped {:+.4f} (p={:.3f}), uncapped {:+.4f} (p={:.3f})".format(
            old["mean_difference"], old["p"], new["mean_difference"], new["p"]))
        print("   slope p: capped {:.3f}, uncapped {:.3f}".format(old["slope_p"], new["slope_p"]))

        for target, old_row in old["per_target"].items():
            new_row = new["per_target"][target]

            print("   {:>5}  bilinear {:.4f} -> {:.4f}   learned {:.4f} -> {:.4f}   difference {:+.4f} -> {:+.4f}".format(
                target, old_row["bilinear"], new_row["bilinear"], old_row["learned"], new_row["learned"],
                old_row["difference"], new_row["difference"]))


def main():
    runs = find_runs()
    print("{} runs in {}".format(len(runs), RUN_GROUP))

    # Every run must share one input size and one split, or the label matrices would not align
    input_sizes = {int(metrics["input_size"]) for _, metrics in runs.values()}
    split_seeds = {int(metrics["split_seed"]) for _, metrics in runs.values()}
    assert len(input_sizes) == 1 and len(split_seeds) == 1, (input_sizes, split_seeds)

    capped, uncapped = build_label_matrices(input_sizes.pop(), split_seeds.pop())
    changed_films = print_label_changes(capped, uncapped)

    results = {}

    for key, (run_dir, metrics) in sorted(runs.items()):
        cached = np.load(run_dir / "test_scores.npz")

        # The cached labels must equal the rebuilt capped labels row for row, or film order has drifted
        assert np.array_equal(cached["labels"], capped), "film order or labels drifted for {}".format(run_dir)

        threshold = metrics["fitted_threshold"]
        results[key] = {
            "capped": score_run(cached["scores"], capped, threshold),
            "uncapped": score_run(cached["scores"], uncapped, threshold),
        }

        # The capped rescore must reproduce what the run reported during training
        stored_micro_auc = metrics["test_results"]["micro_auc"]
        rescored_micro_auc = results[key]["capped"]["micro_auc"]
        assert abs(rescored_micro_auc - stored_micro_auc) < 1e-4, (rescored_micro_auc, stored_micro_auc)

    report = {label_set: summarise_label_set(results, label_set) for label_set in LABEL_SETS}
    print_comparison(report)

    # Saving per-run results and the summary for the report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs_by_name = {"{}_{}_s{}".format(*key): value for key, value in results.items()}
    output = {"changed_films": changed_films, "runs": runs_by_name, "summary": report}
    json.dump(output, open(OUTPUT_PATH, "w"), indent=1)

    print("\nWrote {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
