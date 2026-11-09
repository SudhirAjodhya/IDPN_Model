"""
Re-scores the NODE21 sweep on the full test set and on the test films that never appear
in training or validation, using the existing checkpoints.

Usage (from Model/):
    python -m scripts.eval.rescore_node21_clean
"""

import json
import statistics
import sys
from pathlib import Path

import torch
from scipy import stats
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval.compare_resizer_outputs import load_model                          # noqa: E402
from scripts.eval.plot_resizer_branches import find_run                              # noqa: E402
from scripts.eval.plot_supervisor_figures import CPM_BUDGETS, sensitivity_at_budget  # noqa: E402
from src.data.node21 import NODE21Dataset                                            # noqa: E402
from src.data.source_tagged import SourceTaggedDataset                               # noqa: E402
from src.data.splits import build_three_way_split                                    # noqa: E402
from src.evaluation.node21_detection_metrics import compute_box_detection_metrics     # noqa: E402
import scripts.train as T                                                            # noqa: E402

INPUT_SIZE = 1024
TARGETS = (224, 256, 384, 512, 768)
SEEDS = (42, 7, 123)
BILINEAR = "bilinear"
LEARNED = "node21_talebi_zeroinit"
ARMS = (BILINEAR, LEARNED)
METRICS = ("matched", "cpm", "ap")
SUBSETS = ("full", "clean")
OUTPUT_PATH = ROOT / "outputs" / "rescore" / "node21_clean.json"


def make_loader(dataset):
    """Unshuffled evaluation loader, kept identical for both subsets."""
    return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True,
                      persistent_workers=True)


def count_subset(rows):
    """Films, nodule-bearing films and nodules in a set of NODE21 annotation rows."""
    nodule_rows = rows[rows["label"] == 1]

    return {
        "films": int(rows["img_name"].nunique()),
        "nodule_films": int(nodule_rows["img_name"].nunique()),
        "nodules": int(len(nodule_rows)),
    }


def build_test_loaders(seed):
    """
    Build the full and leak-free NODE21 test loaders for one training seed.

    Args:
        seed: run seed, which sets the train/validation split (the test split is fixed)

    Returns:
        (full loader, clean loader, dict of film and nodule counts for both)
    """
    train_set, validation_set, test_sets, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=INPUT_SIZE, seed=seed, test_seed=0, val_frac=0.20,
        group_by_image=True)

    # Every film name the model could have seen during training or checkpoint selection
    seen_films = set(train_set.dataset.df["img_name"]) | set(validation_set.dataset.df["img_name"])

    full_rows = test_sets["node21"].dataset.df
    clean_rows = full_rows[~full_rows["img_name"].isin(seen_films)]

    clean_dataset = NODE21Dataset(clean_rows, T.NODE21_IMAGE_DIR, augment=False, input_size=INPUT_SIZE,
                                  group_by_image=True)
    clean_set = SourceTaggedDataset(clean_dataset, "node21")

    counts = {"full": count_subset(full_rows), "clean": count_subset(clean_rows)}

    return make_loader(test_sets["node21"]), make_loader(clean_set), counts


def summarise_detections(box_metrics):
    """Table II's three numbers from one compute_box_detection_metrics result."""
    curve = box_metrics["froc"]
    cpm = statistics.mean(sensitivity_at_budget(curve, budget) for budget in CPM_BUDGETS)

    return {
        "matched": sensitivity_at_budget(curve),
        "cpm": float(cpm),
        "ap": box_metrics["average_precision"],
    }


def summarise_subset(results, subset):
    """
    Per-target means for both arms, the paired difference, and the paired tests.

    Args:
        results: dict mapping (target, variant, seed) to {subset: metrics}
        subset: "full" or "clean"

    Returns:
        dict mapping each metric to its per-target table and paired statistics
    """
    summary = {}

    for metric in METRICS:
        per_target = {}
        differences = []
        decimations = []

        for target in TARGETS:
            bilinear = [results[(target, BILINEAR, seed)][subset][metric] for seed in SEEDS]
            learned = [results[(target, LEARNED, seed)][subset][metric] for seed in SEEDS]
            paired = [learned_value - bilinear_value for learned_value, bilinear_value in zip(learned, bilinear)]

            per_target[target] = {
                "bilinear": statistics.mean(bilinear),
                "bilinear_sd": statistics.stdev(bilinear),
                "learned": statistics.mean(learned),
                "learned_sd": statistics.stdev(learned),
                "difference": statistics.mean(paired),
            }

            differences.extend(paired)
            decimations.extend([INPUT_SIZE / target] * len(paired))

        t_test = stats.ttest_1samp(differences, 0.0)
        slope = stats.linregress(decimations, differences)

        summary[metric] = {
            "per_target": per_target,
            "mean_difference": statistics.mean(differences),
            "positive": sum(difference > 0 for difference in differences),
            "t": float(t_test.statistic),
            "p": float(t_test.pvalue),
            "slope_p": float(slope.pvalue),
        }

    return summary


def print_comparison(report):
    """Print each metric's full-set and clean-subset results side by side."""
    for metric in METRICS:
        old = report["full"][metric]
        new = report["clean"][metric]

        print("\n{}".format(metric))
        print("   paired difference: full {:+.4f} (p={:.3f}, {}/15 positive), clean {:+.4f} (p={:.3f}, {}/15 positive)".format(
            old["mean_difference"], old["p"], old["positive"], new["mean_difference"], new["p"], new["positive"]))

        for target in TARGETS:
            old_row = old["per_target"][target]
            new_row = new["per_target"][target]

            print("   {:>4}  bilinear {:.3f} -> {:.3f}   learned {:.3f} -> {:.3f}   difference {:+.3f} -> {:+.3f}".format(
                target, old_row["bilinear"], new_row["bilinear"], old_row["learned"], new_row["learned"],
                old_row["difference"], new_row["difference"]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    subset_counts = {}

    for seed in SEEDS:
        full_loader, clean_loader, counts = build_test_loaders(seed)
        subset_counts[seed] = counts
        print("seed {}: {}".format(seed, counts), flush=True)

        for target in TARGETS:
            for arm in ARMS:
                model, metrics = load_model(find_run(ROOT, arm, target, seed), device)

                full = summarise_detections(compute_box_detection_metrics(model, full_loader, device, INPUT_SIZE))
                clean = summarise_detections(compute_box_detection_metrics(model, clean_loader, device, INPUT_SIZE))

                # The full-set rescore must reproduce the sensitivity the run reported during training
                stored = sensitivity_at_budget(metrics["box_metrics"]["iou0.25"]["froc"])
                drift = abs(full["matched"] - stored)
                warning = "" if drift < 1e-3 else "   <-- DRIFT {:.4f}".format(drift)

                results[(target, arm, seed)] = {"full": full, "clean": clean, "stored_matched": stored}
                print("  t{:<4} {:<24} full {:.3f} (stored {:.3f}){}   clean {:.3f}".format(
                    target, arm, full["matched"], stored, warning, clean["matched"]), flush=True)

                del model
                torch.cuda.empty_cache()

    report = {subset: summarise_subset(results, subset) for subset in SUBSETS}
    print_comparison(report)

    # Saving per-run results and the summary for the report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs_by_name = {"t{}_{}_s{}".format(*key): value for key, value in results.items()}
    output = {"subset_counts": subset_counts, "runs": runs_by_name, "summary": report}
    json.dump(output, open(OUTPUT_PATH, "w"), indent=1)

    print("\nWrote {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()
