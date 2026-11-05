"""
Print every number in the report's tables, as LaTeX rows, from what is on disk.

  Table I    parameter counts, from a freshly built detector
  Table II   NODE21 sensitivity at a matched 0.25 FP/img, CPM and AP, mean +/- sample SD
  Table III  VinDr-CXR micro-AUC, macro-AUC, macro-AP and micro-F1, mean +/- sample SD, with the
             paired statistics quoted in the Results text

NODE21 sensitivity is NOT the `sensitivity` field in metrics.json. That field is read at
a fixed score threshold of 0.3; the reported quantity is the FROC curve interpolated at
0.25 FP/img, which `plot_supervisor_figures.load` computes. Reusing it here keeps the
table and the resolution-sweep figure on one definition.

Macro-AP needs the cached per-film scores (`test_scores.npz`), so run
`score_vindr_checkpoints` first.

Usage (from Model/):
    python -m scripts.eval.report_tables
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval.plot_supervisor_figures import load                   # noqa: E402
from scripts.eval.score_vindr_checkpoints import derived_metrics         # noqa: E402

NODE21_RECIPE = "node21_vindr+vin2048+vnr1+sel-val_sens_matched"
NODE21_GROUPS = "v15_selection_and_scale,v19_focal_norm_fix"
NODE21_INPUT = 1024

VINDR_GROUP = "v23_vindr_3seed"
VINDR_INPUT = 2048

BASELINE, LEARNED = "bilinear", "node21_talebi_zeroinit"


def mean_sd(values):
    """Mean and sample standard deviation, as used throughout the report."""
    return statistics.mean(values), statistics.stdev(values)


def parameter_table():
    """Table I: parameter count per top-level module, for both arms at target 224."""
    from src.models.detector import NoduleDetector

    print("\nTable I - parameter counts")

    for variant in (LEARNED, BASELINE):
        model = NoduleDetector(variant=variant, head="fcos", target_size=224,
                               backbone="convnext_fpn")
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())

        parts = {name: count(child) for name, child in model.named_children()}
        print("  {:<24} total {:>12,}   {}".format(
            variant, count(model), "  ".join("{} {:,}".format(k, v) for k, v in parts.items())))


def arm_pair(baseline, learned, digits):
    """Both arms as mean +/- SD LaTeX cells, with the higher mean in bold."""
    (b_mean, b_sd), (l_mean, l_sd) = mean_sd(baseline), mean_sd(learned)
    cells = ["{:.{d}f} $\\pm$ {:.{d}f}".format(b_mean, b_sd, d=digits),
             "{:.{d}f} $\\pm$ {:.{d}f}".format(l_mean, l_sd, d=digits)]

    # Ties at the printed precision stay unbolded
    if round(b_mean, digits) != round(l_mean, digits):
        best = int(l_mean > b_mean)
        mean_text, rest = cells[best].split(" $\\pm$")
        cells[best] = "\\textbf{" + mean_text + "} $\\pm$" + rest

    return " & ".join(cells)


def paired_statistics(deltas, ratios, metric):
    """Paired t-test on learned minus bilinear, and its slope against decimation."""
    test = stats.ttest_1samp(deltas, 0.0)
    slope = stats.linregress(ratios, deltas)

    print("  {:<10} mean {:+.4f}  t = {:.2f}  p = {:.3f}  positive {}/{}  |  "
          "slope vs decimation {:+.5f}/unit  r = {:+.3f}  p = {:.4f}".format(
              metric, np.mean(deltas), test.statistic, test.pvalue,
              sum(delta > 0 for delta in deltas), len(deltas),
              slope.slope, slope.rvalue, slope.pvalue))


def node21_table(node21_groups):
    """Table II: matched-FP sensitivity, CPM and AP per output resolution."""
    runs = load("fcos", recipe=NODE21_RECIPE, run_group=node21_groups, input_size=NODE21_INPUT)
    metrics = ("matched", "cpm", "ap")

    print("\nTable II - NODE21 sensitivity at 0.25 FP/img, CPM, AP")

    for target in sorted({key[1] for key in runs}):
        baseline, learned = runs.get((BASELINE, target), {}), runs.get((LEARNED, target), {})

        if len(baseline) < 2 or len(learned) < 2:
            print("  {:<5} incomplete: {} bilinear, {} learned seeds".format(
                target, len(baseline), len(learned)))
            continue

        row = ["{:<5} & {:.2f}$\\times$".format(target, NODE21_INPUT / target)]
        row += [arm_pair([v[metric] for v in baseline.values()],
                         [v[metric] for v in learned.values()], 3) for metric in metrics]
        print(" & ".join(row) + " \\\\")

    # Paired within seed over the decimated targets only, matching the table's rows. At
    # target = input there is no decimation and bilinear fails to train the 256 grid.
    decimated = sorted(t for (_, t) in runs if t < NODE21_INPUT)
    print("\nPaired statistics over decimated targets {}".format(sorted(set(decimated))))

    for metric in metrics:
        deltas, ratios = [], []

        for target in sorted(set(decimated)):
            for seed, entry in runs[(LEARNED, target)].items():
                deltas.append(entry[metric] - runs[(BASELINE, target)][seed][metric])
                ratios.append(NODE21_INPUT / target)

        paired_statistics(deltas, ratios, metric)


def vindr_cells(vindr_group):
    """(target, seed, variant) -> the Table III metrics for every scored VinDr run."""
    cells = {}

    for path in (ROOT / "outputs" / "vindr").rglob("metrics.json"):
        metrics = json.load(open(path))

        if metrics.get("run_group") != vindr_group: continue

        scores_path = path.parent / "test_scores.npz"

        if not scores_path.exists():
            raise SystemExit("{} has no test_scores.npz - run score_vindr_checkpoints first"
                             .format(path.parent.name))

        scored = np.load(scores_path)
        macro_ap = derived_metrics(scored["scores"], scored["labels"],
                                   metrics["fitted_threshold"])["macro_average_precision"]

        cells[(metrics["target_size"], metrics["seed"], metrics["variant"])] = {
            "micro_auc": metrics["test_results"]["micro_auc"],
            "macro_auc": metrics["test_results"]["macro_auc"],
            "macro_ap": macro_ap,
            "micro_f1": metrics["test_results"]["micro_f1"],
        }

    return cells


def vindr_table(vindr_group):
    """Table III rows, then the paired tests quoted in the Results section."""
    cells = vindr_cells(vindr_group)
    targets = sorted({key[0] for key in cells})
    seeds = sorted({key[1] for key in cells})

    metrics = ("micro_auc", "macro_auc", "macro_ap", "micro_f1")

    print("\nTable III - VinDr-CXR, " + ", ".join(metrics))

    for target in targets:
        row = ["{:<5} & {:.2f}$\\times$".format(target, VINDR_INPUT / target)]
        row += [arm_pair([cells[(target, seed, BASELINE)][metric] for seed in seeds],
                         [cells[(target, seed, LEARNED)][metric] for seed in seeds], 4)
                for metric in metrics]
        print(" & ".join(row) + " \\\\")

    # Paired within seed: the arms share a seed, split and data order
    print("\nPaired statistics over all {} cells".format(len(targets) * len(seeds)))

    for metric in metrics:
        deltas, ratios = [], []

        for target in targets:
            for seed in seeds:
                deltas.append(cells[(target, seed, LEARNED)][metric]
                              - cells[(target, seed, BASELINE)][metric])
                ratios.append(VINDR_INPUT / target)

        paired_statistics(deltas, ratios, metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node21-groups", default=NODE21_GROUPS)
    parser.add_argument("--vindr-group", default=VINDR_GROUP)
    arguments = parser.parse_args()

    parameter_table()
    node21_table(arguments.node21_groups)
    vindr_table(arguments.vindr_group)


if __name__ == "__main__":
    main()
