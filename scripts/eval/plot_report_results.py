"""
The three Quantitative Results figures in the report, without titles or footers.

  results_vindr_delta.png       learned minus bilinear on VinDr against decimation, one
                                point per paired seed, for micro-AUC and macro-AP
  results_node21_froc.png       seed-averaged FROC curves at the lowest and highest
                                decimated targets in Table II
  results_vindr_per_class.png   per-class AUC difference at 224 and 1024 pixels

Everything is read from the same files as `report_tables`, so the figures and tables
share one definition of every number.

Usage (from Model/):
    python -m scripts.eval.plot_report_results --out-dir outputs/report_figures
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval.plot_run_diagnostics import recipe_key                          # noqa: E402
from scripts.eval.plot_supervisor_figures import run_group_matches, BUDGET        # noqa: E402
from scripts.eval.plot_vindr_sweep import COLOURS, DELTA_COLOUR, THIN_CLASS_FILMS  # noqa: E402
from scripts.eval.plot_vindr_sweep import load as load_vindr                      # noqa: E402
from scripts.eval.report_tables import (BASELINE, LEARNED, NODE21_GROUPS,         # noqa: E402
                                        NODE21_RECIPE, VINDR_GROUP, VINDR_INPUT,
                                        vindr_cells)

LABELS = {BASELINE: "Bilinear", LEARNED: "Learned"}
FROC_TARGETS = (224, 768)


def figure_vindr_delta(plt, out_dir):
    """Paired difference against decimation ratio, with the regression the text quotes."""
    cells = vindr_cells(VINDR_GROUP)
    targets = sorted({key[0] for key in cells})
    seeds = sorted({key[1] for key in cells})

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for axis, (metric, label) in zip(axes, (("micro_auc", "micro-AUC"), ("macro_ap", "macro-AP"))):
        ratios, deltas = [], []

        # One point per (target, seed) pair, then the per-target mean over seeds
        for target in targets:
            target_deltas = [cells[(target, seed, LEARNED)][metric]
                             - cells[(target, seed, BASELINE)][metric] for seed in seeds]
            ratios += [VINDR_INPUT / target] * len(seeds)
            deltas += target_deltas

            axis.scatter([VINDR_INPUT / target] * len(seeds), target_deltas, s=12,
                         color=DELTA_COLOUR, alpha=0.45, linewidths=0)
            axis.scatter(VINDR_INPUT / target, statistics.mean(target_deltas), s=34,
                         marker="D", color=DELTA_COLOUR)

        fit = stats.linregress(ratios, deltas)
        line = np.linspace(min(ratios), max(ratios), 50)

        axis.plot(line, fit.intercept + fit.slope * line, color=DELTA_COLOUR, linewidth=1.4,
                  label="fit, slope {:+.4f}, p = {:.3f}".format(fit.slope, fit.pvalue))
        axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_xlabel("Decimation ratio (2048 / output)")
        axis.set_ylabel("$\\Delta$ " + label + " (learned $-$ bilinear)")
        axis.legend(frameon=False, fontsize=7, loc="upper left")
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(out_dir / "results_vindr_delta.png", dpi=300)
    plt.close(figure)


def figure_node21_froc(plt, out_dir):
    """Mean FROC over seeds for both arms, at the two ends of the decimated range."""
    curves = {}

    for path in glob.glob(str(ROOT / "outputs" / "**" / "metrics.json"), recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != "fcos" or run.get("input_size") != 1024: continue
        if run.get("variant") not in LABELS or run.get("target_size") not in FROC_TARGETS: continue
        if recipe_key(run) != NODE21_RECIPE or not run_group_matches(run, NODE21_GROUPS): continue

        froc = run["box_metrics"]["iou0.25"]["froc"]
        curves.setdefault((run["variant"], run["target_size"]), []).append(froc)

    # Each seed's curve is interpolated onto a shared FP axis before averaging
    fp_grid = np.linspace(0.0, 2.0, 300)
    figure, axes = plt.subplots(1, len(FROC_TARGETS), figsize=(7.0, 2.6), sharey=True)

    for axis, target in zip(axes, FROC_TARGETS):
        for variant in (BASELINE, LEARNED):
            per_seed = []

            for froc in curves[(variant, target)]:
                fp = np.array([point["fp_per_image"] for point in froc])
                sensitivity = np.array([point["sensitivity"] for point in froc])
                order = np.argsort(fp)
                per_seed.append(np.interp(fp_grid, fp[order], sensitivity[order]))

            axis.plot(fp_grid, np.mean(per_seed, axis=0), color=COLOURS[variant],
                      linewidth=1.5, label=LABELS[variant])

        axis.axvline(BUDGET, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_title("{0} $\\times$ {0}".format(target), fontsize=9)
        axis.set_xlabel("False positives per image")
        axis.set_xlim(0.0, 2.0)
        axis.set_ylim(0.0, 0.9)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Sensitivity")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(out_dir / "results_node21_froc.png", dpi=300)
    plt.close(figure)


def figure_vindr_per_class(plt, out_dir):
    """Per-class AUC difference, seed-averaged, at the most and least decimated outputs."""
    runs = load_vindr(VINDR_GROUP)
    targets = sorted({key[0] for key in runs})
    seeds = sorted({key[2] for key in runs})
    ends = (targets[0], targets[-1])

    def class_auc(target, variant, seed):
        """Class name -> test AUC for one run."""
        return {entry["name"]: entry["auc"]
                for entry in runs[(target, variant, seed)]["test_results"]["per_class"]}

    sample = runs[(targets[0], BASELINE, seeds[0])]["test_results"]["per_class"]
    films = {entry["name"]: entry["n_test_films"] for entry in sample}

    deltas = {}

    for target in ends:
        per_seed = [{name: class_auc(target, LEARNED, seed)[name]
                     - class_auc(target, BASELINE, seed)[name] for name in films}
                    for seed in seeds]
        deltas[target] = {name: statistics.mean(d[name] for d in per_seed) for name in films}

    # Ordered by the difference at the most decimated output, largest gain at the top
    names = sorted(films, key=lambda name: deltas[ends[0]][name])
    positions = np.arange(len(names))
    height = 0.38

    figure, axis = plt.subplots(figsize=(3.5, 4.4))

    for offset, target, colour in ((height / 2, ends[0], DELTA_COLOUR), (-height / 2, ends[1], "0.6")):
        axis.barh(positions + offset, [deltas[target][name] for name in names], height,
                  color=colour, label="{0} $\\times$ {0}".format(target))

    axis.set_yticks(positions)
    axis.set_yticklabels(["{} ({})".format(name, films[name]) for name in names], fontsize=7)

    # Classes too thin to carry a per-class claim alone are greyed out
    for tick, name in zip(axis.get_yticklabels(), names):
        if films[name] < THIN_CLASS_FILMS: tick.set_color("0.55")

    axis.axvline(0.0, color="0.3", linewidth=0.8)
    axis.set_xlabel("$\\Delta$ AUC (learned $-$ bilinear)")
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "results_vindr_per_class.png", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "report_figures")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figure_vindr_delta(plt, args.out_dir)
    figure_node21_froc(plt, args.out_dir)
    figure_vindr_per_class(plt, args.out_dir)
    print("wrote 3 figures to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
