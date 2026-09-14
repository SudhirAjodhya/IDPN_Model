"""
Size-stratified figures: per-bin FROC curves, and sensitivity by bin across variants.

Reads `box_metrics.iou0.25.froc_by_size`, which sweeps each size bin over the SAME global
threshold grid and plots it against the SAME pooled false-positive axis. That is the only
well-posed way to read size at a matched budget: a false positive matched no ground truth,
so it belongs to no size bin and a per-bin FP rate does not exist. One global threshold puts
the whole image set at 0.25 FP/img and every bin is read there.

Not the stored `by_size` block - that is computed at a fixed 0.3 score threshold, where the
arms are not on a common operating point.

The S bin is drawn faded and labelled with its n, because with 10 nodules its curve moves in
steps of 0.1 and reads as structure when it is quantisation.

Writes to outputs/figures/:
    fig7_size_froc_t<target><tag>.png     one panel per size bin, all variants
    fig8_size_bars<tag>.png               sensitivity by bin, grouped by target

Run:
  python scripts/eval/plot_size_stratified.py --run-group v15_selection_and_scale,v16_attn_ablation
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval.plot_run_diagnostics import COLOURS, LABELS, ORDER, run_group_matches

OUT_DIR = Path("outputs/figures")
BUDGET = 0.25
BINS = ("S", "M", "L")

# Longest-axis edges are 10 and 20 mm; node21_detection_metrics derives mm from the box's
# fraction of image width times CHEST_WIDTH_MM=358, so the mm/pixel mapping is exact.
BIN_TITLE = {
    "S": "Small  (<10 mm)",
    "M": "Medium  (10-20 mm)",
    "L": "Large  (>=20 mm)",
}


def read_at_budget(curve, budget=BUDGET):
    """Sensitivity where the shared pooled FP axis crosses the budget."""
    if not curve: return None

    fp = np.array([p["fp_per_image"] for p in curve])
    se = np.array([p["sensitivity"] for p in curve])
    order = np.argsort(fp)

    return float(np.interp(budget, fp[order], se[order], left=0.0, right=se[order][-1]))


def load(recipe_filter, run_group):
    """(target, variant) -> {seed: {"curves": {bin: curve}, "counts": {bin: n}}}"""
    from scripts.eval.plot_run_diagnostics import recipe_key

    runs = {}

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != "fcos" or run.get("variant") not in COLOURS: continue
        if not run_group_matches(run, run_group): continue

        box = (run.get("box_metrics") or {}).get("iou0.25")

        if not box or not box.get("froc_by_size"): continue
        if recipe_filter is not None and recipe_key(run) != recipe_filter: continue

        counts = {b: (box.get("by_size", {}).get(b) or {}).get("n_nodules", 0) for b in BINS}
        key = (run["target_size"], run["variant"])
        runs.setdefault(key, {})[run.get("seed")] = {
            "curves": {b: box["froc_by_size"].get(b) or [] for b in BINS},
            "counts": counts,
        }

    return runs


def mean_curve(entries, bin_name, grid):
    """Seed-averaged sensitivity on a shared FP grid, only where every seed reaches."""
    stacked = []

    for entry in entries:
        curve = entry["curves"][bin_name]

        if not curve: continue

        fp = np.array([p["fp_per_image"] for p in curve])
        se = np.array([p["sensitivity"] for p in curve])
        order = np.argsort(fp)
        stacked.append(np.interp(grid, fp[order], se[order],
                                 left=np.nan, right=se[order][-1]))

    if not stacked: return None

    stacked = np.asarray(stacked)
    covered = ~np.isnan(stacked)

    return np.where(covered.all(axis=0),
                    np.where(covered, stacked, 0.0).sum(axis=0) / len(stacked), np.nan)


def figure_size_froc(runs, target, tag):
    """One panel per size bin; every variant's seed-averaged curve on each."""
    present = [v for (t, v) in runs if t == target]

    if not present: return None

    ordered = [v for v in ORDER if v in present]
    grid = np.linspace(0, 2.0, 300)
    counts = next(iter(runs[(target, ordered[0])].values()))["counts"]

    figure, axes_row = plt.subplots(1, 3, figsize=(13.2, 4.5), sharey=True)

    for axes, bin_name in zip(axes_row, BINS):
        n_nodules = counts.get(bin_name, 0)
        underpowered = 0 < n_nodules <= 20

        for variant in ordered:
            entries = list(runs[(target, variant)].values())
            curve = mean_curve(entries, bin_name, grid)

            if curve is None: continue

            axes.plot(grid, curve, color=COLOURS[variant], lw=1.9,
                      alpha=0.45 if underpowered else 1.0,
                      label="{}  (n={})".format(LABELS[variant].replace("\n", " "), len(entries)))

        axes.axvline(BUDGET, color="black", ls="--", lw=1.0, alpha=0.6)
        axes.set(xlabel="False positives / image (pooled)", xlim=(0, 2.0), ylim=(0, 1.0))
        axes.set_title("{}   n={}".format(BIN_TITLE[bin_name], n_nodules), fontsize=10.5)
        axes.grid(alpha=0.22)

        if underpowered:
            axes.text(0.5, 0.06,
                      "n={} - sensitivity moves in steps of {:.2f}\nnot interpretable".format(
                          n_nodules, 1 / n_nodules),
                      transform=axes.transAxes, ha="center", fontsize=8.5,
                      style="italic", color="#8A5A12")

    axes_row[0].set_ylabel("Sensitivity")
    axes_row[0].legend(fontsize=8, loc="upper left")

    figure.suptitle("Size-stratified FROC, target {}x{} - all bins read at the SAME pooled "
                    "FP axis".format(target, target), fontsize=11.5)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    path = OUT_DIR / "fig7_size_froc_t{}{}.png".format(target, tag)
    figure.savefig(path, dpi=180)
    plt.close(figure)

    return path


def figure_size_bars(runs, tag):
    """Sensitivity at the matched budget, grouped by target, one cluster per size bin."""
    targets = sorted({t for t, _ in runs})
    variants = [v for v in ORDER if any(v == var for _, var in runs)]

    if not targets or not variants: return None

    figure, axes_row = plt.subplots(1, len(BINS), figsize=(4.4 * len(BINS), 4.6), sharey=True)

    if len(BINS) == 1: axes_row = [axes_row]

    counts = {}

    for key, seeds in runs.items():
        counts.update(next(iter(seeds.values()))["counts"])

    for axes, bin_name in zip(axes_row, BINS):
        n_nodules = counts.get(bin_name, 0)
        underpowered = 0 < n_nodules <= 20
        width = 0.8 / max(len(variants), 1)

        for index, variant in enumerate(variants):
            centres, heights, lows, highs = [], [], [], []

            for position, target in enumerate(targets):
                seeds = runs.get((target, variant))

                if not seeds: continue

                values = [read_at_budget(e["curves"][bin_name]) for e in seeds.values()]
                values = [v for v in values if v is not None]

                if not values: continue

                centres.append(position + (index - (len(variants) - 1) / 2) * width)
                heights.append(float(np.mean(values)))
                lows.append(float(np.mean(values)) - min(values))
                highs.append(max(values) - float(np.mean(values)))

            if not centres: continue

            axes.bar(centres, heights, width * 0.92, color=COLOURS[variant],
                     alpha=0.45 if underpowered else 0.92,
                     label=LABELS[variant].replace("\n", " ") if bin_name == "M" else None)

            if any(l > 0 or h > 0 for l, h in zip(lows, highs)):
                axes.errorbar(centres, heights, yerr=[lows, highs], fmt="none",
                              ecolor="#333333", elinewidth=0.9, capsize=2.5, alpha=0.7)

        axes.set_xticks(range(len(targets)))
        axes.set_xticklabels(["{}".format(t) for t in targets])
        axes.set(xlabel="Output resolution", ylim=(0, 1.0))
        axes.set_title("{}   n={}".format(BIN_TITLE[bin_name], n_nodules), fontsize=10.5)
        axes.grid(alpha=0.22, axis="y")

        if underpowered:
            axes.text(0.5, 0.92, "n={} - not interpretable".format(n_nodules),
                      transform=axes.transAxes, ha="center", fontsize=8.5,
                      style="italic", color="#8A5A12")

    axes_row[0].set_ylabel("Sensitivity at {} FP/img".format(BUDGET))
    axes_row[min(1, len(BINS) - 1)].legend(fontsize=8.5, loc="upper left")

    figure.suptitle("Sensitivity by nodule size at a matched {} FP/img "
                    "(bars are seed means, whiskers the observed range)".format(BUDGET),
                    fontsize=11.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    path = OUT_DIR / "fig8_size_bars{}.png".format(tag)
    figure.savefig(path, dpi=180)
    plt.close(figure)

    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", default=None, help="exact recipe tag to keep")
    parser.add_argument("--run-group", default=None,
                        help="substring, or several comma-separated, on run_group")
    arguments = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load(arguments.recipe, arguments.run_group)

    if not runs:
        print("no fcos runs with froc_by_size matched. Run from the Model/ directory.")
        return

    tag = ""

    if arguments.recipe:
        tag += "_" + arguments.recipe.replace("+", "-").replace(".", "")

    if arguments.run_group:
        tag += "_" + "+".join(g.strip() for g in arguments.run_group.split(",") if g.strip())

    for target in sorted({t for t, _ in runs}):
        path = figure_size_froc(runs, target, tag)
        print("wrote {}".format(path) if path else "skipped t{}".format(target))

    path = figure_size_bars(runs, tag)
    print("wrote {}".format(path) if path else "skipped fig8")

    print("\nAll bins read against the SAME pooled FP axis - false positives match no")
    print("ground truth, so a per-bin FP rate is not defined.")


if __name__ == "__main__":
    main()
