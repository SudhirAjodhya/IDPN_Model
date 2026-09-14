"""
Every metric plotted against output resolution, one panel each, variants as lines.

fig1 answers one question - matched-FP sensitivity against resolution. This answers the rest
of them on the same axis, so a claim can be checked against several metrics at once instead
of resting on the single number the headline uses.

TWO KINDS OF PANEL, AND THE DIFFERENCE MATTERS.

  MATCHED-BUDGET panels (marked "matched") read each model at its own threshold, chosen so
  the whole test set sits at 0.25 FP/img. Cross-variant comparison is valid here.

  FIXED-THRESHOLD panels (marked "thr 0.3" / "thr 0.5") read every model at the same score.
  Useful for seeing how a variant's operating point DRIFTS with resolution, and included
  because the drift is itself informative - but NOT valid for saying one variant beats
  another, because the arms do not share a score scale. On this project a fixed 0.3
  threshold showed +0.068 where the matched budget showed +0.048, the difference being
  24% more false positives.

  THRESHOLD-FREE panels (FROC AUC, AP, image-level ROC AUC) integrate over the whole curve
  and so need no operating point at all. These are the most robust cross-variant panels
  here, and are worth quoting alongside the matched number.

Writes outputs/figures/fig10_metrics_vs_resolution<tag>.png

Run:
  python scripts/eval/plot_metric_vs_resolution.py --run-group v15_selection_and_scale,v16_attn_ablation
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
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


def read_curve(curve, x_key, y_key, at):
    if not curve: return None

    x = np.array([p[x_key] for p in curve], dtype=float)
    y = np.array([p[y_key] for p in curve], dtype=float)
    order = np.argsort(x)

    return float(np.interp(at, x[order], y[order], left=y[order][0], right=y[order][-1]))


def precision_from(sensitivity, fp_per_image, n_nodules, n_images):
    if sensitivity is None or fp_per_image is None: return None

    true_positives = sensitivity * n_nodules
    false_positives = fp_per_image * n_images

    if true_positives + false_positives <= 0: return None

    return true_positives / (true_positives + false_positives)


def metrics_for(run):
    """Every scalar this figure plots, for one run."""
    box = run["box_metrics"]["iou0.25"]
    froc = box["froc"]
    roc = (box.get("image_level") or {}).get("roc") or []
    image_level = box.get("image_level") or {}
    n_nodules, n_images = box.get("n_nodules"), box.get("n_images")

    matched_sens = read_curve(froc, "fp_per_image", "sensitivity", BUDGET)
    matched_thr = read_curve(froc, "fp_per_image", "threshold", BUDGET)

    values = {
        "matched_sens": matched_sens,
        "matched_prec": precision_from(matched_sens, BUDGET, n_nodules, n_images),
        "matched_thr":  matched_thr,
        "froc_auc":     box.get("froc_auc"),
        "ap":           box.get("average_precision"),
        "img_auc":      image_level.get("roc_auc"),
    }

    for threshold in (0.3, 0.5):
        sensitivity = read_curve(froc, "threshold", "sensitivity", threshold)
        fp_per_image = read_curve(froc, "threshold", "fp_per_image", threshold)
        key = str(threshold).replace(".", "")

        values["sens_t" + key] = sensitivity
        values["fp_t" + key] = fp_per_image
        values["prec_t" + key] = precision_from(sensitivity, fp_per_image, n_nodules, n_images)
        values["imgspec_t" + key] = read_curve(roc, "threshold", "specificity", threshold)

    # Medium bin only: the best-powered stratum, and where the effect concentrates
    by_size = box.get("froc_by_size") or {}
    values["medium_matched"] = read_curve(by_size.get("M") or [],
                                          "fp_per_image", "sensitivity", BUDGET)
    values["large_matched"] = read_curve(by_size.get("L") or [],
                                         "fp_per_image", "sensitivity", BUDGET)

    return values


# (key, title, y-label, comparison validity)
PANELS = [
    ("matched_sens",   "Sensitivity (matched)",        "sens @ 0.25 FP/img", "matched"),
    ("froc_auc",       "FROC AUC",                     "AUC",                "free"),
    ("ap",             "Average precision",            "AP",                 "free"),
    ("img_auc",        "Image-level ROC AUC",          "AUC",                "free"),
    ("medium_matched", "Medium nodules (matched)",     "sens @ 0.25 FP/img", "matched"),
    ("large_matched",  "Large nodules (matched)",      "sens @ 0.25 FP/img", "matched"),
    ("matched_prec",   "Precision (matched)",          "precision",          "matched"),
    ("sens_t03",       "Sensitivity at thr 0.3",       "sensitivity",        "fixed"),
    ("fp_t03",         "False positives at thr 0.3",   "FP / image",         "fixed"),
    ("prec_t03",       "Precision at thr 0.3",         "precision",          "fixed"),
    ("imgspec_t03",    "Image specificity at thr 0.3", "specificity",        "fixed"),
    ("matched_thr",    "Threshold needed for budget",  "score threshold",    "matched"),
]

VALIDITY_NOTE = {
    "matched": ("#0F6B66", "matched budget - comparable"),
    "free":    ("#2F5AA8", "threshold-free - comparable"),
    "fixed":   ("#8A5A12", "fixed threshold - NOT comparable"),
}


def load(run_group, recipe_filter):
    from scripts.eval.plot_run_diagnostics import recipe_key

    runs = defaultdict(list)

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != "fcos" or run.get("variant") not in COLOURS: continue
        if not run_group_matches(run, run_group): continue

        box = (run.get("box_metrics") or {}).get("iou0.25")

        if not box or not box.get("froc"): continue
        if recipe_filter is not None and recipe_key(run) != recipe_filter: continue

        runs[(run["target_size"], run["variant"])].append(metrics_for(run))

    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-group", default=None,
                        help="substring, or several comma-separated, on run_group")
    parser.add_argument("--recipe", default=None)
    arguments = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load(arguments.run_group, arguments.recipe)

    if not runs:
        print("no fcos runs matched. Run from the Model/ directory.")
        return

    targets = sorted({t for t, _ in runs})
    variants = [v for v in ORDER if any(v == var for _, var in runs)]

    columns = 4
    rows = (len(PANELS) + columns - 1) // columns
    figure, axes_grid = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.1 * rows))
    axes_flat = axes_grid.flatten()

    for axes, (key, title, y_label, validity) in zip(axes_flat, PANELS):
        colour, note = VALIDITY_NOTE[validity]

        for variant in variants:
            xs, ys, lows, highs = [], [], [], []

            for target in targets:
                entries = runs.get((target, variant))

                if not entries: continue

                values = [e[key] for e in entries if e.get(key) is not None]

                if not values: continue

                xs.append(target)
                ys.append(float(np.mean(values)))
                lows.append(float(np.mean(values)) - min(values))
                highs.append(max(values) - float(np.mean(values)))

            if not xs: continue

            axes.errorbar(xs, ys, yerr=[lows, highs], color=COLOURS[variant], lw=1.8,
                          marker="o", markersize=4.5, capsize=2.5, elinewidth=0.9,
                          label=LABELS[variant].replace("\n", " "))

        axes.set_title(title, fontsize=10, color=colour)
        axes.set_xlabel("Output resolution", fontsize=8.5)
        axes.set_ylabel(y_label, fontsize=8.5)
        axes.set_xticks(targets)
        axes.set_xticklabels([str(t) for t in targets], fontsize=8)
        axes.tick_params(axis="y", labelsize=8)
        axes.grid(alpha=0.22)
        axes.text(0.02, 0.03, note, transform=axes.transAxes, fontsize=6.8,
                  color=colour, style="italic")

    for axes in axes_flat[len(PANELS):]:
        axes.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                  fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    figure.suptitle("Every metric against output resolution. Panel titles are colour-coded "
                    "by whether cross-variant comparison is valid.", fontsize=12)
    figure.tight_layout(rect=(0, 0.045, 1, 0.965))

    tag = ""

    if arguments.recipe:
        tag += "_" + arguments.recipe.replace("+", "-").replace(".", "")

    if arguments.run_group:
        tag += "_" + "+".join(g.strip() for g in arguments.run_group.split(",") if g.strip())

    path = OUT_DIR / "fig10_metrics_vs_resolution{}.png".format(tag)
    figure.savefig(path, dpi=170)
    plt.close(figure)

    print("wrote {}".format(path))
    print("\nPanels marked 'fixed threshold' show operating-point DRIFT with resolution.")
    print("They are informative about a single variant and invalid for comparing two.")


if __name__ == "__main__":
    main()
