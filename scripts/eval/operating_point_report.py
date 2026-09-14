"""
Operating characteristics across score thresholds: recall, precision, FP/img, specificity.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR.

For: showing how ONE model behaves as the threshold moves - the sensitivity you buy and the
false alarms you pay. That is the operating-characteristic question, and a fixed-threshold
table answers it well.

NOT for: claiming one variant beats another. The arms do not share a score scale, so at a
common threshold one is simply making more guesses. On this project, at threshold 0.3 the
learned resizer read +0.068 while spending 24% more false positives; at a matched budget the
same pair read +0.048. Cross-variant claims belong at a matched FP budget - the row marked
MATCHED below - and nowhere else on this table.

ON THRESHOLD DIRECTION. A LOWER threshold keeps more predictions, so it is LESS strict and
produces MORE false positives. Wanting good sensitivity "at a low threshold" is backwards;
what is wanted is good sensitivity at a STRICT operating point, which means a HIGH threshold
and few FP/img. That is what the matched 0.25 FP/img row reports.

ON SPECIFICITY. There is no box-level specificity: "true negative box" has no denominator,
since an image contains unboundedly many non-boxes. Specificity here is IMAGE level - of
the 374 nodule-free films, how many were called clean. Sensitivity is reported at both
levels and the table says which is which:

    box  recall      of 147 nodules, how many were localised (IoU >= 0.25)
    box  precision   of the predictions kept, how many hit a nodule
    box  FP/img      false alarms per image, the cost axis
    img  sensitivity of 143 nodule-bearing films, how many were flagged
    img  specificity of 374 clean films, how many were left alone

Everything is derived from stored metrics - no model loading, no GPU.

Run:
  python scripts/eval/operating_point_report.py --run-group v15_selection_and_scale,v16_attn_ablation
  python scripts/eval/operating_point_report.py --run-group v15_selection_and_scale --target 384 --markdown
"""
import argparse
import glob
import json
import statistics
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
THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def interpolate(curve, x_key, y_key, at, descending_x=False):
    """Read a stored curve at an arbitrary x, without extrapolating past its ends."""
    if not curve: return None

    x = np.array([p[x_key] for p in curve], dtype=float)
    y = np.array([p[y_key] for p in curve], dtype=float)
    order = np.argsort(x)

    return float(np.interp(at, x[order], y[order], left=y[order][0], right=y[order][-1]))


def precision_from(sensitivity, fp_per_image, n_nodules, n_images):
    """TP / (TP + FP), both recovered from the curve's own rates."""
    if sensitivity is None or fp_per_image is None: return None

    true_positives = sensitivity * n_nodules
    false_positives = fp_per_image * n_images

    if true_positives + false_positives <= 0: return None

    return true_positives / (true_positives + false_positives)


def load(run_group, recipe_filter, target_filter):
    from scripts.eval.plot_run_diagnostics import recipe_key

    runs = defaultdict(list)

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != "fcos" or run.get("variant") not in COLOURS: continue
        if not run_group_matches(run, run_group): continue
        if target_filter and run.get("target_size") != target_filter: continue

        box = (run.get("box_metrics") or {}).get("iou0.25")

        if not box or not box.get("froc"): continue
        if recipe_filter is not None and recipe_key(run) != recipe_filter: continue

        runs[(run["target_size"], run["variant"])].append({
            "seed":       run.get("seed"),
            "froc":       box["froc"],
            "roc":        (box.get("image_level") or {}).get("roc") or [],
            "n_nodules":  box.get("n_nodules"),
            "n_images":   box.get("n_images"),
            "n_pos_img":  (box.get("image_level") or {}).get("n_positive_images"),
            "n_neg_img":  (box.get("image_level") or {}).get("n_negative_images"),
        })

    return runs


def row_at_threshold(entry, threshold):
    """Every metric at one score threshold, for one run."""
    sensitivity = interpolate(entry["froc"], "threshold", "sensitivity", threshold)
    fp_per_image = interpolate(entry["froc"], "threshold", "fp_per_image", threshold)

    return {
        "recall":      sensitivity,
        "fp":          fp_per_image,
        "precision":   precision_from(sensitivity, fp_per_image,
                                      entry["n_nodules"], entry["n_images"]),
        "img_sens":    interpolate(entry["roc"], "threshold", "sensitivity", threshold),
        "img_spec":    interpolate(entry["roc"], "threshold", "specificity", threshold),
    }


def row_at_budget(entry, budget=BUDGET):
    """Every metric where this run's own curve crosses the FP budget."""
    threshold = interpolate(entry["froc"], "fp_per_image", "threshold", budget)
    row = row_at_threshold(entry, threshold)
    row["threshold"] = threshold

    return row


def average(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]

    return statistics.mean(values) if values else None


def fmt(value, decimals=3):
    return "-" if value is None else "{:.{d}f}".format(value, d=decimals)


def emit(headers, rows, markdown):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]

    def line(values):
        cells = [str(v).rjust(w) if i else str(v).ljust(w)
                 for i, (v, w) in enumerate(zip(values, widths))]

        return ("| " + " | ".join(cells) + " |") if markdown else ("  " + "  ".join(cells))

    out = [line(headers)]

    if markdown:
        out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    out.extend(line(r) for r in rows)

    return "\n".join(out)


def figure_operating_points(runs, target, tag):
    """Three panels: PR, image-level ROC, and the FROC with the budget marked."""
    present = [v for (t, v) in runs if t == target]

    if not present: return None

    ordered = [v for v in ORDER if v in present]
    figure, (pr_axes, roc_axes, froc_axes) = plt.subplots(1, 3, figsize=(13.4, 4.5))

    recall_grid = np.linspace(0.02, 1.0, 200)
    fp_grid = np.linspace(0.0, 2.0, 200)
    spec_grid = np.linspace(0.0, 1.0, 200)

    for variant in ordered:
        entries = runs[(target, variant)]
        colour = COLOURS[variant]
        label = "{}  (n={})".format(LABELS[variant].replace("\n", " "), len(entries))

        # Precision against recall, averaged only where every seed reaches
        stacked = []

        for entry in entries:
            sens = np.array([p["sensitivity"] for p in entry["froc"]])
            fp = np.array([p["fp_per_image"] for p in entry["froc"]])
            precision = np.array([
                precision_from(s, f, entry["n_nodules"], entry["n_images"]) or np.nan
                for s, f in zip(sens, fp)])
            order = np.argsort(sens)
            stacked.append(np.interp(recall_grid, sens[order], precision[order],
                                     left=np.nan, right=np.nan))

        stacked = np.asarray(stacked)
        covered = ~np.isnan(stacked)
        mean = np.where(covered.all(axis=0),
                        np.where(covered, stacked, 0.0).sum(axis=0) / len(stacked), np.nan)
        pr_axes.plot(recall_grid, mean, color=colour, lw=1.9, label=label)

        # Image-level ROC: sensitivity against specificity, drawn on a specificity axis so
        # it reads the same direction as the rest of the figure
        roc_stack = []

        for entry in entries:
            if not entry["roc"]: continue

            spec = np.array([p["specificity"] for p in entry["roc"]])
            sens = np.array([p["sensitivity"] for p in entry["roc"]])
            order = np.argsort(spec)
            roc_stack.append(np.interp(spec_grid, spec[order], sens[order],
                                       left=sens[order][0], right=sens[order][-1]))

        if roc_stack:
            roc_axes.plot(spec_grid, np.mean(roc_stack, axis=0), color=colour, lw=1.9,
                          label=label)

        # FROC
        froc_stack = []

        for entry in entries:
            fp = np.array([p["fp_per_image"] for p in entry["froc"]])
            sens = np.array([p["sensitivity"] for p in entry["froc"]])
            order = np.argsort(fp)
            froc_stack.append(np.interp(fp_grid, fp[order], sens[order],
                                        left=0.0, right=sens[order][-1]))

        froc_axes.plot(fp_grid, np.mean(froc_stack, axis=0), color=colour, lw=1.9, label=label)

    pr_axes.set(xlabel="Recall (nodules localised)", ylabel="Precision",
                xlim=(0, 1.0), ylim=(0, 0.8))
    pr_axes.set_title("Precision-recall (box level)", fontsize=10.5)
    pr_axes.legend(fontsize=7.5, loc="upper right")

    roc_axes.plot([1, 0], [0, 1], color="#999999", ls=":", lw=1.0)
    roc_axes.set(xlabel="Specificity (clean films left alone)", ylabel="Sensitivity (films flagged)",
                 xlim=(1.0, 0.0), ylim=(0, 1.02))
    roc_axes.set_title("Image-level ROC", fontsize=10.5)

    froc_axes.axvline(BUDGET, color="black", ls="--", lw=1.0, alpha=0.65)
    froc_axes.text(BUDGET + 0.04, 0.04, "reported\nbudget", fontsize=7.5, alpha=0.8)
    froc_axes.set(xlabel="False positives / image", ylabel="Sensitivity",
                  xlim=(0, 2.0), ylim=(0, 0.9))
    froc_axes.set_title("FROC (box level)", fontsize=10.5)

    for axes in (pr_axes, roc_axes, froc_axes):
        axes.grid(alpha=0.22)

    figure.suptitle("Operating characteristics, target {}x{}. Curves are seed means; "
                    "cross-variant claims belong at the marked budget only.".format(target, target),
                    fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    path = OUT_DIR / "fig9_operating_points_t{}{}.png".format(target, tag)
    figure.savefig(path, dpi=180)
    plt.close(figure)

    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-group", default=None,
                        help="substring, or several comma-separated, on run_group")
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--target", type=int, default=None, help="restrict to one target")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load(arguments.run_group, arguments.recipe, arguments.target)

    if not runs:
        print("no fcos runs matched. Run from the Model/ directory.")
        return

    sample = next(iter(runs.values()))[0]

    print("\nOPERATING CHARACTERISTICS ACROSS SCORE THRESHOLDS")
    print("{} nodules on {} images   |   {} nodule-bearing films, {} clean films".format(
        sample["n_nodules"], sample["n_images"], sample["n_pos_img"], sample["n_neg_img"]))
    print("\nA LOWER threshold keeps MORE predictions - less strict, more false positives.")
    print("Specificity is IMAGE level; box-level specificity does not exist (no true-negative")
    print("box denominator). Only the MATCHED row is valid for comparing variants.")

    for target in sorted({t for t, _ in runs}):
        present = [v for (t, v) in runs if t == target]
        ordered = [v for v in ORDER if v in present]

        print("\n" + "=" * 92)
        print("TARGET {}".format(target))
        print("=" * 92)

        for variant in ordered:
            entries = runs[(target, variant)]
            rows = []

            for threshold in THRESHOLDS:
                per_run = [row_at_threshold(e, threshold) for e in entries]
                rows.append([
                    fmt(threshold, 2),
                    fmt(average(per_run, "recall")),
                    fmt(average(per_run, "precision")),
                    fmt(average(per_run, "fp"), 2),
                    fmt(average(per_run, "img_sens")),
                    fmt(average(per_run, "img_spec")),
                ])

            budget_rows = [row_at_budget(e) for e in entries]
            rows.append([
                "MATCHED",
                fmt(average(budget_rows, "recall")),
                fmt(average(budget_rows, "precision")),
                "{:.2f}".format(BUDGET),
                fmt(average(budget_rows, "img_sens")),
                fmt(average(budget_rows, "img_spec")),
            ])

            print("\n{}  (n={} seeds)".format(LABELS[variant].replace("\n", " "), len(entries)))
            print(emit(["thresh", "box recall", "box prec", "FP/img", "img sens", "img spec"],
                       rows, arguments.markdown))

            matched_threshold = average(budget_rows, "threshold")

            if matched_threshold is not None:
                print("  MATCHED row sits at threshold {:.3f} for this variant - fitted per "
                      "model, which is the point.".format(matched_threshold))

    if not arguments.no_figures:
        tag = ""

        if arguments.recipe:
            tag += "_" + arguments.recipe.replace("+", "-").replace(".", "")

        if arguments.run_group:
            tag += "_" + "+".join(g.strip() for g in arguments.run_group.split(",") if g.strip())

        print()

        for target in sorted({t for t, _ in runs}):
            path = figure_operating_points(runs, target, tag)
            print("wrote {}".format(path) if path else "skipped t{}".format(target))


if __name__ == "__main__":
    main()
