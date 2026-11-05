"""
Per-run training diagnostics and the precision-side metrics, for the supervisor review.

Answers the request for "loss graphs (train vs val), sensitivity over epochs, precision,
false positives at thresholds, FROC" on the checkpoint that was actually selected.

NOTHING HERE NEEDS A RE-RUN. Precision and F1 are recovered from the stored FROC curve:
each point carries sensitivity and fp_per_image, and the run records n_nodules and n_images,
so

    TP = sensitivity * n_nodules        FP = fp_per_image * n_images
    precision = TP / (TP + FP)

which is exact, not an approximation - those are the same counts the curve was built from.

WHAT IS NOT HERE, AND WHY
  Image-level SPECIFICITY cannot be recovered. It is the fraction of nodule-free test images
  with no detection above threshold, and metrics.json stores only the aggregate false-positive
  count, not per-image flags. Adding it means a new counter in
  src/evaluation/node21_detection_metrics.py and a re-evaluation pass over every checkpoint.

READING THE CURVES
  The validation sensitivity panel is deliberately noisy-looking, because it is. Adjacent
  epochs swing by ~0.15 on a 924-image validation set, so the selected checkpoint is the
  argmax of a noisy series and sits above the model's true level. trainer.py:320-323 records
  this, and it is why the training curve should be shown with the selection epoch marked
  rather than quoted as a performance number.

Usage:
    python scripts/eval/plot_run_diagnostics.py --target 384
    python scripts/eval/plot_run_diagnostics.py --target 512 --seed 42
    python scripts/eval/plot_run_diagnostics.py --demo
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

BUDGET = 0.25
OUT_DIR = Path("outputs/figures")

COLOURS = {
    "bilinear":                    "#4C72B0",
    "node21_talebi":               "#DD8452",
    "node21_talebi_zeroinit":      "#C44E52",
    "node21_talebi_zeroinit_attn": "#8172B2",
    # Anti-aliased family, 2026-09-06. Same hue as the aliased arm it replaces, darker, so
    # a chart mixing them is obviously wrong rather than subtly wrong - though the run
    # groups are separate precisely so that mixing does not happen by accident.
    "bilinear_antialias":             "#2F4B73",
    "bicubic_antialias":              "#55843F",
}
LABELS = {
    "bilinear":                    "Bilinear",
    "node21_talebi":               "Talebi (random init)",
    "node21_talebi_zeroinit":      "Talebi (zero-init)",
    "node21_talebi_zeroinit_attn": "Talebi + attention",
    "bilinear_antialias":             "Bilinear + anti-alias",
    "bicubic_antialias":              "Bicubic + anti-alias",
}
ORDER = ["bilinear", "bilinear_antialias", "bicubic_antialias", "node21_talebi",
         "node21_talebi_zeroinit", "node21_talebi_zeroinit_attn"]


def run_group_matches(run, run_group):
    """
    True when the run belongs to any of the requested groups.

    Accepts a COMMA-SEPARATED list because arms of one comparison can live in different
    run groups: bilinear and node21_talebi_zeroinit are in v15_selection_and_scale while
    node21_talebi_zeroinit_attn is in v16_attn_ablation, yet all three share a recipe and
    a selection rule and belong on the same chart. The alternative - a looser substring
    like "v1" - would silently pull in v11 through v14 as well, which is the pooling
    mistake this filter exists to prevent.

    Args:
        run: parsed metrics.json
        run_group: None for no filter, else one substring or several comma-separated

    Returns:
        bool
    """
    if run_group is None: return True

    actual = str(run.get("run_group", ""))

    return any(wanted.strip() in actual for wanted in run_group.split(",") if wanted.strip())


def precision_recall_from_froc(froc, n_nodules, n_images):
    """
    Recover (recall, precision, fp_per_image, threshold) from a stored FROC curve.

    Exact rather than approximate: the FROC points were themselves built from these counts.
    """
    points = []

    for point in froc:
        true_positives  = point["sensitivity"] * n_nodules
        false_positives = point["fp_per_image"] * n_images
        denominator = true_positives + false_positives
        precision = true_positives / denominator if denominator else 0.0
        points.append((point["sensitivity"], precision,
                       point["fp_per_image"], point["threshold"]))

    return points


def at_budget(points, n_nodules, n_images, budget=BUDGET):
    """
    Recall, precision, F1 and score threshold at a false-positive budget.

    Precision is DERIVED, never interpolated. At a budget of `budget` FP per image the false
    positive count is `budget * n_images` by definition, so once recall is interpolated the
    rest is exact:

        TP = recall * n_nodules,  FP = budget * n_images,  precision = TP / (TP + FP)

    Interpolating precision directly against FP gives nonsense here: precision changes very
    steeply near the low-FP end, where pre-2026-08-31 FROC curves are sparsest, and the first
    point typically sits at sensitivity 0 with a single false positive.
    """
    if not points: return None

    fp = np.array([p[2] for p in points])
    order = np.argsort(fp)

    def interp(index):
        values = np.array([p[index] for p in points])[order]
        return float(np.interp(budget, fp[order], values,
                               left=values[0], right=values[-1]))

    recall = interp(0)
    true_positives  = recall * n_nodules
    false_positives = budget * n_images
    denominator = true_positives + false_positives
    precision = true_positives / denominator if denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"recall": recall, "precision": precision, "f1": f1,
            "threshold": interp(3)}


def recipe_key(run):
    """
    Stable tag for every axis that changes a run but is not variant / input / target.

    WITHOUT THIS, runs sharing (variant, seed) silently overwrite each other in the dict
    below and the figure blends arms that are not comparable. It has bitten twice: v13's
    CXR-pretrained backbone runs merged into the ConvNeXt bars, and v14's Arm V, Arm N and
    Arm C - which differ only in training data and VinDr input resolution - collapsed onto
    a single cell. Both failures were invisible in the output.

    Args:
        run: parsed metrics.json

    Returns:
        Short tag, empty for the canonical NODE21-only ConvNeXt recipe so pre-existing
        figures keep their labels unchanged.
    """
    split_info = run.get("split_info") or {}
    parts = []

    backbone = run.get("backbone", "convnext_fpn")

    if backbone != "convnext_fpn":
        parts.append(backbone)

    if split_info.get("datasets", "node21") != "node21":
        parts.append(split_info["datasets"])

    vindr_input = split_info.get("vindr_input_size")

    if vindr_input and vindr_input != run.get("input_size"):
        parts.append(f"vin{vindr_input}")

    ratio = split_info.get("vindr_negative_ratio")

    if ratio is not None:
        parts.append(f"vnr{ratio:g}")

    # Checkpoint SELECTION RULE. v14 and earlier selected on val_sens - sensitivity at a
    # fixed 0.3 score threshold - while reporting at matched 0.25 FP/img. v15 onward
    # selects on val_sens_matched so the criterion matches the reported metric. Runs
    # selected under different rules are NOT comparable, and without this they share a tag
    # and blend into one bar.
    select_on = run.get("select_on", "val_sens")

    if select_on != "val_sens":
        parts.append(f"sel-{select_on}")

    return "+".join(parts)


def load(target, seed=None, recipe=None, run_group=None, input_size=1024):
    """
    Every fcos run at this target, keyed by (variant, seed, recipe tag).

    Args:
        target: target resolution to select
        seed: optional seed filter
        recipe: optional exact recipe tag to keep (see recipe_key)
        run_group: optional substring filter on run_group. Needed because some pipeline
            changes are NOT recorded in metrics.json and so cannot appear in the recipe
            tag - notably the 2026-08-27 INTER_AREA dataloader fix, which changed 63% of
            NODE21 films. Pre-fix and post-fix runs otherwise share a tag and would be
            averaged together.

    Returns:
        dict keyed by (variant, seed, recipe_tag)
    """
    runs = {}

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != "fcos" or run.get("letterbox"): continue
        # Input resolution is an EXPERIMENTAL AXIS as of 2026-09-18, not a constant. It was
        # hardcoded to 1024 here, so runs at any other input were dropped without a word -
        # the silent-filter failure this project has hit repeatedly. Runs at different
        # inputs are NOT comparable (decimation ratio differs), so they are separated here
        # rather than merged, but the caller now chooses which input to look at.
        if run.get("input_size") != input_size or run.get("target_size") != target: continue
        if run.get("variant") not in COLOURS: continue
        if seed is not None and run.get("seed") != seed: continue

        box_metrics = (run.get("box_metrics") or {}).get("iou0.25")

        if not box_metrics or not box_metrics.get("froc"): continue

        tag = recipe_key(run)

        if recipe is not None and tag != recipe: continue
        if not run_group_matches(run, run_group): continue

        # NOT filtered on history: runs rebuilt by recover_metrics.py have box_metrics but
        # no per-epoch history. They are valid for the metrics table and only excluded from
        # the curve panels, which is handled at the call site.
        runs[(run["variant"], run["seed"], tag)] = run

    return runs


def figure_training_curves(runs, target, plt, tag=""):
    """Train/val loss, validation sensitivity and validation FP per epoch, per variant."""
    keys = sorted(runs, key=lambda k: (ORDER.index(k[0]), k[1], k[2]))

    if not keys: return None

    figure, axes_grid = plt.subplots(3, len(keys), figsize=(4.0 * len(keys), 9.0),
                                     squeeze=False)

    for column, key in enumerate(keys):
        variant, seed, _recipe = key
        run = runs[key]
        history = run["history"]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        selected = run.get("best_val_sensitivity_epoch")

        # Row 0 - the overfitting question
        axes = axes_grid[0][column]
        axes.plot(epochs, history["train_loss"], color="#4C72B0", lw=1.6, label="train")
        axes.plot(epochs, history["val_loss"], color="#C44E52", lw=1.4, alpha=0.85,
                  label="val")

        if selected: axes.axvline(selected, color="black", ls=":", lw=1.2)

        axes.set_title("{}\nseed {}".format(LABELS[variant], seed), fontsize=9.5)
        axes.grid(alpha=0.22)

        if column == 0:
            axes.set_ylabel("Loss")
            axes.legend(fontsize=8)

        # Row 1 - what the checkpoint was selected on
        axes = axes_grid[1][column]
        axes.plot(epochs, history["sensitivity"], color="#55A868", lw=1.6)

        if selected:
            axes.axvline(selected, color="black", ls=":", lw=1.2)
            axes.plot([selected], [run["best_val_sensitivity"]], "o", color="black", ms=6)
            axes.annotate("selected\nepoch {}".format(selected),
                          xy=(selected, run["best_val_sensitivity"]),
                          xytext=(4, -26), textcoords="offset points", fontsize=7.5)

        axes.set(ylim=(0, 1.0))
        axes.grid(alpha=0.22)

        if column == 0: axes.set_ylabel("Validation sensitivity")

        # Row 2 - the cost side, which sensitivity alone hides
        axes = axes_grid[2][column]
        axes.plot(epochs, history["fp_per_image"], color="#8172B2", lw=1.6)

        if selected: axes.axvline(selected, color="black", ls=":", lw=1.2)

        axes.axhline(BUDGET, color="black", ls="--", lw=1.0, alpha=0.6)
        axes.set(xlabel="Epoch", ylim=(0, max(2.0, max(history["fp_per_image"]) * 1.1)))
        axes.grid(alpha=0.22)

        if column == 0: axes.set_ylabel("Validation FP / image")

    figure.suptitle("Training diagnostics at target {}x{} - dotted line marks the selected "
                    "checkpoint, dashed line the {} FP/img budget".format(target, target, BUDGET),
                    fontsize=11)
    figure.text(0.5, 0.008,
                "Validation sensitivity swings by ~0.15 between adjacent epochs on 924 images, "
                "so the selected epoch is the argmax of a noisy series\nand sits above the "
                "model's true level. Read it as a selection record, not as a performance number.",
                ha="center", fontsize=8, color="dimgray")
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    # Recipe MUST be in the filename or two arms silently overwrite each other, which is
    # gotcha 9 (output paths must encode recipe knobs) reappearing as figures.
    path = OUT_DIR / "fig5_training_curves_t{}{}.png".format(target, tag)
    figure.savefig(path, dpi=180)
    plt.close(figure)

    return path


def figure_precision_recall(runs, target, plt, tag=""):
    """Precision against recall, and precision against FP budget, on the test split."""
    by_variant = {}

    for (variant, seed, recipe), run in runs.items():
        box_metrics = run["box_metrics"]["iou0.25"]
        by_variant.setdefault(variant, []).append(
            precision_recall_from_froc(box_metrics["froc"],
                                       box_metrics["n_nodules"], box_metrics["n_images"]))

    if not by_variant: return None

    def full_coverage_mean(curves):
        """
        Mean across seeds, kept only where EVERY seed contributes a value.

        np.interp returns NaN outside a seed's own recall range, so a plain nanmean draws
        the tails from whichever subset of seeds happens to reach there - a one-seed tail
        rendered identically to a three-seed middle, with the legend still claiming n=3.
        It also emits "Mean of empty slice" wherever no seed reaches. Masking to full
        coverage makes each curve stop where its evidence stops.

        Args:
            curves: list of per-seed arrays over the shared grid, NaN outside each range

        Returns:
            Array with the mean where all seeds cover, NaN elsewhere (matplotlib skips it)
        """
        stacked = np.asarray(curves, dtype=float)
        covered = ~np.isnan(stacked)

        # Sum over zero-filled copies to avoid warnings, then blank the partial columns
        totals = np.where(covered, stacked, 0.0).sum(axis=0)

        return np.where(covered.all(axis=0), totals / len(stacked), np.nan)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    recall_grid = np.linspace(0, 1.0, 200)
    fp_grid = np.linspace(0.02, 2.0, 200)

    for variant in ORDER:
        if variant not in by_variant: continue

        seeds = by_variant[variant]
        precision_by_recall, precision_by_fp = [], []

        for points in seeds:
            recall = np.array([p[0] for p in points])
            precision = np.array([p[1] for p in points])
            fp = np.array([p[2] for p in points])
            order = np.argsort(recall)
            precision_by_recall.append(np.interp(recall_grid, recall[order], precision[order],
                                                 left=np.nan, right=np.nan))
            order = np.argsort(fp)
            precision_by_fp.append(np.interp(fp_grid, fp[order], precision[order],
                                             left=np.nan, right=np.nan))

        label = "{}  (n={})".format(LABELS[variant], len(seeds))

        left.plot(recall_grid, full_coverage_mean(precision_by_recall),
                  color=COLOURS[variant], lw=1.9, label=label)
        right.plot(fp_grid, full_coverage_mean(precision_by_fp),
                   color=COLOURS[variant], lw=1.9, label=label)

    left.set(xlabel="Recall (sensitivity)", ylabel="Precision", xlim=(0, 1.0), ylim=(0, 0.75))
    left.set_title("Precision-recall", fontsize=10.5)
    left.grid(alpha=0.22)
    left.legend(fontsize=8, loc="upper right")

    right.axvline(BUDGET, color="black", ls="--", lw=1.0, alpha=0.6)
    right.set(xlabel="False positives / image", ylabel="Precision", xlim=(0, 2.0), ylim=(0, 0.75))
    right.set_title("Precision against operating point", fontsize=10.5)
    right.grid(alpha=0.22)

    figure.suptitle("Precision on the NODE21 test split, target {}x{} "
                    "(recovered from the stored FROC counts)".format(target, target),
                    fontsize=11)
    figure.text(0.5, 0.012,
                "Precision is low for every variant including bilinear: at 0.25 FP/img the "
                "detector is right about a minority of what it flags.",
                ha="center", fontsize=8, color="dimgray")
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = OUT_DIR / "fig6_precision_t{}{}.png".format(target, tag)
    figure.savefig(path, dpi=180)
    plt.close(figure)

    return path


def print_metrics_table(runs, target, tag=""):
    """The precision-side numbers, at the matched budget and at the fixed threshold."""
    import statistics

    by_variant = {}

    for (variant, seed, recipe), run in runs.items():
        box_metrics = run["box_metrics"]["iou0.25"]
        points = precision_recall_from_froc(box_metrics["froc"],
                                            box_metrics["n_nodules"], box_metrics["n_images"])
        matched = at_budget(points, box_metrics["n_nodules"], box_metrics["n_images"])
        true_positives, false_positives = box_metrics["tp"], box_metrics["fp"]
        fixed_precision = (true_positives / (true_positives + false_positives)
                           if true_positives + false_positives else 0.0)

        by_variant.setdefault(variant, []).append({
            "matched": matched,
            "fixed_recall": box_metrics["sensitivity"],
            "fixed_precision": fixed_precision,
            "fixed_fp": box_metrics["fp_per_image"],
        })

    print()
    print("=" * 96)
    print("TARGET {}x{} - test split, IoU 0.25, {} test nodules".format(
        target, target, next(iter(runs.values()))["box_metrics"]["iou0.25"]["n_nodules"]))
    print("=" * 96)
    print()
    # Whether these numbers are trustworthy depends on how densely the STORED curve
    # samples the low-FP region, so check it rather than assuming. Curves written before
    # 2026-08-31 sampled 25 points by rank and left one below 1 FP/img, making any value
    # read at 0.25 a chord across a gap that understates recall ~2x.
    density = min(
        sum(1 for point in run["box_metrics"]["iou0.25"]["froc"] if point["fp_per_image"] < 1.0)
        for run in runs.values())

    if density < 10:
        print("  At a MATCHED budget of {} FP/img -- SPARSE CURVE, DO NOT QUOTE:".format(BUDGET))
        print("    Only {} stored point(s) below 1 FP/img, so this is a chord across a gap".format(density))
        print("    and understates recall by roughly 2x. Re-evaluate with:")
        print("      python -m scripts.eval.reevaluate_runs --run-group <group>")
        print("    or measure directly with:")
        print("      python -m scripts.eval.paired_nodule_test --a <runA> --b <runB>")
    else:
        print("  At a MATCHED budget of {} FP/img  ({} stored points below 1 FP/img):".format(
            BUDGET, density))
    print("    {:<26}{:>4}{:>11}{:>12}{:>8}{:>12}".format(
        "variant", "n", "recall", "precision", "F1", "score thr"))

    for variant in ORDER:
        if variant not in by_variant: continue

        entries = by_variant[variant]
        mean = lambda key: statistics.mean(e["matched"][key] for e in entries)
        print("    {:<26}{:>4}{:>11.3f}{:>12.3f}{:>8.3f}{:>12.3f}".format(
            LABELS[variant], len(entries), mean("recall"), mean("precision"),
            mean("f1"), mean("threshold")))

    print()
    print("  At the FIXED score threshold 0.3 (shown only to expose operating-point drift):")
    print("    {:<26}{:>4}{:>11}{:>12}{:>10}".format(
        "variant", "n", "recall", "precision", "FP/img"))

    for variant in ORDER:
        if variant not in by_variant: continue

        entries = by_variant[variant]
        mean = lambda key: statistics.mean(e[key] for e in entries)
        print("    {:<26}{:>4}{:>11.3f}{:>12.3f}{:>10.3f}".format(
            LABELS[variant], len(entries), mean("fixed_recall"),
            mean("fixed_precision"), mean("fixed_fp")))

    print()
    print("  Image-level specificity IS available now, in")
    print("    box_metrics.iou0.25.image_level (sensitivity, specificity, roc_auc).")


def main():
    # Module-level so every figure-writing helper below picks it up without
    # threading the path through each signature
    global OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    # No choices= here. The allowlist that used to sit on this argument predated the
    # 768 and 1024 runs and would have rejected them, which is the same failure the
    # --variant allowlist caused in the v15 sweep.
    parser.add_argument("--target", type=int, default=384)
    parser.add_argument("--input-size", type=int, default=1024,
                        help="Resizer INPUT resolution. Runs at different inputs face "
                             "different decimation ratios and are never merged.")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Restrict the training-curve panels to one seed.")
    parser.add_argument("--recipe", type=str, default=None,
                        help="Recipe tag to plot, e.g. '' for NODE21-only ConvNeXt or "
                             "'node21_vindr+vin2048+vnr1'. Runs from different recipes are "
                             "NOT comparable and must not share a figure. Omit to list what "
                             "is available.")
    parser.add_argument("--run-group", type=str, default=None,
                        help="Substring filter on run_group, e.g. 'v14'. Needed to separate "
                             "runs from before and after the 2026-08-27 INTER_AREA fix, "
                             "which is not recorded in metrics.json and so cannot appear in "
                             "the recipe tag.")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Directory for the figures. Point this somewhere separate when "
                             "regenerating an old result set, so the new files cannot be "
                             "confused with the current ones.")
    arguments = parser.parse_args()

    OUT_DIR = Path(arguments.out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = load(arguments.target, recipe=arguments.recipe, run_group=arguments.run_group,
                    input_size=arguments.input_size)

    if not all_runs:
        print("No fcos runs at target {}. Run from the Model/ directory.".format(arguments.target))
        return

    # Refuse to blend recipes into one figure. Averaging a NODE21-only arm with a pooled
    # VinDr arm, or a ConvNeXt arm with a DenseNet one, produces a plausible-looking bar
    # that means nothing - which is exactly how this defect went unnoticed before.
    present = sorted({key[2] for key in all_runs})

    if arguments.recipe is None and len(present) > 1:
        print("Multiple recipes present at target {}; they are NOT comparable.".format(
            arguments.target))
        print("Re-run with --recipe set to exactly one of:")

        for tag in present:
            variants = sorted({k[0] for k in all_runs if k[2] == tag})
            seeds = sorted({k[1] for k in all_runs if k[2] == tag})
            print("  --recipe '{}'   variants={} seeds={}".format(
                tag, ",".join(v.replace("node21_talebi", "talebi") for v in variants), seeds))

        return

    if present:
        print("recipe: '{}'".format(present[0]))

    # Sanitised for a filename, and prefixed so the canonical recipe keeps its old name.
    file_tag = ""

    if present and present[0]:
        file_tag = "_" + present[0].replace("+", "-").replace(".", "")

    if arguments.run_group:
        # Commas survive into filenames and make the path awkward to quote in a shell,
        # so a multi-group request joins with "+" instead
        file_tag += "_" + "+".join(g.strip() for g in arguments.run_group.split(",") if g.strip())

    # Input belongs in the filename for the same reason the recipe does: two sweeps that
    # differ only by input would otherwise overwrite each other.
    if arguments.input_size != 1024:
        file_tag += "_i{}".format(arguments.input_size)

    # Curves are per-run, so restrict to one seed to keep the grid readable
    curve_seed = arguments.seed if arguments.seed is not None else 42
    # One column per VARIANT, preferring the requested seed but falling back to any seed
    # that still has history. Runs rebuilt by recover_metrics.py lost theirs, and pinning a
    # single seed silently dropped bilinear from this figure entirely at target 384.
    curve_runs = {}

    for variant in ORDER:
        candidates = [(k, v) for k, v in all_runs.items()
                      if k[0] == variant and v.get("history")]

        if not candidates: continue

        preferred = [kv for kv in candidates if kv[0][1] == curve_seed]
        key, run = (preferred or sorted(candidates, key=lambda kv: kv[0][1]))[0]
        curve_runs[key] = run

    if curve_runs:
        path = figure_training_curves(curve_runs, arguments.target, plt, tag=file_tag)
        print("wrote {}".format(path))
    else:
        print("no runs at seed {} for the curve panels".format(curve_seed))

    path = figure_precision_recall(all_runs, arguments.target, plt, tag=file_tag)
    print("wrote {}".format(path))

    print_metrics_table(all_runs, arguments.target, tag=present[0] if present else "")


def _demo():
    """Assert-based self-check on the precision recovery and budget interpolation."""
    # 10 nodules, 100 images. At sensitivity 0.5 and 0.1 FP/img: TP=5, FP=10, precision=1/3
    froc = [{"sensitivity": 0.0, "fp_per_image": 0.0,  "threshold": 0.9},
            {"sensitivity": 0.5, "fp_per_image": 0.1,  "threshold": 0.5},
            {"sensitivity": 0.8, "fp_per_image": 1.0,  "threshold": 0.1}]
    points = precision_recall_from_froc(froc, n_nodules=10, n_images=100)

    assert abs(points[1][1] - 5 / 15) < 1e-9, points[1]
    assert abs(points[2][1] - 8 / 108) < 1e-9, points[2]
    assert points[0][1] == 0.0

    # Interpolating at exactly a sampled FP value must return that point, not a blend
    at = at_budget(points, 10, 100, 0.1)
    assert abs(at["recall"] - 0.5) < 1e-9, at
    assert abs(at["precision"] - 5 / 15) < 1e-9, at
    expected_f1 = 2 * (5 / 15) * 0.5 / ((5 / 15) + 0.5)
    assert abs(at["f1"] - expected_f1) < 1e-9, at

    # Beyond the measured range it clamps rather than extrapolating to nonsense
    assert abs(at_budget(points, 10, 100, 99.0)["recall"] - 0.8) < 1e-9
    assert at_budget([], 10, 100) is None

    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        main()

