"""
Figures for the supervisor meeting: bilinear vs the Talebi family.

Deliberately restricted to bilinear, node21_talebi_zeroinit and node21_talebi_zeroinit_attn.
The adaptive-resizer family (warp, gated, dilated) is excluded.

ALL FOUR VARIANTS NOW HAVE FCOS RUNS AT 1024->384 (v11, 2026-08-24), so the family shares
one chart. Earlier versions of this script had to fence node21_talebi_zeroinit_attn onto a
separate centernet_det panel and drop node21_talebi entirely; both gaps are closed.

Figure 1 is still bilinear vs node21_talebi_zeroinit only, because those are the only two
variants with runs at more than one target resolution.

Pre-v6 runs are never read: this script requires box_metrics, which only exist from v6
onward. Anything earlier used the 85.7 px hit tolerance (trap 4.3 in
docs/context/HANDOVER.md) - 38% of image width at target 224 - and is not interpretable.

EVERYTHING IS READ AT A MATCHED FALSE-POSITIVE BUDGET of 0.25 FP/img, which is what NODE21
itself reports at. Never at a fixed score threshold - at threshold 0.3 two variants in this
project read as wins and both are negative once the operating point is controlled.

Run: python scripts/eval/plot_supervisor_figures.py
     python scripts/eval/plot_supervisor_figures.py --demo
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

# Shared with plot_run_diagnostics rather than duplicated: one definition of what makes
# two runs incomparable, so the two scripts cannot drift apart on it.
from scripts.eval.plot_run_diagnostics import recipe_key

BUDGET = 0.25
BINOMIAL_SE = 0.041                       # n = 147 test nodules
OUT_DIR = Path("outputs/figures")

COLOURS = {
    "bilinear":                    "#4C72B0",
    "node21_talebi":               "#DD8452",
    "node21_talebi_zeroinit":      "#C44E52",
    "node21_talebi_zeroinit_attn": "#8172B2",
}
LABELS = {
    "bilinear":                    "Bilinear",
    "node21_talebi":               "Talebi\n(random init)",
    "node21_talebi_zeroinit":      "Talebi\n(zero-init)",
    "node21_talebi_zeroinit_attn": "Talebi + attention\n(zero-init)",
}

# Reading order for the family chart: baseline, then the paper's method, then the two
# modifications, so the chart reads left to right as the design decisions were made.
FAMILY_ORDER = ["bilinear", "node21_talebi",
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


def sensitivity_at_budget(curve, budget=BUDGET):
    """Interpolate a FROC curve at a false-positive budget."""
    if not curve: return None

    fp = np.array([p["fp_per_image"] for p in curve])
    se = np.array([p["sensitivity"] for p in curve])
    order = np.argsort(fp)

    return float(np.interp(budget, fp[order], se[order], left=0.0, right=se[order][-1]))


def load(head, recipe="", run_group=None, exclude_run_group=None):
    """
    (variant, target) -> {seed: {...}} for one head, squash runs only, at input 1024.

    RESTRICTED TO ONE RECIPE. Without this a v14 pooled-VinDr run and a v12 NODE21-only
    run at the same (variant, target, seed) overwrite each other silently, and a v13
    DenseNet-CXR run merges into the ConvNeXt bars - which is exactly what happened to
    the t384 panels of the figures deleted on 2026-08-31.

    Args:
        head: detection head to select
        recipe: exact recipe tag (see recipe_key); "" is canonical NODE21-only ConvNeXt
        run_group: optional substring filter, for axes not recorded in metrics.json
            such as the 2026-08-27 INTER_AREA dataloader fix
    """
    runs = {}

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != head or run.get("letterbox"): continue
        if run.get("input_size") != 1024: continue
        if run.get("variant") not in COLOURS: continue

        box_metrics = (run.get("box_metrics") or {}).get("iou0.25")

        if not box_metrics or not box_metrics.get("froc"): continue

        if recipe is not None and recipe_key(run) != recipe: continue
        if not run_group_matches(run, run_group): continue
        # The 2026-08-27 INTER_AREA dataloader fix is not recorded in metrics.json, so
        # pre- and post-fix runs share a recipe tag. Only v14 is post-fix, and the
        # resolution sweep legitimately spans v9/v11/v12, so excluding is the usable
        # filter here - restricting to a single group would empty the sweep.
        if exclude_run_group and exclude_run_group in str(run.get("run_group", "")): continue

        key = (run["variant"], run["target_size"])
        runs.setdefault(key, {})[run["seed"]] = {
            "matched":     sensitivity_at_budget(box_metrics["froc"]),
            "attribution": run.get("attribution") or {},
            # How many stored curve points sit below 1 FP/img. Rank-sampled curves (pre
            # 2026-08-31) leave a gap straddling the 0.25 budget, so the reported number is
            # a chord across it. Threshold-grid curves land ~30 points there and need no
            # caveat, so the warning below is printed only when the density says so.
            "n_below_1fp": sum(1 for point in box_metrics["froc"]
                               if point["fp_per_image"] < 1.0),
        }

    return runs


def summarise(entry):
    """Mean across seeds, plus the observed min/max as an honest error bar."""
    values = [v["matched"] for v in entry.values() if v["matched"] is not None]

    if not values: return None

    return {
        "mean": statistics.mean(values),
        "low":  min(values),
        "high": max(values),
        "n":    len(values),
    }


def figure_resolution_sweep(runs, plt, tag=""):
    """Headline: does the learned resizer beat bilinear, and what actually moves the needle."""
    # Only targets where BOTH arms exist, so every group is a paired comparison
    targets = [t for t in (224, 256, 384, 512)
               if ("bilinear", t) in runs and ("node21_talebi_zeroinit", t) in runs]

    variants = ["bilinear", "node21_talebi_zeroinit"]
    figure, axes = plt.subplots(figsize=(9.0, 5.8))
    width = 0.36
    positions = np.arange(len(targets))
    tops = {}

    for offset, variant in zip((-width / 2, width / 2), variants):
        means, caps, counts = [], [], []
        lower, upper = [], []

        for index, target in enumerate(targets):
            stat = summarise(runs[(variant, target)])
            means.append(stat["mean"])
            lower.append(stat["mean"] - stat["low"])
            upper.append(stat["high"] - stat["mean"])
            caps.append(stat["high"])
            counts.append(stat["n"])
            tops[(variant, target)] = (positions[index] + offset, stat["mean"])

        bars = axes.bar(positions + offset, means, width, label=LABELS[variant],
                        color=COLOURS[variant], edgecolor="black", linewidth=0.6)
        axes.errorbar(positions + offset, means, yerr=[lower, upper], fmt="none",
                      ecolor="black", capsize=4, lw=1.0)

        # Value above the error-bar cap, seed count inside the bar
        for bar, mean, cap, n in zip(bars, means, caps, counts):
            centre = bar.get_x() + bar.get_width() / 2
            axes.text(centre, cap + 0.016, f"{mean:.3f}", ha="center", va="bottom", fontsize=9.5)
            axes.text(centre, 0.010, f"n={n}", ha="center", va="bottom",
                      fontsize=8, color="white")

    # A bilinear 384 -> 512 connector used to be drawn here, captioned "+X from backbone
    # resolution alone". REMOVED 2026-09-14: it compared a 3-seed mean against a 1-seed
    # point, and once the 384 cell was seeded the gap fell to -0.006, so the caption
    # rendered as "+-0.01 from backbone resolution alone" - annotating noise in bold green
    # and inviting it to be read as a finding. Resolution effects belong in fig10, which
    # plots every metric against resolution with the seed count visible per point.

    axes.set(xticks=positions, xlabel="Backbone input resolution (target size)",
             ylabel="Sensitivity at {} FP / image".format(BUDGET), ylim=(0, 0.68))
    axes.set_xticklabels(["{}x{}".format(t, t) for t in targets])
    axes.set_title("NODE21 nodule detection at a matched false-positive budget\n"
                   "ConvNeXt-T + FPN, FCOS head, 147 test nodules", fontsize=11.5)
    axes.legend(loc="upper left", frameon=True)
    axes.grid(axis="y", alpha=0.25)

    # Single-seed groups cannot support a claim; say so on the figure, not just in the talk
    single = [t for t in targets
              if summarise(runs[("bilinear", t)])["n"] == 1
              or summarise(runs[("node21_talebi_zeroinit", t)])["n"] == 1]

    caption = ("Error bars = observed seed range. Binomial SE at n=147 nodules is "
               "+/-{:.3f}, so differences below that are not resolvable.".format(BINOMIAL_SE))

    if single:
        caption += ("\n" + ", ".join("{}x{}".format(t, t) for t in single) +
                    " is single-seed - treat as indicative only.")

    figure.text(0.5, 0.015, caption, ha="center", fontsize=8, color="dimgray")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    path = OUT_DIR / "fig1_resolution_sweep{}.png".format(tag)
    figure.savefig(path, dpi=200)
    plt.close(figure)

    return path, targets


def figure_mechanism(runs, plt, tag=""):
    """Why the residual does not help: the loss asks, the residual answers elsewhere."""
    entry = runs.get(("node21_talebi_zeroinit", 384))

    if not entry: return None

    gradient = [v["attribution"].get("gradient_enrichment") for v in entry.values()]
    residual = [v["attribution"].get("residual_enrichment") for v in entry.values()]
    gradient = [g for g in gradient if g is not None]
    residual = [r for r in residual if r is not None]

    if not gradient or not residual: return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.8))

    # Left: the loss's instruction, hugely concentrated on nodules
    mean_gradient = statistics.mean(gradient)
    left.bar([0], [mean_gradient], 0.5, color="#55A868", edgecolor="black", linewidth=0.6)
    left.errorbar([0], [mean_gradient],
                  yerr=[[mean_gradient - min(gradient)], [max(gradient) - mean_gradient]],
                  fmt="none", ecolor="black", capsize=5)
    left.axhline(1.0, color="black", ls="--", lw=1.2)
    left.text(0.30, 1.15, "1.0 = a nodule is treated\nlike any other patch",
              fontsize=8.5, va="bottom")
    left.set(xticks=[0], yscale="log", ylim=(0.5, 500),
             ylabel="Enrichment vs size-matched control boxes")
    left.set_xticklabels(["gradient into the resizer\ndL / d(residual)"])
    left.set_title("The loss ASKS for nodules:\n{:.0f}x concentrated".format(mean_gradient),
                   fontsize=10.5)
    left.grid(axis="y", alpha=0.25)

    # Right: what the residual actually does, for EVERY learned variant. Plotting all three
    # is the point - the attention block was added specifically to make the residual attend
    # to nodules, and it did not.
    learned = [v for v in FAMILY_ORDER if v != "bilinear" and (v, 384) in runs]
    residual_means = []

    for index, variant in enumerate(learned):
        values = [e["attribution"].get("residual_enrichment")
                  for e in runs[(variant, 384)].values()]
        values = [v for v in values if v is not None]

        if not values:
            residual_means.append(None)
            continue

        mean = statistics.mean(values)
        residual_means.append(mean)
        right.bar([index], [mean], 0.6, color=COLOURS[variant],
                  edgecolor="black", linewidth=0.6)
        right.errorbar([index], [mean],
                       yerr=[[mean - min(values)], [max(values) - mean]],
                       fmt="none", ecolor="black", capsize=4)
        right.text(index, max(values) + 0.03, "{:.3f}".format(mean),
                   ha="center", va="bottom", fontsize=10)

    right.axhline(1.0, color="black", ls="--", lw=1.2)
    right.text(len(learned) - 0.5, 1.03, "1.0 = no preference",
               ha="right", fontsize=8.5, va="bottom")
    right.set(xticks=range(len(learned)), ylim=(0, 1.35),
              ylabel="Enrichment vs size-matched control boxes")
    right.set_xticklabels([LABELS[v].replace("Talebi", "").replace("\n", " ").strip()
                           for v in learned], fontsize=9)
    right.set_title("Every residual ANSWERS elsewhere:\nall below 1, all AVOID nodules",
                    fontsize=10.5)
    right.grid(axis="y", alpha=0.25)

    figure.suptitle("Why the learned residual does not help: it lands on ribs and vessels, "
                    "not on nodules", fontsize=11.5)
    figure.text(0.5, 0.015,
                "1024->384, FCOS, 2-3 seeds each. Each nodule scored against 8 size-matched "
                "control boxes at random non-overlapping positions in the same film.\n"
                "The attention block was added to make the residual attend to nodules; "
                "enrichment did not rise.",
                ha="center", fontsize=8, color="dimgray")

    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = OUT_DIR / "fig2_mechanism_enrichment{}.png".format(tag)
    figure.savefig(path, dpi=200)
    plt.close(figure)

    return path


def figure_family(runs, plt, tag=""):
    """
    The whole Talebi family against bilinear at one resolution, on the current head.

    Replaces the earlier centernet_det panel: as of v11 every variant here has FCOS runs at
    1024->384, so the comparison no longer has to be split across two heads.
    """
    present = [(v, summarise(runs[(v, 384)])) for v in FAMILY_ORDER if (v, 384) in runs]

    if len(present) < 2: return None

    figure, axes = plt.subplots(figsize=(9.0, 5.8))
    baseline = present[0][1]["mean"]

    for index, (variant, stat) in enumerate(present):
        axes.bar([index], [stat["mean"]], 0.6, color=COLOURS[variant],
                 edgecolor="black", linewidth=0.6)

        if stat["n"] > 1:
            axes.errorbar([index], [stat["mean"]],
                          yerr=[[stat["mean"] - stat["low"]], [stat["high"] - stat["mean"]]],
                          fmt="none", ecolor="black", capsize=5, lw=1.0)

        axes.text(index, stat["high"] + 0.008, "{:.3f}".format(stat["mean"]),
                  ha="center", va="bottom", fontsize=10.5)
        axes.text(index, 0.006, "n={}".format(stat["n"]), ha="center", va="bottom",
                  fontsize=8, color="white")

        if index > 0:
            delta = stat["mean"] - baseline
            axes.text(index, stat["high"] + 0.030, "{:+.3f}".format(delta),
                      ha="center", va="bottom", fontsize=9.5, color="firebrick",
                      fontweight="bold")

    # The noise floor, drawn so the eye cannot read a difference smaller than it
    axes.axhspan(baseline - BINOMIAL_SE, baseline + BINOMIAL_SE, color="grey", alpha=0.13)
    axes.axhline(baseline, color="#4C72B0", ls="--", lw=1.2)
    axes.text(len(present) - 0.45, baseline + BINOMIAL_SE + 0.004,
              "bilinear +/- binomial SE ({:.3f})".format(BINOMIAL_SE),
              ha="right", va="bottom", fontsize=8.5, color="dimgray")

    axes.set(xticks=range(len(present)),
             ylabel="Sensitivity at {} FP / image".format(BUDGET), ylim=(0, 0.44))
    axes.set_xticklabels([LABELS[v] for v, _ in present], fontsize=9.5)
    axes.set_title("The Talebi family against bilinear, 1024->384\n"
                   "ConvNeXt-T + FPN, FCOS head, matched false-positive budget",
                   fontsize=11.5)
    axes.grid(axis="y", alpha=0.25)

    figure.text(0.5, 0.015,
                "Every learned variant sits below bilinear, and every difference is inside "
                "the noise floor. Zero-init recovers most of what\nrandom init loses; "
                "the attention block adds nothing over plain zero-init.",
                ha="center", fontsize=8, color="dimgray")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    path = OUT_DIR / "fig3_talebi_family{}.png".format(tag)
    figure.savefig(path, dpi=200)
    plt.close(figure)

    return path


def figure_froc_grid(plt, recipe="", run_group=None, exclude_run_group=None, tag=""):
    """
    Full FROC curves, one panel per target resolution.

    The bar charts read each curve at a single budget. This shows the whole operating
    characteristic, which is a stronger statement: where two curves coincide there is no
    threshold at which one beats the other, so the null cannot be recovered by retuning.

    Curves are averaged across seeds on a shared false-positive grid. Reads the raw `froc`
    block rather than the interpolated summary, so it needs its own loader.
    """
    curves = {}

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != "fcos" or run.get("letterbox"): continue
        if run.get("input_size") != 1024 or run.get("variant") not in COLOURS: continue

        box_metrics = (run.get("box_metrics") or {}).get("iou0.25")

        if not box_metrics or not box_metrics.get("froc"): continue

        if recipe is not None and recipe_key(run) != recipe: continue
        if not run_group_matches(run, run_group): continue
        # The 2026-08-27 INTER_AREA dataloader fix is not recorded in metrics.json, so
        # pre- and post-fix runs share a recipe tag. Only v14 is post-fix, and the
        # resolution sweep legitimately spans v9/v11/v12, so excluding is the usable
        # filter here - restricting to a single group would empty the sweep.
        if exclude_run_group and exclude_run_group in str(run.get("run_group", "")): continue

        key = (run["variant"], run["target_size"])
        curves.setdefault(key, {})[run["seed"]] = box_metrics["froc"]

    targets = sorted({t for _, t in curves})

    if not targets: return None

    grid = np.linspace(0, 2.0, 300)
    figure, axes_row = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.6),
                                    sharey=True)

    if len(targets) == 1: axes_row = [axes_row]

    for axes, target in zip(axes_row, targets):
        for variant in FAMILY_ORDER:
            if (variant, target) not in curves: continue

            seeds = curves[(variant, target)]
            interpolated = []

            for curve in seeds.values():
                fp = np.array([p["fp_per_image"] for p in curve])
                se = np.array([p["sensitivity"] for p in curve])
                order = np.argsort(fp)
                interpolated.append(np.interp(grid, fp[order], se[order],
                                              left=0.0, right=se[order][-1]))

            mean = np.mean(interpolated, axis=0)
            axes.plot(grid, mean, color=COLOURS[variant], lw=1.9,
                      label="{}  (n={})".format(LABELS[variant].replace("\n", " "), len(seeds)))

        axes.axvline(BUDGET, color="black", ls="--", lw=1.0, alpha=0.6)
        axes.set(xlabel="False positives / image", xlim=(0, 2.0), ylim=(0, 0.9))
        axes.set_title("target {}x{}".format(target, target), fontsize=10.5)
        axes.grid(alpha=0.22)

    axes_row[0].set_ylabel("Sensitivity")

    # One shared legend: no single panel holds every variant, so a per-panel legend would
    # silently under-report which arms are present elsewhere in the row.
    handles, labels = {}, []

    for axes in axes_row:
        for line, label in zip(*axes.get_legend_handles_labels()):
            name = label.split("  (n=")[0]

            if name not in handles:
                handles[name] = line
                labels.append(name)

    figure.legend([handles[name] for name in labels], labels,
                  loc="upper center", ncol=len(labels), fontsize=8.5,
                  bbox_to_anchor=(0.5, 0.90), frameon=False)

    figure.suptitle("FROC curves, ConvNeXt-T + FPN, FCOS head, input 1024. "
                    "Dashed line = the 0.25 FP/img budget NODE21 reports at.",
                    fontsize=11)
    figure.text(0.5, 0.015,
                "Where curves coincide there is no operating point at which one variant "
                "beats another - a stronger statement than a single matched-budget number.",
                ha="center", fontsize=8, color="dimgray")
    figure.tight_layout(rect=(0, 0.06, 1, 0.86))
    path = OUT_DIR / "fig4_froc_grid{}.png".format(tag)
    figure.savefig(path, dpi=200)
    plt.close(figure)

    return path


def print_setup_summary(runs, targets):
    """The configuration facts needed to talk through architecture and training."""
    print()
    print("=" * 78)
    print("SETUP SUMMARY - for the architecture and training part of the talk")
    print("=" * 78)
    print("""
  Data        NODE21, 4,882 frontal CXRs, 1,134 with nodules (1,476 boxes).
              Test split fixed by test_seed=0: 521 images, 147 nodules.
              val_frac 0.20. JSRT excluded since v3 - training on both inflates
              NODE21 test sensitivity through extra nodule exposure.

  Pipeline    image -> LDM (resizer) -> 3-channel replicate -> ImageNet normalise
              -> ConvNeXt-T + FPN (~31M params) -> head -> boxes

  Backbone    ConvNeXt-tiny + FPN, ImageNet-pretrained. The head reads only
              fpn_features["stage1"], so the FPN acts as semantic enrichment of one
              high-resolution level rather than as multi-scale detection.

  Head        FCOS (classification + LTRB + centerness). Chosen over CenterNet-det for
              two reasons: it is better (+0.146 bilinear, +0.214 Talebi at 1024->512),
              and its seed spread is 0.020 against CenterNet-det's 0.082. Every effect
              in this project is 0.01-0.08, so on CenterNet-det none would be readable.
              FCOS supervises every cell inside a box rather than one cell per nodule -
              6x more supervised cells at target 224, 32x at 512.

  Loss        Task loss only (focal_v5). No perceptual or reconstruction term on the
              resizer. This is faithful to Talebi and Milanfar, where the resizer is
              trained purely by the downstream task gradient.

  Optimiser   AdamW, weight decay 0.05. LR 2e-4, constant. The backbone runs at
              0.1 x LR (backbone_lr_factor) so pretrained weights are not destroyed
              while the randomly-initialised resizer and head train at full rate.
              Gradient clipping at norm 1.0. bfloat16 autocast.

  Training    batch 8 (grad_accum holds the effective batch at 8 when memory forces a
              smaller micro-batch at target 512), up to 50 epochs, early stopping with
              patience 15 on validation sensitivity. Runs converged at epoch 21-30.
              Augmentation profile v3: rotation +/-1-5 deg, no translation, additive
              noise sigma=0.01 at p=0.3.

  Evaluation  Box IoU matching at 0.25 (0.5 also reported), one sample per image.
              Reported at a MATCHED false-positive budget of 0.25 FP/img, which is what
              NODE21 itself reports at. Never at a fixed score threshold: FCOS scores
              sqrt(cls x centerness), so a fixed 0.3 threshold sits far right on the
              curve, and two variants in this project read as wins there that are
              negative once the operating point is controlled.

  Statistics  WRONG YARDSTICK, kept only for continuity with earlier slides. +/-0.041 is
              the MARGINAL binomial SE of ONE proportion, but both arms are scored on the
              SAME 147 nodules, so the comparison is PAIRED. Use McNemar on the discordant
              nodules (paired SE is typically ~+/-0.030):
                python -m scripts.eval.paired_nodule_test --a <runA> --b <runB>
""")

    print("  Numbers behind figure 1 (sensitivity at {} FP/img)".format(BUDGET))

    # Measure the curve density rather than assuming it. Printing the interpolation caveat
    # over threshold-grid curves is not a harmless extra warning: it tells the reader to
    # discount numbers that are in fact measured at the budget.
    sparsest = min((seed_entry["n_below_1fp"]
                    for target in targets
                    for variant in ("bilinear", "node21_talebi_zeroinit")
                    for seed_entry in runs.get((variant, target), {}).values()),
                   default=0)

    if sparsest < 10:
        print("  -- INTERPOLATED. The sparsest curve here holds only {} points below".format(sparsest))
        print("  1 FP/img. Rank-sampled curves leave 0.00 to 0.65 FP/img with nothing")
        print("  between, so these are chords across a gap and understate recall by")
        print("  roughly 2x. On the v14 runs the SIGN of the delta flipped once measured.")
        print("  Refresh them: python scripts/eval/reevaluate_runs.py")

    else:
        print("  -- MEASURED on the fixed threshold grid ({} points below 1 FP/img on the".format(sparsest))
        print("  sparsest curve), so the budget is read off the curve, not across a gap.")
    print("    {:>8}{:>12}{:>16}{:>10}".format("target", "bilinear", "Talebi 0-init", "delta"))

    for target in targets:
        b = summarise(runs[("bilinear", target)])
        t = summarise(runs[("node21_talebi_zeroinit", target)])
        delta = t["mean"] - b["mean"]
        note = ""

        if min(b["n"], t["n"]) == 1:
            note = "  single-seed"
        elif abs(delta) > BINOMIAL_SE:
            note = "  outside SE"

        print("    {:>8}{:>12.3f}{:>16.3f}{:>+10.3f}{}".format(
            target, b["mean"], t["mean"], delta, note))

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=str, default="",
                        help="Recipe tag to plot. Default "" is the canonical NODE21-only "
                             "ConvNeXt recipe. Runs from different recipes are NOT "
                             "comparable and must never share a figure.")
    parser.add_argument("--run-group", type=str, default=None,
                        help="Substring filter on run_group, e.g. v12. Needed for axes not "
                             "recorded in metrics.json, such as the INTER_AREA fix.")
    parser.add_argument("--exclude-run-group", type=str, default=None,
                        help="Substring of run_group to EXCLUDE, e.g. v14 to keep only "
                             "pre-INTER_AREA-fix runs. The resolution sweep spans v9/v11/v12 "
                             "so --run-group alone cannot express it.")
    arguments = parser.parse_args()

    tag = ""

    if arguments.recipe:
        tag += "_" + arguments.recipe.replace("+", "-").replace(".", "")

    if arguments.run_group:
        # Commas survive into filenames and make the path awkward to quote in a shell,
        # so a multi-group request joins with "+" instead
        tag += "_" + "+".join(g.strip() for g in arguments.run_group.split(",") if g.strip())

    if arguments.exclude_run_group:
        tag += "_no" + arguments.exclude_run_group

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load("fcos", recipe=arguments.recipe, run_group=arguments.run_group,
                exclude_run_group=arguments.exclude_run_group)
    print("recipe: '{}'{}".format(arguments.recipe,
          "  run_group~" + arguments.run_group if arguments.run_group else ""))

    if not runs:
        print("No fcos runs found. Run from the Model/ directory.")
        return

    path1, targets = figure_resolution_sweep(runs, plt, tag=tag)
    print("wrote {}".format(path1))

    path2 = figure_mechanism(runs, plt, tag=tag)
    print("wrote {}".format(path2) if path2 else "skipped fig2 (no attribution data)")

    path3 = figure_family(runs, plt, tag=tag)
    print("wrote {}".format(path3) if path3 else "skipped fig3 (family incomplete at t384)")

    path4 = figure_froc_grid(plt, recipe=arguments.recipe, run_group=arguments.run_group,
                             exclude_run_group=arguments.exclude_run_group, tag=tag)
    print("wrote {}".format(path4) if path4 else "skipped fig4 (no froc curves)")

    print_setup_summary(runs, targets)


def _demo():
    """Assert-based self-check on the curve reader and the seed summariser."""
    curve = [{"fp_per_image": 0.0, "sensitivity": 0.0},
             {"fp_per_image": 0.5, "sensitivity": 0.5},
             {"fp_per_image": 2.0, "sensitivity": 0.9}]
    assert abs(sensitivity_at_budget(curve, 0.25) - 0.25) < 1e-9
    assert sensitivity_at_budget([], 0.25) is None
    assert abs(sensitivity_at_budget(curve, 99.0) - 0.9) < 1e-9      # clamps, no extrapolation

    entry = {42: {"matched": 0.30, "attribution": {}},
             7:  {"matched": 0.32, "attribution": {}},
             123: {"matched": 0.28, "attribution": {}}}
    stat = summarise(entry)
    assert stat["n"] == 3 and abs(stat["mean"] - 0.30) < 1e-9
    assert stat["low"] == 0.28 and stat["high"] == 0.32

    assert summarise({1: {"matched": None, "attribution": {}}}) is None
    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        main()
