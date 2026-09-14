"""
Read the v7 sweep and answer its two questions.

Arm V asks whether freezing the backbone reduces SEED VARIANCE, so it is summarised by
spread (range and standard deviation across seeds), not by the mean.

Arm I asks whether dense box supervision helps the learned resizer SPECIFICALLY, so it
is summarised by the interaction - the FCOS gain for Talebi minus the FCOS gain for
bilinear - not by the head main effect.

Run: python scripts/eval/analyse_v7.py
"""

import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUN_GROUP = "v7_recipe_and_fcos"

# Binomial standard error on the NODE21 test split: 147 nodules at p ~ 0.55
BINOMIAL_SE = 0.041


def load_runs(run_group=RUN_GROUP):
    """Collect every completed run in the group, keyed by its full configuration."""
    runs = {}

    for path in glob.glob(f"outputs/**/*_{run_group}/*/metrics.json", recursive=True):
        d = json.load(open(path))
        box = d.get("box_metrics")

        if not box: continue

        key = (d.get("variant"), d.get("head"), d.get("input_size"), d.get("target_size"),
               d.get("seed"), d.get("freeze_backbone_epochs", 0), d.get("aug_profile", "v3"),
               round(float(d.get("backbone_lr_factor", 0.1)), 4))
        runs[key] = box

    return runs


def sensitivity(box, iou="iou0.25"):
    return box[iou]["sensitivity"]


def report_arm_v(runs):
    print("=" * 74)
    print("ARM V -- does freezing the backbone reduce SEED VARIANCE?")
    print("  talebi_zeroinit / centernet_det / 1024->224.  Read the SPREAD, not the mean.")
    print("=" * 74)

    seeds = (42, 7, 123)
    summary = {}

    for freeze in (0, 3):
        values, ap_values = [], []

        for seed in seeds:
            key = ("node21_talebi_zeroinit", "centernet_det", 1024, 224, seed, freeze, "v3", 0.1)
            box = runs.get(key)

            if box is None: continue

            values.append(sensitivity(box))
            ap_values.append(box["iou0.5"]["average_precision"])

        if len(values) < 2:
            print(f"\n  freeze={freeze}: only {len(values)} run(s) found - cannot measure spread")
            continue

        spread = max(values) - min(values)
        summary[freeze] = (values, spread, st.pstdev(values))

        listed = "  ".join(f"s{s}:{v:.3f}" for s, v in zip(seeds, values))
        print(f"\n  freeze={freeze}   {listed}")
        print(f"              mean {st.mean(values):.3f}   RANGE {spread:.3f}   sd {st.pstdev(values):.3f}"
              f"   AP@.5 mean {st.mean(ap_values):.3f}")

    if 0 in summary and 3 in summary:
        range_off, range_on = summary[0][1], summary[3][1]
        mean_off, mean_on = st.mean(summary[0][0]), st.mean(summary[3][0])

        print(f"\n  spread {range_off:.3f} (off) -> {range_on:.3f} (on),  change {range_on - range_off:+.3f}")
        print(f"  mean   {mean_off:.3f} (off) -> {mean_on:.3f} (on),  change {mean_on - mean_off:+.3f}")

        if range_on < range_off * 0.6:
            print("\n  => Freezing materially reduces run-to-run spread. Worth adopting as the")
            print("     default recipe, and worth reporting as a methodological finding.")
        elif range_on < range_off:
            print("\n  => Freezing reduces spread, but not decisively at n=3 seeds. Suggestive only.")
        else:
            print("\n  => Freezing does NOT reduce spread. The variance floor is the dataset, not")
            print("     backbone disruption. Report 0.10 as the resolution limit of this study.")

        if range_off < BINOMIAL_SE * 1.5:
            print(f"  NOTE: freeze=0 spread ({range_off:.3f}) is already near the binomial SE")
            print(f"        ({BINOMIAL_SE:.3f}); there may be little variance left to remove.")


def report_arm_i(runs):
    print("\n" + "=" * 74)
    print("ARM I -- does DENSE supervision help the learned resizer SPECIFICALLY?")
    print("  1024->512, where FCOS supervises ~32x more cells than CenterNet.")
    print("  Read the INTERACTION, not the head main effect.")
    print("=" * 74)

    seeds = (42, 7)
    cell = {}

    for variant in ("bilinear", "node21_talebi_zeroinit"):
        for head in ("centernet_det", "fcos"):
            values = []

            for seed in seeds:
                box = runs.get((variant, head, 1024, 512, seed, 0, "v3", 0.1))

                if box is not None: values.append(sensitivity(box))

            if values: cell[(variant, head)] = values

    if len(cell) < 4:
        print(f"\n  only {len(cell)}/4 cells present - arm incomplete")
        return

    print(f"\n  {'':<26}{'centernet_det':>16}{'fcos':>10}{'FCOS gain':>12}")
    gains = {}

    for variant in ("bilinear", "node21_talebi_zeroinit"):
        centernet = st.mean(cell[(variant, "centernet_det")])
        fcos      = st.mean(cell[(variant, "fcos")])
        gains[variant] = fcos - centernet
        print(f"  {variant:<26}{centernet:>16.3f}{fcos:>10.3f}{gains[variant]:>+12.3f}")

    interaction = gains["node21_talebi_zeroinit"] - gains["bilinear"]
    print(f"\n  INTERACTION (Talebi gain - bilinear gain): {interaction:+.3f}")
    print(f"  Binomial SE on a single cell is +/-{BINOMIAL_SE:.3f}; a difference of differences")
    print(f"  across 2 seeds carries roughly +/-{BINOMIAL_SE * 2:.3f}.")

    if abs(interaction) < BINOMIAL_SE * 2:
        print("\n  => Interaction is inside noise. No evidence that the resizer was")
        print("     gradient-starved; if FCOS helps, it helps both arms alike.")
    elif interaction > 0:
        print("\n  => Talebi gains MORE from dense supervision than bilinear does. This is the")
        print("     first positive evidence for the gradient-density argument - but confirm at")
        print("     a third seed before it goes in the report.")
    else:
        print("\n  => Talebi gains LESS from dense supervision than bilinear. The opposite of")
        print("     the gradient-starvation hypothesis.")

    # Per-seed detail, because a mean over 2 seeds hides a disagreement
    print("\n  per-seed detail:")
    for (variant, head), values in sorted(cell.items()):
        listed = "  ".join(f"s{s}:{v:.3f}" for s, v in zip(seeds, values))
        print(f"    {variant:<26}{head:<16}{listed}")


def report_arm_t(runs):
    print("\n" + "=" * 74)
    print("ARM T -- attention-subtraction branch, first evaluation under a valid metric")
    print("  1024->384, centernet_det. Paired against its own zeroinit baseline.")
    print("=" * 74)

    seeds = (42, 7)
    cell = {}

    for variant in ("node21_talebi_zeroinit", "node21_talebi_zeroinit_attn"):
        values = []

        for seed in seeds:
            box = runs.get((variant, "centernet_det", 1024, 384, seed, 0, "v3", 0.1))

            if box is not None: values.append((seed, box))

        if values: cell[variant] = values

    if len(cell) < 2:
        print(f"\n  only {len(cell)}/2 variants present - arm incomplete")
        return

    print(f"\n  {'variant':<32}{'sens@.25':>10}{'AP@.5':>9}{'FROC':>8}   per-seed")
    means = {}

    for variant, values in cell.items():
        sens = [sensitivity(b) for _, b in values]
        ap   = [b["iou0.5"]["average_precision"] for _, b in values]
        froc = [b["iou0.25"]["froc_auc"] for _, b in values]
        means[variant] = st.mean(sens)
        listed = "  ".join(f"s{s}:{sensitivity(b):.3f}" for s, b in values)
        print(f"  {variant:<32}{st.mean(sens):>10.3f}{st.mean(ap):>9.3f}{st.mean(froc):>8.3f}   {listed}")

    delta = means["node21_talebi_zeroinit_attn"] - means["node21_talebi_zeroinit"]
    print(f"\n  attention - zeroinit: {delta:+.3f}   (binomial SE +/-{BINOMIAL_SE:.3f} per cell)")

    if abs(delta) < BINOMIAL_SE * 1.5:
        print("\n  => Inside noise. The attention-subtraction branch neither helps nor hurts")
        print("     under a valid evaluation. Report as an inconclusive ablation.")
    elif delta > 0:
        print("\n  => Attention subtraction helps. Confirm at a third seed before reporting;")
        print("     it is the only variant that has ever cleared noise on this project.")
    else:
        print("\n  => Attention subtraction hurts. The subtracted branch is removing signal,")
        print("     not just globally-consistent anatomy.")


def report_optional(runs):
    """Arms G and L, printed only when present."""
    baseline = runs.get(("node21_talebi_zeroinit", "centernet_det", 1024, 224, 42, 0, "v3", 0.1))

    aug_v4 = runs.get(("node21_talebi_zeroinit", "centernet_det", 1024, 224, 42, 0, "v4", 0.1))
    if aug_v4 and baseline:
        print("\n" + "=" * 74)
        print("ARM G -- aug profile v4, no pos-frac (disambiguates the 2026-08-18 drop)")
        print("=" * 74)
        print(f"  v3 baseline  sens {sensitivity(baseline):.3f}   FROC {baseline['iou0.25']['froc_auc']:.3f}")
        print(f"  v4 aug only  sens {sensitivity(aug_v4):.3f}   FROC {aug_v4['iou0.25']['froc_auc']:.3f}")
        print(f"  delta        {sensitivity(aug_v4) - sensitivity(baseline):+.3f} sens, "
              f"{aug_v4['iou0.25']['froc_auc'] - baseline['iou0.25']['froc_auc']:+.3f} FROC")
        print("  => attributes the earlier drop to augmentation vs to positive oversampling.")

    lr_runs = [(f, runs.get(("node21_talebi_zeroinit", "centernet_det", 1024, 224, 42, 0, "v3", f)))
               for f in (0.01, 0.03)]
    if any(r for _, r in lr_runs) and baseline:
        print("\n" + "=" * 74)
        print("ARM L -- backbone learning-rate factor")
        print("=" * 74)
        print(f"  0.1 (default)  sens {sensitivity(baseline):.3f}")

        for factor, box in lr_runs:
            if box: print(f"  {factor:<14} sens {sensitivity(box):.3f}   ({sensitivity(box) - sensitivity(baseline):+.3f})")


def main():
    runs = load_runs()
    print(f"\nloaded {len(runs)} completed runs from run group '{RUN_GROUP}'")

    if not runs:
        print("No runs found. Has the sweep been run?")
        return

    report_arm_v(runs)
    report_arm_i(runs)
    report_arm_t(runs)
    report_optional(runs)
    print()


if __name__ == "__main__":
    main()
