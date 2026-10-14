"""
Read the v8 sweep, reporting the mechanism readout first.

Detection sensitivity here moves by ~0.02-0.08 against a seed spread of 0.020 (fcos), so
most differences will be marginal. Residual enrichment is the more sensitive instrument
and is the quantity the v8 architectures were designed to change - so it leads.

Run: python scripts/eval/analyse_v8.py
"""

import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUN_GROUP = "v8_adaptive_resizers"
BINOMIAL_SE = 0.041

# Talebi's measured baseline, from the 2026-08-21 attribution run
BASELINE_ENRICHMENT = 0.78

RECEPTIVE_FIELD = {
    "bilinear": None,
    "node21_talebi_zeroinit": 11,
    "node21_ldm_gated": 11,
    "node21_ldm_dilated": 63,
    "node21_ldm_dilated_gated": 63,
    "node21_ldm_warp": None,          # warps the grid; RF is not the relevant axis
}

GATED = {"node21_ldm_gated", "node21_ldm_dilated_gated"}


def load_runs():
    runs = {}

    for path in glob.glob(f"outputs/**/*_{RUN_GROUP}/*/metrics.json", recursive=True):
        d = json.load(open(path))

        if not d.get("box_metrics"): continue

        key = (d.get("variant"), d.get("seed"), bool(d.get("letterbox", False)))
        runs[key] = d

    return runs


def cell(runs, variant, seeds, letterbox=False):
    """Collect (sensitivity, enrichment) across seeds for one variant."""
    sensitivities, enrichments = [], []

    for seed in seeds:
        d = runs.get((variant, seed, letterbox))

        if d is None: continue

        sensitivities.append(d["box_metrics"]["iou0.25"]["sensitivity"])
        attribution = d.get("attribution")

        if attribution and attribution.get("residual_enrichment") is not None:
            enrichments.append(attribution["residual_enrichment"])

    return sensitivities, enrichments


def line(variant, sensitivities, enrichments, label=None):
    name = label or variant
    rf = RECEPTIVE_FIELD.get(variant)
    rf_text = f"{rf:>4}" if rf else "   -"
    gate = "yes" if variant in GATED else " no"

    sens = f"{st.mean(sensitivities):.3f}" if sensitivities else "  -  "
    spread = f"{max(sensitivities) - min(sensitivities):.3f}" if len(sensitivities) > 1 else "  -  "
    enrich = f"{st.mean(enrichments):.3f}" if enrichments else "  -  "

    return f"  {name:<28}{rf_text}{gate:>6}{sens:>9}{spread:>9}{enrich:>12}"


def report_arm_m(runs, seeds):
    print("=" * 82)
    print("ARM M -- receptive field x selectivity")
    print("  Talebi's residual enrichment was 0.62-0.85: it puts LESS energy on nodules")
    print("  than on matched background. 1.0 = no preference. Higher is the repair.")
    print("=" * 82)
    print(f"\n  {'variant':<28}{'RF':>4}{'gate':>6}{'sens':>9}{'spread':>9}{'ENRICHMENT':>12}")

    results = {}

    for variant in ("node21_talebi_zeroinit", "node21_ldm_dilated",
                    "node21_ldm_gated", "node21_ldm_dilated_gated"):
        sensitivities, enrichments = cell(runs, variant, seeds)

        if not sensitivities: continue

        results[variant] = (sensitivities, enrichments)
        print(line(variant, sensitivities, enrichments))

    if len(results) < 4:
        print(f"\n  {len(results)}/4 cells present - arm incomplete")
        return

    def mean_enrichment(v):
        return st.mean(results[v][1]) if results[v][1] else None

    control  = mean_enrichment("node21_talebi_zeroinit")
    dilated  = mean_enrichment("node21_ldm_dilated")
    gated    = mean_enrichment("node21_ldm_gated")
    both     = mean_enrichment("node21_ldm_dilated_gated")

    if None in (control, dilated, gated, both):
        print("\n  attribution missing on some cells - was --attribution passed?")
        return

    print(f"\n  enrichment change against the zeroinit control ({control:.3f}):")
    print(f"    dilation alone   {dilated - control:+.3f}")
    print(f"    gating alone     {gated - control:+.3f}")
    print(f"    both             {both - control:+.3f}")

    interaction = (both - control) - (dilated - control) - (gated - control)
    print(f"    interaction      {interaction:+.3f}")

    print()
    if both > 1.0:
        print("  => Both together push enrichment ABOVE 1.0: the residual now concentrates")
        print("     on nodules. The measured deficit is repaired. Check whether detection followed.")
    elif interaction > 0.03 and both > max(dilated, gated):
        print("  => Superadditive: a gate is only useful given the context to gate on. This is")
        print("     exactly what the receptive-field explanation predicts.")
    elif dilated > gated + 0.03:
        print("  => Receptive field was the binding constraint; selectivity alone does little.")
    elif gated > dilated + 0.03:
        print("  => Selectivity was the binding constraint; a wider view alone does little.")
    else:
        print("  => Neither intervention moves enrichment. The deficit is not receptive field")
        print("     or gating - look at the resampling family instead (Arm W).")


def report_arm_w(runs, seeds):
    print("\n" + "=" * 82)
    print("ARM W -- non-uniform resampling: learn WHERE to sample, not what to add")
    print("=" * 82)
    print(f"\n  {'variant':<28}{'RF':>4}{'gate':>6}{'sens':>9}{'spread':>9}{'ENRICHMENT':>12}")

    results = {}

    for variant in ("bilinear", "node21_talebi_zeroinit", "node21_ldm_warp"):
        sensitivities, enrichments = cell(runs, variant, seeds)

        if not sensitivities: continue

        results[variant] = (sensitivities, enrichments)
        print(line(variant, sensitivities, enrichments))

    if "node21_ldm_warp" not in results or "bilinear" not in results:
        print("\n  arm incomplete")
        return

    warp_sens = st.mean(results["node21_ldm_warp"][0])
    bilinear_sens = st.mean(results["bilinear"][0])
    delta = warp_sens - bilinear_sens

    print(f"\n  warp - bilinear: {delta:+.3f} sensitivity "
          f"(binomial SE +/-{BINOMIAL_SE:.3f} per cell)")

    if delta > 2 * BINOMIAL_SE:
        print("\n  => The warping resizer beats bilinear. This is the first learned downsampler")
        print("     in this project to do so - confirm at a third seed before reporting.")
    elif abs(delta) <= 2 * BINOMIAL_SE:
        print("\n  => Inside noise. Non-uniform resampling does not beat bilinear here either,")
        print("     which would generalise the negative result beyond the Talebi family.")
    else:
        print("\n  => The warping resizer is worse than bilinear.")


def report_arm_l(runs):
    print("\n" + "=" * 82)
    print("ARM L -- aspect ratio: squash vs letterbox (39% of films are non-square)")
    print("=" * 82)

    rows = []

    for variant in ("bilinear", "node21_talebi_zeroinit"):
        squash, _ = cell(runs, variant, [42], letterbox=False)
        letterbox, _ = cell(runs, variant, [42], letterbox=True)

        if squash and letterbox:
            rows.append((variant, squash[0], letterbox[0]))

    if len(rows) < 2:
        print(f"\n  {len(rows)}/2 variants present - arm incomplete")
        return

    print(f"\n  {'variant':<28}{'squash':>10}{'letterbox':>12}{'delta':>9}")
    for variant, squash, letterbox in rows:
        print(f"  {variant:<28}{squash:>10.3f}{letterbox:>12.3f}{letterbox - squash:>+9.3f}")

    deltas = {v: lb - sq for v, sq, lb in rows}
    difference = deltas["node21_talebi_zeroinit"] - deltas["bilinear"]

    print(f"\n  differential (resizer gain - bilinear gain): {difference:+.3f}")

    if all(d > BINOMIAL_SE for d in deltas.values()):
        print("\n  => Letterboxing helps BOTH arms: a data-pipeline fix, not part of the resizer")
        print("     story. Adopt it, and rerun the load-bearing comparison under it.")
    elif abs(difference) < BINOMIAL_SE:
        print("\n  => No differential effect. Aspect distortion was not driving the comparison.")
    else:
        print("\n  => Letterboxing affects the two arms differently - it interacts with the")
        print("     resizer and cannot be treated as a neutral pipeline fix.")


def main():
    runs = load_runs()
    print(f"\nloaded {len(runs)} completed runs from '{RUN_GROUP}'")

    if not runs:
        print("No runs found. Has the sweep been run?")
        return

    seeds = sorted({seed for _, seed, _ in runs}, key=lambda s: (s is None, s))

    report_arm_m(runs, seeds)
    report_arm_w(runs, seeds)
    report_arm_l(runs)

    print(f"\nreference: Talebi zeroinit measured {BASELINE_ENRICHMENT:.2f} enrichment on 2026-08-21")
    print("enrichment 1.0 = treats nodules like any other same-size patch of chest\n")


if __name__ == "__main__":
    main()
