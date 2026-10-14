"""
Size-stratified sensitivity at a MATCHED false-positive budget, across variants.

WHY NOT THE STORED by_size BLOCK. `box_metrics.iou0.25.by_size` is computed at a FIXED
score threshold of 0.3, which is the wrong yardstick for the same reason it is wrong pooled:
the arms do not share a score scale, so at 0.3 one variant is simply making more guesses.
This reads `froc_by_size` instead - per-bin sensitivity swept over the same global threshold
grid, plotted against the SAME pooled false-positive axis - so all three bins are read at
the operating point NODE21 reports at.

False positives belong to no size bin (they matched no ground truth), so there is no such
thing as a per-bin FP rate. The budget is therefore global: one threshold is chosen so the
whole image set sits at 0.25 FP/img, and each bin is then read at that threshold. That
answers "at the operating point we report, how many small / medium / large nodules were
found", which is the only well-posed size question here.

POWER WARNING, applied automatically below. The test split holds 10 small, 81 medium and
56 large nodules. With n=10 sensitivity moves in steps of 0.1 and no difference smaller
than one nodule is measurable, so the S column cannot support a claim in either direction.
It is printed because its absence would be conspicuous, not because it is evidence.

Reads stored metrics only - no model loading, no GPU.

Run:
  python scripts/eval/size_stratified_comparison.py --run-groups v15_selection_and_scale,v16_attn_ablation
"""
import argparse
import glob
import json
import statistics
from collections import defaultdict

BUDGET = 0.25
BINS = ("S", "M", "L")

BIN_LABEL = {"S": "S <10mm", "M": "M 10-20mm", "L": "L >=20mm"}

SHORT = {
    "bilinear":                       "bilinear",
    "bilinear_antialias":             "bilinear+AA",
    "bicubic":                        "bicubic",
    "bicubic_antialias":              "bicubic+AA",
    "node21_talebi_zeroinit":         "Talebi 0-init",
    "node21_talebi_zeroinit_aa":      "Talebi 0-init+AA",
    "node21_talebi_zeroinit_attn":    "Talebi 0-init+attn",
    "node21_talebi_zeroinit_attn_aa": "Talebi 0-init+attn+AA",
}

# Print order: baselines first, then learned, so a table reads baseline -> variant
ORDER = ["bilinear", "bilinear_antialias", "bicubic", "bicubic_antialias",
         "node21_talebi_zeroinit", "node21_talebi_zeroinit_aa",
         "node21_talebi_zeroinit_attn", "node21_talebi_zeroinit_attn_aa"]


def read_at_budget(curve, budget=BUDGET):
    """Sensitivity where the shared pooled FP axis crosses the budget."""
    points = sorted((p["fp_per_image"], p["sensitivity"]) for p in curve)

    if not points: return None

    below = [p for p in points if p[0] <= budget]
    above = [p for p in points if p[0] > budget]

    if not below: return 0.0
    if not above: return below[-1][1]

    (fp_low, se_low), (fp_high, se_high) = below[-1], above[0]

    if fp_high == fp_low: return se_high

    return se_low + (budget - fp_low) / (fp_high - fp_low) * (se_high - se_low)


def load(run_groups):
    grouped = defaultdict(list)
    counts = {}

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        group = str(run.get("run_group", ""))

        if not any(wanted in group for wanted in run_groups): continue

        box = (run.get("box_metrics") or {}).get("iou0.25")

        if not box or not box.get("froc_by_size"): continue

        entry = {"seed": run.get("seed"), "select_on": run.get("select_on"), "group": group}

        for bin_name in BINS:
            entry[bin_name] = read_at_budget(box["froc_by_size"].get(bin_name) or [])

        entry["pooled"] = read_at_budget(box.get("froc") or [])
        grouped[(run["target_size"], run["variant"])].append(entry)

        for bin_name, stats in (box.get("by_size") or {}).items():
            if stats.get("n_nodules"): counts[bin_name] = stats["n_nodules"]

    return grouped, counts


def cell(values):
    clean = [v for v in values if v is not None]

    if not clean: return "-"
    if len(clean) == 1: return "{:.3f}".format(clean[0])

    half = (max(clean) - min(clean)) / 2

    return "{:.3f}+/-{:.3f}".format(statistics.mean(clean), half)


def mean_of(values):
    clean = [v for v in values if v is not None]

    return statistics.mean(clean) if clean else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-groups", required=True,
                        help="comma-separated substrings; runs from any of them are included")
    parser.add_argument("--baseline", default="bilinear",
                        help="variant every other row is differenced against")
    arguments = parser.parse_args()

    run_groups = [g.strip() for g in arguments.run_groups.split(",") if g.strip()]
    grouped, counts = load(run_groups)

    if not grouped:
        print("no runs matched {}".format(run_groups))
        return

    rules = {e["select_on"] for entries in grouped.values() for e in entries}

    print("\nSIZE-STRATIFIED SENSITIVITY at a matched {} FP/img".format(BUDGET))
    print("run groups: {}".format(", ".join(sorted({e["group"] for v in grouped.values() for e in v}))))
    print("selection rule: {}".format(", ".join(sorted(map(str, rules)))))

    if len(rules) > 1:
        print("  WARNING: mixed selection rules - these rows are NOT comparable")

    print("\nTest split holds {} small, {} medium, {} large nodules.".format(
        counts.get("S", "?"), counts.get("M", "?"), counts.get("L", "?")))
    print("Values are mean+/-half-range across seeds.")

    for target in sorted({t for t, _ in grouped}):
        present = [v for t, v in grouped if t == target]
        ordered = [v for v in ORDER if v in present] + sorted(set(present) - set(ORDER))

        print("\n" + "=" * 76)
        print("TARGET {}".format(target))
        print("=" * 76)
        print("  {:<24} {:>3}  {:>15} {:>15} {:>15} {:>9}".format(
            "variant", "n", BIN_LABEL["S"], BIN_LABEL["M"], BIN_LABEL["L"], "pooled"))

        baseline_means = None

        for variant in ordered:
            entries = grouped[(target, variant)]
            row = [cell([e[b] for e in entries]) for b in BINS]

            print("  {:<24} {:>3}  {:>15} {:>15} {:>15} {:>9}".format(
                SHORT.get(variant, variant), len(entries), row[0], row[1], row[2],
                cell([e["pooled"] for e in entries])))

            if variant == arguments.baseline:
                baseline_means = {b: mean_of([e[b] for e in entries]) for b in BINS}
                baseline_means["pooled"] = mean_of([e["pooled"] for e in entries])

        if not baseline_means: continue

        print("\n  delta vs {}:".format(SHORT.get(arguments.baseline, arguments.baseline)))

        for variant in ordered:
            if variant == arguments.baseline: continue

            entries = grouped[(target, variant)]
            deltas = []

            for key in list(BINS) + ["pooled"]:
                mean = mean_of([e[key] for e in entries])
                base = baseline_means.get(key)
                deltas.append("{:+.3f}".format(mean - base)
                              if mean is not None and base is not None else "-")

            print("  {:<24} {:>3}  {:>15} {:>15} {:>15} {:>9}".format(
                SHORT.get(variant, variant), len(entries), *deltas))

    small = counts.get("S")

    if small:
        print("\n" + "-" * 76)
        print("POWER: the S column has n={}, so sensitivity there moves in steps of {:.2f}".format(
            small, 1 / small))
        print("and NO difference smaller than one nodule is measurable. Do not read the S")
        print("deltas as evidence in either direction - report the limitation instead.")

    print("\nNOTE False positives match no ground truth, so they belong to no size bin.")
    print("     There is no per-bin FP rate; one global threshold puts the whole image set")
    print("     at {} FP/img and every bin is read there.".format(BUDGET))


if __name__ == "__main__":
    main()
