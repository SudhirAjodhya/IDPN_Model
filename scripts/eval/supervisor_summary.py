"""
One consolidated results table per run group, for reporting rather than for analysis.

Everything here is already in metrics.json; the point is that it is scattered across four
blocks and several runs, so it never gets read as a whole. Three of these numbers were
explicitly requested at the 2026-08-25 supervisor review and have been sitting in the
stored metrics unreported since: image-level specificity, image-level ROC AUC, and
gradient enrichment.

Reads ONLY stored metrics - no model loading, no GPU, seconds to run. It does not recompute
anything, so it cannot disagree with the paired tests; where a number needs recomputing
(matched-FP sensitivity re-read from the curve) that stays in the dedicated scripts.

Reported at the MATCHED false-positive budget, never at a fixed score threshold.

Run:
  python scripts/eval/supervisor_summary.py --run-group v15_selection_and_scale
  python scripts/eval/supervisor_summary.py --run-group v16_fair_comparators --markdown
"""
import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

BUDGET = 0.25

SHORT = {
    "bilinear":                        "bilinear",
    "bilinear_antialias":              "bilinear+AA",
    "bicubic":                         "bicubic",
    "bicubic_antialias":               "bicubic+AA",
    "node21_talebi_zeroinit":          "Talebi 0-init",
    "node21_talebi_zeroinit_aa":       "Talebi 0-init+AA",
    "node21_talebi_zeroinit_attn":     "Talebi 0-init+attn",
    "node21_talebi_zeroinit_attn_aa":  "Talebi 0-init+attn+AA",
    "node21_talebi_zeroinit_wide":     "Talebi 0-init wide",
}


def sensitivity_at_budget(curve, budget=BUDGET):
    """Read the FROC curve at the budget. Linear between the two bracketing points."""
    points = sorted(((p["fp_per_image"], p["sensitivity"]) for p in curve))

    if not points: return None

    below = [p for p in points if p[0] <= budget]
    above = [p for p in points if p[0] > budget]

    if not below: return 0.0
    if not above: return below[-1][1]

    (fp_low, se_low), (fp_high, se_high) = below[-1], above[0]

    if fp_high == fp_low: return se_high

    weight = (budget - fp_low) / (fp_high - fp_low)

    return se_low + weight * (se_high - se_low)


def spread(values):
    """mean +/- half-range. Half-range, not SD: with 3 seeds an SD is barely meaningful."""
    if not values: return None, None
    if len(values) == 1: return values[0], None

    return statistics.mean(values), (max(values) - min(values)) / 2


def cell(values, decimals=3):
    mean, half = spread(values)

    if mean is None: return "-"
    if half is None: return "{:.{d}f}".format(mean, d=decimals)

    return "{:.{d}f}+/-{:.{d}f}".format(mean, half, d=decimals)


def load(run_group):
    """Every run in the group, keyed (target, variant) -> list of per-run dicts."""
    grouped = defaultdict(list)

    for path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue

        # Comma-separated, because the arms of one comparison can sit in different run
        # groups: bilinear and node21_talebi_zeroinit are in v15_selection_and_scale while
        # node21_talebi_zeroinit_attn is in v16_attn_ablation, yet they share a recipe and a
        # selection rule. A looser substring like "v1" would pull in v11-v14 as well.
        wanted = [g.strip() for g in run_group.split(",") if g.strip()]

        if not any(group in str(run.get("run_group", "")) for group in wanted): continue

        box = (run.get("box_metrics") or {}).get("iou0.25")

        if not box or not box.get("froc"): continue

        image_level = box.get("image_level") or {}
        attribution = run.get("attribution") or {}
        by_size = box.get("by_size") or {}

        grouped[(run["target_size"], run["variant"])].append({
            "seed":        run.get("seed"),
            "matched":     sensitivity_at_budget(box["froc"]),
            "froc_auc":    box.get("froc_auc"),
            "ap":          box.get("average_precision"),
            "img_sens":    image_level.get("sensitivity"),
            "img_spec":    image_level.get("specificity"),
            "img_auc":     image_level.get("roc_auc"),
            "resid_enr":   attribution.get("residual_enrichment"),
            "grad_enr":    attribution.get("gradient_enrichment"),
            "size_S":      (by_size.get("S") or {}).get("sensitivity"),
            "size_M":      (by_size.get("M") or {}).get("sensitivity"),
            "size_L":      (by_size.get("L") or {}).get("sensitivity"),
            "epochs":      run.get("epochs_trained"),
            "select_on":   run.get("select_on"),
            "num_workers": run.get("num_workers"),
        })

    return grouped


def collect(runs, field):
    return [r[field] for r in runs if r.get(field) is not None]


def table(rows, headers, markdown):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]

    def line(values, pad="|"):
        cells = [str(v).ljust(w) for v, w in zip(values, widths)]

        return ("| " + " | ".join(cells) + " |") if markdown else ("  " + "  ".join(cells))

    out = [line(headers)]

    if markdown:
        out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")

    out.extend(line(r) for r in rows)

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-group", required=True,
                        help="substring on run_group, or several comma-separated")
    parser.add_argument("--markdown", action="store_true", help="pipe-delimited for pasting")
    arguments = parser.parse_args()

    grouped = load(arguments.run_group)

    if not grouped:
        print("no runs matched run_group containing '{}'".format(arguments.run_group))
        return

    rules = {r["select_on"] for runs in grouped.values() for r in runs}
    workers = {r["num_workers"] for runs in grouped.values() for r in runs}

    print("\nRUN GROUP: {}".format(arguments.run_group))
    print("selection rule: {}   num_workers: {}".format(
        ", ".join(map(str, sorted(map(str, rules)))),
        ", ".join(map(str, sorted(map(str, workers))))))

    if len(rules) > 1:
        print("  WARNING: mixed selection rules in one group - these rows are NOT comparable")

    # None means the run predates 2026-09-02, when num_workers started being recorded - that
    # is unknown, not different, and must not read as the same alarm as two known values.
    recorded = {w for w in workers if w is not None}

    if len(recorded) > 1:
        print("  WARNING: mixed num_workers {} - the augmentation stream differs between "
              "rows, so they are not comparable".format(sorted(recorded)))

    elif None in workers and recorded:
        print("  note: some runs predate num_workers being recorded; assumed to match the "
              "recorded value {}".format(sorted(recorded)[0]))

    print("\nAll values mean+/-half-range across seeds. Detection read at {} FP/img.".format(BUDGET))

    # ---- Detection, at the matched budget ----
    print("\n1. DETECTION (nodule level, matched {} FP/img)".format(BUDGET))
    rows = []

    for (target, variant) in sorted(grouped, key=lambda k: (k[0], k[1])):
        runs = grouped[(target, variant)]
        rows.append([target, SHORT.get(variant, variant), len(runs),
                     cell(collect(runs, "matched")),
                     cell(collect(runs, "froc_auc")),
                     cell(collect(runs, "ap"))])

    print(table(rows, ["target", "variant", "n", "sens@0.25FP", "FROC AUC", "AP"],
                arguments.markdown))

    # ---- Image level: the 2026-08-25 request ----
    print("\n2. IMAGE LEVEL (is this film abnormal? 143 positive / 374 negative)")
    rows = []

    for (target, variant) in sorted(grouped, key=lambda k: (k[0], k[1])):
        runs = grouped[(target, variant)]
        rows.append([target, SHORT.get(variant, variant), len(runs),
                     cell(collect(runs, "img_sens")),
                     cell(collect(runs, "img_spec")),
                     cell(collect(runs, "img_auc"))])

    print(table(rows, ["target", "variant", "n", "sensitivity", "specificity", "ROC AUC"],
                arguments.markdown))

    # ---- Mechanism ----
    print("\n3. MECHANISM (>1.0 means the resizer treats nodules differently to matched controls)")
    rows = []

    for (target, variant) in sorted(grouped, key=lambda k: (k[0], k[1])):
        runs = grouped[(target, variant)]
        resid = collect(runs, "resid_enr")
        grad = collect(runs, "grad_enr")

        if not resid and not grad: continue

        rows.append([target, SHORT.get(variant, variant), len(runs),
                     cell(resid), cell(grad, decimals=1)])

    if rows:
        print(table(rows, ["target", "variant", "n", "residual enr", "gradient enr"],
                    arguments.markdown))

    # ---- Size, with the caveat attached ----
    print("\n4. BY NODULE SIZE  (S n=10, M n=81, L n=56 - S CANNOT support any claim,")
    print("   sensitivity there quantises in steps of 0.1)")
    rows = []

    for (target, variant) in sorted(grouped, key=lambda k: (k[0], k[1])):
        runs = grouped[(target, variant)]
        rows.append([target, SHORT.get(variant, variant), len(runs),
                     cell(collect(runs, "size_S"), 2),
                     cell(collect(runs, "size_M"), 2),
                     cell(collect(runs, "size_L"), 2)])

    print(table(rows, ["target", "variant", "n", "S (n=10)", "M (n=81)", "L (n=56)"],
                arguments.markdown))

    print("\nNOTE Precision and F1 are omitted deliberately. At a matched FP budget the false")
    print("     positive count is pinned (0.25 x 517 = 129), so precision is a deterministic")
    print("     function of recall and carries no independent information.")


if __name__ == "__main__":
    main()
