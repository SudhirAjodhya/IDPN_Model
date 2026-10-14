"""
Size-stratified (S/M/L) sensitivity across resolutions, read post-hoc from saved runs.

Pure reader. Touches no checkpoint and runs no forward pass, so it costs nothing to run
and cannot perturb any result.

READ THE CAVEATS THE SCRIPT PRINTS AT THE END BEFORE QUOTING ANY NUMBER FROM IT.

QUOTE THE MATCHED-FP TABLE, NEVER THE FIXED-THRESHOLD ONES. `by_size` is recorded only at
score_threshold=0.3, which lands wherever it lands on each run's own curve - measured
spread 0.55 to 1.15 FP/img across these runs. That is trap 4.1 in docs/context/HANDOVER.md,
and it is not theoretical here: at target 384 `node21_ldm_gated` reads +0.031 against
bilinear at the fixed threshold and -0.011 at 0.25 FP/img, the entire apparent gain being
its 1.149 FP/img operating point. `node21_ldm_warp` at target 224 goes +0.020 to -0.017 the
same way. The fixed tables are printed only to show how far that drift reaches.

The matched-FP table needs `froc_by_size`, written by
scripts/eval/backfill_froc_by_size.py. Watch the seed counts: a config with fewer seeds in
the matched table than the fixed one has runs that failed the backfill's regression check,
and comparing a 1-seed arm against a 3-seed baseline is its own confound.

One limit is structural and cannot be fixed here. The S bin holds ~10 of the 147 test
nodules, so binomial SE at n=10 is about +/-0.15, roughly 4x the +/-0.041 that already
limits the pooled number. Nothing the S column shows at one or two seeds is resolvable, and
a same-sign S delta across several variants is more likely one unlucky bilinear baseline
than a real small-nodule effect - they all share that baseline.

The M and L bins (n ~ 81 and ~ 56) are the only columns worth reading, and even those carry
SE of about +/-0.055 and +/-0.067.

Run: python scripts/eval/analyse_size_resolution.py
     python scripts/eval/analyse_size_resolution.py --head centernet_det --iou 0.5
     python scripts/eval/analyse_size_resolution.py --demo
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

BINS = ("S", "M", "L")
BIN_LABELS = {"S": "S (<10mm)", "M": "M (10-20mm)", "L": "L (>=20mm)"}
POOLED_BINOMIAL_SE = 0.041
MATCHED_FP_BUDGET = 0.25                  # the operating point NODE21 itself reports at


def binomial_se(proportion, n):
    """Standard error of a proportion. Returns None for an empty bin."""
    if not n: return None

    return math.sqrt(max(proportion * (1.0 - proportion), 0.0) / n)


def load_runs(head, iou_key, include_letterbox):
    """(variant, input_size, target_size) -> list of one record per seed."""
    runs = {}

    for metrics_path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(metrics_path))
        except (json.JSONDecodeError, OSError):
            continue

        # Only completed runs of the requested head, with box metrics present
        if run.get("head") != head: continue
        if not run.get("box_metrics"): continue
        if run.get("letterbox") and not include_letterbox: continue

        box_metrics = run["box_metrics"].get(iou_key)

        if not box_metrics or not box_metrics.get("by_size"): continue

        record = {
            "seed":            run["seed"],
            "by_size":         box_metrics["by_size"],
            "froc_by_size":    box_metrics.get("froc_by_size"),      # None until backfilled
            "froc":            box_metrics.get("froc"),
            "sensitivity":     box_metrics["sensitivity"],
            "fp_per_image":    box_metrics["fp_per_image"],
            "score_threshold": box_metrics["score_threshold"],
            "run_dir":         os.path.dirname(metrics_path),
        }

        key = (run["variant"], run["input_size"], run["target_size"])
        runs.setdefault(key, [])

        # A config can appear under more than one run group; keep one record per seed
        if record["seed"] not in [existing["seed"] for existing in runs[key]]:
            runs[key].append(record)

    return runs


def summarise(records):
    """Seed-averaged per-bin sensitivity, pooled sensitivity and operating point."""
    per_bin = {}

    for bin_name in BINS:
        values = [r["by_size"][bin_name]["sensitivity"] for r in records
                  if r["by_size"][bin_name]["sensitivity"] is not None]
        counts = [r["by_size"][bin_name]["n_nodules"] for r in records]

        per_bin[bin_name] = {
            "sensitivity": statistics.mean(values) if values else None,
            "n_nodules":   max(counts) if counts else 0,
        }

    return {
        "per_bin":      per_bin,
        "sensitivity":  statistics.mean(r["sensitivity"] for r in records),
        "fp_per_image": statistics.mean(r["fp_per_image"] for r in records),
        "seeds":        sorted(r["seed"] for r in records),
    }


def format_bin_cell(entry):
    if entry["sensitivity"] is None: return f"{'-':>16}"

    return f"{entry['sensitivity']:9.3f}(n={entry['n_nodules']:>3})"


def print_table(runs, variants, targets, input_size):
    """One block per target resolution, one row per variant, then deltas vs bilinear."""
    header = (f"  {'variant':<22}{'in->tgt':>11}{'seeds':>6}" +
              "".join(f"{BIN_LABELS[b]:>17}" for b in BINS) +
              f"{'pooled':>9}{'FP/img':>9}")

    for target in targets:
        present = [(v, (v, input_size, target)) for v in variants
                   if (v, input_size, target) in runs]

        if not present: continue

        print()
        print("-" * len(header))
        print(f"TARGET {target}")
        print("-" * len(header))
        print(header)

        for variant, key in present:
            summary = summarise(runs[key])
            print(f"  {variant:<22}{f'{input_size}->{target}':>11}{len(summary['seeds']):>6}" +
                  "".join(format_bin_cell(summary["per_bin"][b]) for b in BINS) +
                  f"{summary['sensitivity']:9.3f}{summary['fp_per_image']:9.3f}")

        # Paired deltas against bilinear, only where bilinear was actually run
        baseline_key = ("bilinear", input_size, target)

        if baseline_key not in runs: continue

        baseline = summarise(runs[baseline_key])

        print(f"\n  delta vs bilinear:")

        for variant, key in present:
            if variant == "bilinear": continue

            summary = summarise(runs[key])
            cells = ""

            for bin_name in BINS:
                variant_value  = summary["per_bin"][bin_name]["sensitivity"]
                baseline_value = baseline["per_bin"][bin_name]["sensitivity"]

                if variant_value is None or baseline_value is None:
                    cells += f"{'-':>17}"
                    continue

                delta = variant_value - baseline_value
                n  = summary["per_bin"][bin_name]["n_nodules"]
                se = binomial_se(baseline_value, n)

                # Flag only deltas that clear the SE of their own bin, not the pooled SE
                marker = "*" if se is not None and abs(delta) > se else " "
                cells += f"{delta:+11.3f}{marker:<6}"

            pooled_delta  = summary["sensitivity"] - baseline["sensitivity"]
            pooled_marker = "*" if abs(pooled_delta) > POOLED_BINOMIAL_SE else " "
            print(f"  {variant:<22}{'':>11}{'':>6}{cells}{pooled_delta:+8.3f}{pooled_marker}")


def sensitivity_at_budget(curve, budget):
    """Interpolate one FROC curve at a false-positive budget. Same convention as analyse_v9."""
    if not curve: return None

    fp = np.array([point["fp_per_image"] for point in curve])
    se = np.array([point["sensitivity"] for point in curve])
    order = np.argsort(fp)

    return float(np.interp(budget, fp[order], se[order], left=0.0, right=se[order][-1]))


def summarise_matched(records, budget):
    """Seed-averaged per-bin and pooled sensitivity, all read at the same FP budget."""
    usable = [r for r in records if r["froc_by_size"]]

    if not usable: return None

    per_bin = {}

    for bin_name in BINS:
        values = [sensitivity_at_budget(r["froc_by_size"].get(bin_name), budget) for r in usable]
        values = [v for v in values if v is not None]
        counts = [r["by_size"][bin_name]["n_nodules"] for r in usable]

        per_bin[bin_name] = {
            "sensitivity": statistics.mean(values) if values else None,
            "n_nodules":   max(counts) if counts else 0,
        }

    pooled = [sensitivity_at_budget(r["froc"], budget) for r in usable if r["froc"]]

    return {
        "per_bin":     per_bin,
        "sensitivity": statistics.mean(pooled) if pooled else None,
        "seeds":       sorted(r["seed"] for r in usable),
    }


def print_matched_fp_table(runs, variants, targets, input_size, budget):
    """
    The table that trap 4.1 requires: every bin read at one shared FP budget.

    Returns False when no run carries froc_by_size yet, so the caller can point at the
    backfill script rather than printing an empty section.
    """
    if not any(r["froc_by_size"] for records in runs.values() for r in records):
        return False

    print()
    print("=" * 96)
    print(f"MATCHED FALSE-POSITIVE BUDGET -- every bin read at {budget} FP/img")
    print("  This is the table to quote. The fixed-threshold tables above are kept only")
    print("  to show how far operating-point drift moved them.")
    print("=" * 96)

    header = (f"  {'variant':<22}{'in->tgt':>11}{'seeds':>6}" +
              "".join(f"{BIN_LABELS[b]:>17}" for b in BINS) + f"{'pooled':>9}")

    for target in targets:
        present = [(v, (v, input_size, target)) for v in variants
                   if (v, input_size, target) in runs
                   and summarise_matched(runs[(v, input_size, target)], budget)]

        if not present: continue

        print()
        print(f"TARGET {target}")
        print(header)

        for variant, key in present:
            summary = summarise_matched(runs[key], budget)
            pooled = summary["sensitivity"]
            print(f"  {variant:<22}{f'{input_size}->{target}':>11}{len(summary['seeds']):>6}" +
                  "".join(format_bin_cell(summary["per_bin"][b]) for b in BINS) +
                  (f"{pooled:9.3f}" if pooled is not None else f"{'-':>9}"))

        baseline_key = ("bilinear", input_size, target)
        baseline = summarise_matched(runs[baseline_key], budget) if baseline_key in runs else None

        if baseline is None: continue

        print(f"\n  delta vs bilinear:")

        for variant, key in present:
            if variant == "bilinear": continue

            summary = summarise_matched(runs[key], budget)
            cells = ""

            for bin_name in BINS:
                variant_value  = summary["per_bin"][bin_name]["sensitivity"]
                baseline_value = baseline["per_bin"][bin_name]["sensitivity"]

                if variant_value is None or baseline_value is None:
                    cells += f"{'-':>17}"
                    continue

                delta  = variant_value - baseline_value
                se     = binomial_se(baseline_value, summary["per_bin"][bin_name]["n_nodules"])
                marker = "*" if se is not None and abs(delta) > se else " "
                cells += f"{delta:+11.3f}{marker:<6}"

            if summary["sensitivity"] is None or baseline["sensitivity"] is None:
                print(f"  {variant:<22}{'':>11}{'':>6}{cells}")
                continue

            pooled_delta  = summary["sensitivity"] - baseline["sensitivity"]
            pooled_marker = "*" if abs(pooled_delta) > POOLED_BINOMIAL_SE else " "
            print(f"  {variant:<22}{'':>11}{'':>6}{cells}{pooled_delta:+8.3f}{pooled_marker}")

    return True


def print_resolution_ladder(runs, variants, targets, input_size, budget, matched):
    """
    Does each bin respond to target resolution the same way?

    Reads matched-FP curves when they exist and falls back to the fixed threshold when
    they do not. The distinction is load-bearing: under the fixed threshold this section
    appeared to show M responding to resolution while L stayed flat, and that difference
    turned out to be operating-point drift rather than a property of nodule size. Do not
    read the fallback version as evidence about size.
    """
    print()
    print("=" * 96)
    print("RESOLUTION RESPONSE PER BIN -- does small-nodule sensitivity climb faster than large?")

    if matched:
        print(f"  Read at {budget} FP/img, so rows are comparable across targets.")
    else:
        print("  WARNING: fixed threshold, so a bin can appear to respond to resolution when")
        print("  only its operating point moved. Run the backfill before reading this.")

    print("=" * 96)

    def summary_for(key):
        if matched: return summarise_matched(runs[key], budget)

        return summarise(runs[key])

    for variant in variants:
        ladder = [(t, summary_for((variant, input_size, t)))
                  for t in targets if (variant, input_size, t) in runs]
        ladder = [(t, s) for t, s in ladder if s is not None]

        if len(ladder) < 2: continue

        print(f"\n  {variant}  (input {input_size})")

        for bin_name in BINS:
            values = [(t, s["per_bin"][bin_name]["sensitivity"]) for t, s in ladder
                      if s["per_bin"][bin_name]["sensitivity"] is not None]

            if len(values) < 2:
                print(f"    {BIN_LABELS[bin_name]:<14} insufficient runs")
                continue

            n      = ladder[0][1]["per_bin"][bin_name]["n_nodules"]
            se     = binomial_se(0.5, n) or 0.0
            spread = max(v for _, v in values) - min(v for _, v in values)
            trail  = "  <- inside this bin's own noise" if spread < se else ""

            print(f"    {BIN_LABELS[bin_name]:<14}" +
                  "  ".join(f"t{t}:{v:.3f}" for t, v in values) +
                  f"   spread {spread:.3f}  (bin SE ~{se:.3f}){trail}")


def print_caveats(have_matched):
    print()
    print("=" * 96)
    print("CAVEATS")
    print("=" * 96)

    if have_matched:
        print("  1. The FIXED-THRESHOLD tables at the top sit at score_threshold=0.3, wherever that")
        print("     lands on each run's own curve (see their FP/img column). They are NOT")
        print("     comparable across variants. Quote the MATCHED-FP table instead.")
    else:
        print("  1. FIXED THRESHOLD, NOT MATCHED FP. by_size was computed at score_threshold=0.3.")
        print("     The FP/img column shows where each row actually sits. Rows at different FP/img")
        print("     are NOT comparable; a higher bin sensitivity may just be a looser operating")
        print("     point. This is trap 4.1. No run carries froc_by_size yet - run")
        print("     `python scripts/eval/backfill_froc_by_size.py` to populate it.")

    print("  2. The S bin holds ~10 of 147 test nodules -> SE ~ +/-0.150. Treat the S column as")
    print("     unreadable. M (n~81) SE ~ +/-0.055, L (n~56) SE ~ +/-0.067.")
    print("  3. '*' marks a delta exceeding its own bin's binomial SE. That is a floor on noise,")
    print("     not a significance test, and it ignores seed-to-seed variation.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head",       type=str, default="fcos")
    parser.add_argument("--iou",        type=str, default="0.25", choices=["0.25", "0.5"])
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--letterbox",  action="store_true",
                        help="Include letterbox runs, which are not comparable with squash runs.")
    arguments = parser.parse_args()

    runs = load_runs(arguments.head, f"iou{arguments.iou}", arguments.letterbox)

    if not runs:
        print(f"No runs found for head={arguments.head} iou={arguments.iou}. "
              f"Check you are running from the Model/ directory.")
        return

    # Order variants with the baseline first so deltas read left to right
    variants = sorted({key[0] for key in runs}, key=lambda v: (v != "bilinear", v))
    targets  = sorted({key[2] for key in runs if key[1] == arguments.input_size})

    print(f"\nloaded {sum(len(v) for v in runs.values())} runs across {len(runs)} configs "
          f"(head={arguments.head}, IoU={arguments.iou}, input={arguments.input_size})")

    print_table(runs, variants, targets, arguments.input_size)
    have_matched = print_matched_fp_table(runs, variants, targets, arguments.input_size,
                                          MATCHED_FP_BUDGET)
    print_resolution_ladder(runs, variants, targets, arguments.input_size,
                            MATCHED_FP_BUDGET, have_matched)
    print_caveats(have_matched)


def _demo():
    """Assert-based self-check on synthetic records; no filesystem or GPU needed."""
    records = [
        {"seed": 1, "sensitivity": 0.60, "fp_per_image": 0.5, "score_threshold": 0.3,
         "run_dir": "",
         "by_size": {"S": {"sensitivity": 0.2, "n_nodules": 10},
                     "M": {"sensitivity": 0.6, "n_nodules": 80},
                     "L": {"sensitivity": 0.8, "n_nodules": 57}}},
        {"seed": 2, "sensitivity": 0.70, "fp_per_image": 0.5, "score_threshold": 0.3,
         "run_dir": "",
         "by_size": {"S": {"sensitivity": 0.4, "n_nodules": 10},
                     "M": {"sensitivity": 0.8, "n_nodules": 80},
                     "L": {"sensitivity": 0.9, "n_nodules": 57}}},
    ]

    summary = summarise(records)
    assert abs(summary["per_bin"]["S"]["sensitivity"] - 0.3) < 1e-9, summary
    assert abs(summary["per_bin"]["M"]["sensitivity"] - 0.7) < 1e-9, summary
    assert abs(summary["sensitivity"] - 0.65) < 1e-9, summary
    assert summary["seeds"] == [1, 2], summary

    # An empty bin must not crash the summary or the SE helper
    empty = [{"seed": 1, "sensitivity": 0.5, "fp_per_image": 0.4, "score_threshold": 0.3,
              "run_dir": "",
              "by_size": {"S": {"sensitivity": None, "n_nodules": 0},
                          "M": {"sensitivity": 0.5, "n_nodules": 80},
                          "L": {"sensitivity": 0.5, "n_nodules": 57}}}]
    assert summarise(empty)["per_bin"]["S"]["sensitivity"] is None
    assert binomial_se(0.5, 0) is None
    assert abs(binomial_se(0.5, 100) - 0.05) < 1e-9

    # Reading a FROC curve at a budget interpolates between its points
    curve = [{"threshold": 0.9, "sensitivity": 0.2, "fp_per_image": 0.0},
             {"threshold": 0.5, "sensitivity": 0.6, "fp_per_image": 0.5},
             {"threshold": 0.1, "sensitivity": 0.9, "fp_per_image": 2.0}]
    assert abs(sensitivity_at_budget(curve, 0.25) - 0.4) < 1e-9, sensitivity_at_budget(curve, 0.25)
    assert sensitivity_at_budget([], 0.25) is None

    # Budgets beyond the curve clamp to its endpoints rather than extrapolating
    assert abs(sensitivity_at_budget(curve, 99.0) - 0.9) < 1e-9

    # A run without froc_by_size must be skipped, not crash the matched summary
    stale = [{"seed": 1, "by_size": records[0]["by_size"], "froc_by_size": None, "froc": None}]
    assert summarise_matched(stale, 0.25) is None

    fresh = [{"seed": 1, "by_size": records[0]["by_size"], "froc": curve,
              "froc_by_size": {"S": curve, "M": curve, "L": []}}]
    matched = summarise_matched(fresh, 0.25)
    assert abs(matched["per_bin"]["S"]["sensitivity"] - 0.4) < 1e-9, matched
    assert matched["per_bin"]["L"]["sensitivity"] is None, matched      # empty curve -> no reading
    assert abs(matched["sensitivity"] - 0.4) < 1e-9, matched

    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        main()
