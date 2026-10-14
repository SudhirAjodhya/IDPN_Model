"""
FROC overlay: bilinear against a learned resizer, with both operating points marked.

The figure this makes is the answer to "the variant's sensitivity is higher, so why is it
not winning". Two curves that nearly coincide, and two markers sitting at very different
places along them. Sensitivity and false positives are one curve, not two numbers, and a
resizer that drifts liberal climbs a fixed-threshold readout for free.

Marked on each curve:
  o   where score_threshold=0.3 actually lands, which is what `sensitivity` in metrics.json
      and the val curve during training both report
  |   the 0.25 FP/img budget NODE21 itself reports at, read by interpolation

Curves are averaged across seeds on a shared false-positive grid.

Pure reader: metrics.json only, no checkpoints and no GPU, so it is safe to run while a
backfill is in progress.

Usage:
    python scripts/eval/plot_froc_overlay.py
    python scripts/eval/plot_froc_overlay.py --target 384 --variants bilinear node21_ldm_gated
    python scripts/eval/plot_froc_overlay.py --demo
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

MATCHED_FP_BUDGET = 0.25
MAX_FP = 4.0
GRID_POINTS = 400


def load_curves(head, iou_key, variant, input_size, target_size, include_letterbox=False):
    """Every seed's FROC curve and fixed-threshold operating point for one config."""
    found = []

    for metrics_path in glob.glob("outputs/**/metrics.json", recursive=True):
        try:
            run = json.load(open(metrics_path))
        except (json.JSONDecodeError, OSError):
            continue

        if run.get("head") != head: continue
        if run.get("variant") != variant: continue
        if run.get("input_size") != input_size: continue
        if run.get("target_size") != target_size: continue
        if run.get("letterbox") and not include_letterbox: continue

        box_metrics = (run.get("box_metrics") or {}).get(iou_key)

        if not box_metrics or not box_metrics.get("froc"): continue
        if run["seed"] in [f["seed"] for f in found]: continue

        found.append({
            "seed":            run["seed"],
            "curve":           box_metrics["froc"],
            "fixed_sens":      box_metrics["sensitivity"],
            "fixed_fp":        box_metrics["fp_per_image"],
            "score_threshold": box_metrics["score_threshold"],
        })

    return sorted(found, key=lambda f: f["seed"])


def mean_curve(records, grid):
    """Interpolate every seed onto a shared FP grid, then average."""
    interpolated = []

    for record in records:
        fp = np.array([p["fp_per_image"] for p in record["curve"]])
        se = np.array([p["sensitivity"] for p in record["curve"]])
        order = np.argsort(fp)

        # Clamp beyond the measured range rather than extrapolating off the end
        interpolated.append(np.interp(grid, fp[order], se[order],
                                      left=0.0, right=se[order][-1]))

    return np.mean(interpolated, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants",   type=str, nargs="+",
                        default=["bilinear", "node21_ldm_warp"])
    parser.add_argument("--head",       type=str, default="fcos")
    parser.add_argument("--iou",        type=str, default="0.25", choices=["0.25", "0.5"])
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--target",     type=int, default=224)
    parser.add_argument("--max-fp",     type=float, default=MAX_FP)
    parser.add_argument("--out",        type=Path, default=None)
    arguments = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0, arguments.max_fp, GRID_POINTS)
    figure, axes = plt.subplots(figsize=(8, 5.5))
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    summary_lines = []
    plotted = 0

    for index, variant in enumerate(arguments.variants):
        records = load_curves(arguments.head, f"iou{arguments.iou}", variant,
                              arguments.input_size, arguments.target)

        if not records:
            print(f"  no runs for {variant} at {arguments.input_size}->{arguments.target}")
            continue

        colour = colours[index % len(colours)]
        curve = mean_curve(records, grid)
        axes.plot(grid, curve, color=colour, lw=2,
                  label=f"{variant}  (n={len(records)} seed{'s' if len(records) > 1 else ''})")

        # Where the fixed threshold lands, averaged the same way across seeds
        fixed_fp   = float(np.mean([r["fixed_fp"] for r in records]))
        fixed_sens = float(np.mean([r["fixed_sens"] for r in records]))
        axes.plot(fixed_fp, fixed_sens, "o", color=colour, ms=9, mec="black", mew=1.0, zorder=5)

        matched = float(np.interp(MATCHED_FP_BUDGET, grid, curve))
        axes.plot(MATCHED_FP_BUDGET, matched, "s", color=colour, ms=7,
                  mec="black", mew=1.0, zorder=5)

        threshold = records[0]["score_threshold"]
        summary_lines.append(
            f"  {variant:<22} @thr {threshold}: sens {fixed_sens:.3f} at {fixed_fp:.3f} FP/img"
            f"   |   @{MATCHED_FP_BUDGET} FP: sens {matched:.3f}")
        plotted += 1

    if not plotted:
        print("nothing to plot - check --variants, --target and that you are in Model/")
        return

    axes.axvline(MATCHED_FP_BUDGET, color="black", ls="--", lw=1.0, alpha=0.6)
    axes.annotate(f"NODE21 budget\n{MATCHED_FP_BUDGET} FP/img",
                  xy=(MATCHED_FP_BUDGET, 0.03), xytext=(MATCHED_FP_BUDGET + 0.12, 0.03),
                  fontsize=8, va="bottom", color="black")

    axes.set(xlabel="False positives per image", ylabel="Sensitivity",
             xlim=(0, arguments.max_fp), ylim=(0, 1.0),
             title=f"FROC, {arguments.input_size} -> {arguments.target}, "
                   f"{arguments.head}, IoU {arguments.iou}")
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right", fontsize=9)

    # Spell out what the two marker shapes mean, since that is the whole point
    axes.plot([], [], "o", color="grey", mec="black", label="_nolegend_")
    axes.text(0.02, 0.97,
              "circle = score_threshold 0.3 (what metrics.json and the val curve report)\n"
              "square = read at a matched 0.25 FP/img budget",
              transform=axes.transAxes, fontsize=8, va="top",
              bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))

    figure.tight_layout()

    out_path = arguments.out or Path(
        f"outputs/froc_overlay_{arguments.input_size}_{arguments.target}_{arguments.head}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=160)

    print(f"\nFROC at {arguments.input_size}->{arguments.target}, IoU {arguments.iou}")
    for line in summary_lines:
        print(line)

    print(f"\nThe gap between each pair of readings is operating-point drift, not detection.")
    print(f"wrote {out_path}")


def _demo():
    """Assert-based self-check on the seed-averaging helper; no files, no matplotlib."""
    grid = np.linspace(0, 4.0, 5)                       # 0, 1, 2, 3, 4

    # One seed, a straight line from (0, 0) to (2, 1) then flat
    records = [{"curve": [{"fp_per_image": 0.0, "sensitivity": 0.0},
                          {"fp_per_image": 2.0, "sensitivity": 1.0}]}]
    curve = mean_curve(records, grid)
    assert abs(curve[1] - 0.5) < 1e-9, curve            # interpolated at fp=1
    assert abs(curve[3] - 1.0) < 1e-9, curve            # clamped past the end, not extrapolated

    # Two seeds average pointwise
    records.append({"curve": [{"fp_per_image": 0.0, "sensitivity": 0.0},
                              {"fp_per_image": 2.0, "sensitivity": 0.5}]})
    curve = mean_curve(records, grid)
    assert abs(curve[1] - 0.375) < 1e-9, curve          # mean of 0.5 and 0.25

    # Points arriving out of order must be sorted before interpolation
    shuffled = [{"curve": [{"fp_per_image": 2.0, "sensitivity": 1.0},
                           {"fp_per_image": 0.0, "sensitivity": 0.0}]}]
    assert abs(mean_curve(shuffled, grid)[1] - 0.5) < 1e-9

    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        main()
