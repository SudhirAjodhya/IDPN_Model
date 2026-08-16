"""
Table 4 aggregate plots for the Talebi reproduction.

Two figures per backbone (mirrors paper Figure 5-ish view + a per-config bar):

  1. flops_vs_top1_err_{backbone}.png
       X = GMACs (log scale), Y = top-1 error (%). Two series: Bilinear and
       Proposed. One marker per (config, seed); a solid line connects the
       per-config means. Reproduces the paper's "learned resizer sits below
       the bilinear curve at matched or lower compute" story.

  2. config_bars_{backbone}.png
       X = input resolution (224/256/320/368/448). Grouped bars for
       Bilinear (native res) vs Proposed (→ 224). Height = top-1 acc mean,
       error bars = seed range. Δ annotated above each pair.

Both figures read the same list of per-run result dicts (see
src/utils/results_aggregation_talebi_table4.py). Nothing here writes JSON -
it consumes what the aggregator produces.
"""
from __future__ import annotations
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _group_by_config(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        grouped[r["config"]["label"]].append(r)
    return grouped


def _sorted_by_res(labels: list[str], grouped: dict[str, list[dict]]):
    """Sort config labels by input resolution ascending."""
    def input_res(label):
        return grouped[label][0]["config"]["input_res"]
    return sorted(labels, key=input_res)


def save_flops_vs_error(runs: list[dict], output_dir: Path, backbone: str,
                         metric: str = "final_val_top1") -> None:
    """
    FLOPS-vs-error scatter with mean-of-seeds lines per resizer family.
    `metric` is which accuracy we plot as error (100 - metric).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group_by_config(runs)
    bil = {lab: rs for lab, rs in grouped.items() if lab.startswith("bilinear_")}
    prop = {lab: rs for lab, rs in grouped.items() if lab.startswith("proposed_")}

    def series(cfg_map):
        pts = []
        for lab in _sorted_by_res(list(cfg_map), cfg_map):
            rs = cfg_map[lab]
            gmacs = rs[0]["macs"] / 1e9
            errs = [100.0 - r[metric] for r in rs]
            pts.append((gmacs, errs, lab))
        return pts

    bil_pts  = series(bil)
    prop_pts = series(prop)

    fig, ax = plt.subplots(figsize=(9, 6))

    for pts, color, marker, label in [
        (bil_pts,  "#4A90D9", "o", "Bilinear (native res)"),
        (prop_pts, "#D94A4A", "s", "Proposed (resizer -> 224)"),
    ]:
        if not pts:
            continue
        # per-seed scatter
        for gmacs, errs, lab in pts:
            ax.scatter([gmacs] * len(errs), errs, color=color, marker=marker,
                       alpha=0.4, s=40)
        # mean-of-seeds line
        xs = [p[0] for p in pts]
        ys = [statistics.mean(p[1]) for p in pts]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2, label=label)
        # annotate input res beside each mean point
        for (gmacs, _, lab), y in zip(pts, ys):
            input_res = int(lab.split("_")[1])
            ax.annotate(f"{input_res}", xy=(gmacs, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Compute (GMACs, log scale)")
    ax.set_ylabel(f"Top-1 Error (%)  [from {metric}]")
    ax.set_title(f"{backbone} - Compute vs Top-1 Error  "
                  f"(each dot = one seed; line = per-config mean)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"flops_vs_top1_err_{backbone}_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_config_bars(runs: list[dict], output_dir: Path, backbone: str,
                      metric: str = "final_val_top1") -> None:
    """
    Grouped bar: Bilinear vs Proposed at each input resolution, seeds averaged
    with error bars. Δ annotated above each proposed bar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group_by_config(runs)
    # match bilinear_XXX and proposed_XXX by input resolution
    by_res: dict[int, dict[str, list[float]]] = defaultdict(lambda: {})
    for lab, rs in grouped.items():
        kind, res = lab.split("_")
        by_res[int(res)][kind] = [r[metric] for r in rs]

    resolutions = sorted(by_res.keys())
    if not resolutions:
        return

    x = np.arange(len(resolutions))
    width = 0.35

    bil_means = [statistics.mean(by_res[r].get("bilinear", [0])) for r in resolutions]
    bil_ranges = [(max(by_res[r].get("bilinear", [0])) - min(by_res[r].get("bilinear", [0]))) / 2
                  for r in resolutions]
    prop_means = [statistics.mean(by_res[r].get("proposed", [0])) for r in resolutions]
    prop_ranges = [(max(by_res[r].get("proposed", [0])) - min(by_res[r].get("proposed", [0]))) / 2
                   for r in resolutions]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_a = ax.bar(x - width / 2, bil_means, width, yerr=bil_ranges,
                    label="Bilinear (native res)", color="#4A90D9",
                    alpha=0.85, capsize=4)
    bars_b = ax.bar(x + width / 2, prop_means, width, yerr=prop_ranges,
                    label="Proposed (-> 224)", color="#D94A4A",
                    alpha=0.85, capsize=4)

    for res, bar_b, bm, pm in zip(resolutions, bars_b, bil_means, prop_means):
        delta = pm - bm
        ax.annotate(f"Δ {delta:+.2f}",
                    xy=(bar_b.get_x() + bar_b.get_width() / 2, max(bm, pm)),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color="green" if delta > 0 else "red")

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel(f"Val Top-1 Accuracy (%)  [{metric}]")
    ax.set_title(f"{backbone} - Bilinear vs Proposed at Each Input Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = output_dir / f"config_bars_{backbone}_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all_table4_plots(runs: list[dict], output_dir: Path,
                            backbone: str) -> None:
    """Emit both flops-vs-error and per-config bars, for both final and best-val."""
    for metric in ("final_val_top1", "best_val_top1"):
        save_flops_vs_error(runs, output_dir, backbone, metric=metric)
        save_config_bars(runs, output_dir, backbone, metric=metric)
