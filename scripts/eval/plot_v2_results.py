"""Read outputs/results/aggregated_v2.json, produce PNG/PDF plots for the report.

Four figures:
  1. bar_sens_by_seed.png      — combined-test sens per variant, grouped by seed, w/ mean bar
  2. bar_size_stratified.png    — S/M/L bars per variant (mean across seeds)
  3. bar_val_to_test_drop.png   — val -> test drop per variant (mean +/- std)
  4. froc_curves.png            — FROC curves overlaid (mean across seeds), 0-4 FP range

Only plots the four ConvNeXt-pretrained variants: bilinear, node21_talebi (paper-faithful),
node21_talebi_zeroinit, node21_talebi_zeroinit_attn. Skips backbone-matrix runs (they're
n=1 per config, not comparable). Skips paper-faithful if we only have n=2 checkpoints.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
IN   = ROOT / "outputs" / "results" / "aggregated_v2.json"
OUT  = ROOT / "outputs" / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Ordering + display names + colors (colorblind-safe palette)
VARIANTS = [
    ("bilinear",                     "Bilinear (baseline)",   "#4C72B0"),
    ("node21_talebi",                "Talebi (paper-faithful)","#DD8452"),
    ("node21_talebi_zeroinit",       "Talebi (zero-init)",    "#55A868"),
    ("node21_talebi_zeroinit_attn",  "Zero-init + attention", "#C44E52"),
]


def load_runs():
    data = json.loads(IN.read_text())
    # Filter: convnext_fpn only (pretrained), canonical v2 target=384, variants we care about.
    # Excludes v3 resolution sweep runs (they have varying target sizes and would blend the mean).
    runs = [r for r in data["runs"]
            if r["backbone"] == "convnext_fpn"
            and r["variant"] in {v[0] for v in VARIANTS}
            and r["target_size"] == 384]
    grouped = defaultdict(list)
    for r in runs:
        grouped[r["variant"]].append(r)
    return grouped


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.{{png,pdf}}")


# ============================================================
# Figure 1 — bar chart of combined-test sens per variant, per seed, + mean
# ============================================================
def fig_sens_by_seed(grouped):
    fig, ax = plt.subplots(figsize=(10, 5))
    all_seeds = sorted({r["seed"] for runs in grouped.values() for r in runs})
    x = np.arange(len(all_seeds) + 1)                                              # +1 for "mean"
    n_var = len(VARIANTS)
    bar_w = 0.8 / n_var

    for i, (var, label, color) in enumerate(VARIANTS):
        vals = []
        for seed in all_seeds:
            match = [r["test"]["combined"]["sensitivity"] for r in grouped[var] if r["seed"] == seed]
            vals.append(match[0] if match else np.nan)
        seed_vals = [v for v in vals if not np.isnan(v)]
        vals.append(np.mean(seed_vals) if seed_vals else np.nan)                   # mean bar
        offsets = x - 0.4 + bar_w * (i + 0.5)
        bars = ax.bar(offsets, vals, bar_w, label=label, color=color, edgecolor="black", linewidth=0.5)
        # annotate mean bar
        if not np.isnan(vals[-1]):
            ax.text(offsets[-1], vals[-1] + 0.005, f"{vals[-1]:.3f}",
                    ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in all_seeds] + ["MEAN"])
    ax.set_ylabel("Combined-test sensitivity (peak-val checkpoint)")
    ax.set_title("Combined-test sensitivity by seed — ConvNeXt-pretrained")
    ax.set_ylim(0.0, 0.75)
    ax.axhline(0.589, ls="--", color="#4C72B0", alpha=0.4, lw=1)                   # bilinear mean guide
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "bar_sens_by_seed")


# ============================================================
# Figure 2 — size-stratified NODE21 (S / M / L bars per variant, mean across seeds)
# ============================================================
def fig_size_stratified(grouped):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = ["S", "M", "L"]
    bin_labels = ["S (<10mm)", "M (10-20mm)", "L (>=20mm)"]
    x = np.arange(len(bins))
    n_var = len(VARIANTS)
    bar_w = 0.8 / n_var

    for i, (var, label, color) in enumerate(VARIANTS):
        means, stds, ns = [], [], []
        for b in bins:
            vals = [r["size_stratified_node21"]["per_bin"][b]["sensitivity"]
                    for r in grouped[var] if "size_stratified_node21" in r]
            if vals:
                means.append(np.mean(vals)); stds.append(np.std(vals)); ns.append(len(vals))
            else:
                means.append(np.nan); stds.append(0); ns.append(0)
        offsets = x - 0.4 + bar_w * (i + 0.5)
        n_disp = ns[0] if ns else 0                                                # all bins have same n (same runs)
        ax.bar(offsets, means, bar_w, yerr=stds, capsize=3,
               label=f"{label} (n={n_disp})", color=color, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("NODE21 sensitivity")
    ax.set_title("Size-stratified NODE21 sensitivity — mean +/- std across seeds")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "bar_size_stratified")


# ============================================================
# Figure 3 — val -> test drop distribution per variant
# ============================================================
def fig_val_to_test_drop(grouped):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(VARIANTS))
    for i, (var, label, color) in enumerate(VARIANTS):
        drops = [r["val_to_test_drop"] for r in grouped[var]]
        if not drops: continue
        # jittered scatter + mean bar
        jitter = (np.random.default_rng(i).random(len(drops)) - 0.5) * 0.18
        ax.scatter(np.full(len(drops), i) + jitter, drops, color=color,
                   s=60, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.hlines(np.mean(drops), i - 0.25, i + 0.25, color=color, linewidth=3)
    ax.axhline(0, ls="-", color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([v[1] for v in VARIANTS], rotation=15, ha="right")
    ax.set_ylabel("val -> test drop (test_sens - best_val_sens)")
    ax.set_title("Val -> test generalisation drop  (negative = test worse than val)")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "bar_val_to_test_drop")


# ============================================================
# Figure 4 — FROC curves overlaid, mean across seeds
# ============================================================
def fig_froc_curves(grouped):
    fig, ax = plt.subplots(figsize=(8, 5))
    # Common FP grid for interpolation before averaging
    fp_grid = np.linspace(0, 4, 41)                                                # 0.1 step
    for var, label, color in VARIANTS:
        curves = []
        for r in grouped[var]:
            if not r.get("froc", {}).get("all"): continue
            pts = r["froc"]["all"]["curve"]
            fps  = np.array([p["fp"]   for p in pts])
            sens = np.array([p["sens"] for p in pts])
            if len(fps) < 2: continue
            interp = np.interp(fp_grid, fps, sens, left=sens[0], right=sens[-1])
            curves.append(interp)
        if not curves: continue
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ax.plot(fp_grid, mean, color=color, label=f"{label} (n={len(curves)})", linewidth=2)
        ax.fill_between(fp_grid, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlim(0, 4); ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Sensitivity")
    ax.set_title("FROC curves (all-test)  —  mean +/- std across seeds")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save_fig(fig, "froc_curves")


# ============================================================
# main
# ============================================================
def main():
    assert IN.exists(), f"Run scripts/eval/aggregate_v2_results.py first to build {IN}"
    grouped = load_runs()
    print(f"Loaded {sum(len(v) for v in grouped.values())} runs across {len(grouped)} variants")
    fig_sens_by_seed(grouped)
    fig_size_stratified(grouped)
    fig_val_to_test_drop(grouped)
    fig_froc_curves(grouped)
    print(f"\nAll plots -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
