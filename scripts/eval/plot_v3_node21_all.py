"""Consolidated plots for the v3 node21-only sweeps (v3_node21_sweep + v3_native_1024).

Reads outputs/results/aggregated_v2.json, filters to the two v3 node21 run groups,
and writes to outputs/results_v3_node21/plots/:

  1. sens_by_config.png           — grouped bars of NODE21 sens per (variant, input, target)
  2. sens_vs_resolution.png       — 3-panel line plots (Arm A vs Arm B, native-1024 paired, Talebi input=1024→T)
  3. froc_native_1024.png         — FROC curves at input=1024, one line per (variant, target)
  4. training_curves_v3.png       — train_loss + val_loss + val_sens per config, per variant

Kept separate from outputs/results/plots/ so v2 canonical plots don't get polluted.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
IN   = ROOT / "outputs" / "results" / "aggregated_v2.json"
OUT  = ROOT / "outputs" / "results_v3_node21" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# v3 node21 run groups
V3_GROUPS = {"v3_node21_sweep", "v3_native_1024"}

BILINEAR_COLOR = "#4C72B0"
TALEBI_COLOR   = "#55A868"


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.{{png,pdf}}")


def load_v3_runs():
    data = json.loads(IN.read_text())
    runs = [r for r in data["runs"]
            if r["run_group"] in V3_GROUPS
            and r["backbone"] == "convnext_fpn"
            and r["seed"] == 42]
    return runs


def sens_of(r, split="node21"):
    return r["test"].get(split, r["test"].get("combined", {})).get("sensitivity", np.nan)


# ============================================================
# Figure 1 — grouped bars per config
# ============================================================
def fig_sens_by_config(runs):
    fig, ax = plt.subplots(figsize=(14, 5.5))

    # Order rows: bilinear first, then Talebi. Within each: (input, target) sorted.
    def row_key(r):
        return (0 if r["variant"] == "bilinear" else 1, r["input_size"], r["target_size"])
    ordered = sorted(runs, key=row_key)

    labels = [f"{'B' if r['variant']=='bilinear' else 'T'}: {r['input_size']}→{r['target_size']}" for r in ordered]
    values = [sens_of(r, "node21") for r in ordered]
    colors = [BILINEAR_COLOR if r["variant"] == "bilinear" else TALEBI_COLOR for r in ordered]

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NODE21 test sensitivity")
    ax.set_title("All v3 node21-only configs — bilinear (blue) vs Talebi zero-init (green)")
    ax.set_ylim(0.0, 0.85)
    ax.grid(axis="y", alpha=0.3)

    # legend proxies
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=BILINEAR_COLOR, label="Bilinear"),
                       mpatches.Patch(color=TALEBI_COLOR,   label="Talebi zero-init")],
              loc="upper left")
    save_fig(fig, "sens_by_config")


# ============================================================
# Figure 2 — sens vs resolution (3 panels)
# ============================================================
def fig_sens_vs_resolution(runs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel A: bilinear input=target=X vs Talebi input=X target=224 (Arm A vs Arm B)
    a = axes[0]
    bilin_XX = sorted([r for r in runs if r["variant"]=="bilinear"
                       and r["input_size"] == r["target_size"]],
                      key=lambda r: r["input_size"])
    talebi_X_224 = sorted([r for r in runs if r["variant"]=="node21_talebi_zeroinit"
                           and r["target_size"] == 224],
                          key=lambda r: r["input_size"])
    if bilin_XX:
        xs = [r["input_size"] for r in bilin_XX]
        ys = [sens_of(r, "node21") for r in bilin_XX]
        a.plot(xs, ys, marker="o", color=BILINEAR_COLOR, linewidth=2, label="Bilinear (input=target=X)")
        for x, y in zip(xs, ys):
            a.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=BILINEAR_COLOR)
    if talebi_X_224:
        xs = [r["input_size"] for r in talebi_X_224]
        ys = [sens_of(r, "node21") for r in talebi_X_224]
        a.plot(xs, ys, marker="s", color=TALEBI_COLOR, linewidth=2, label="Talebi (input=X, target=224)")
        for x, y in zip(xs, ys):
            a.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=TALEBI_COLOR)
    a.set_xscale("log", base=2)
    if bilin_XX or talebi_X_224:
        xs = sorted({r["input_size"] for r in bilin_XX + talebi_X_224})
        a.set_xticks(xs); a.set_xticklabels([str(x) for x in xs])
    a.set_xlabel("Input resolution (px)")
    a.set_ylabel("NODE21 test sensitivity")
    a.set_title("v3 sweep — bilinear at input=target vs Talebi→224")
    a.grid(alpha=0.3); a.legend(loc="lower right", fontsize=9)

    # Panel B: native-1024 paired (both do 1024→T)
    b = axes[1]
    bilin_1024_T = sorted([r for r in runs if r["variant"]=="bilinear" and r["input_size"] == 1024],
                          key=lambda r: r["target_size"])
    talebi_1024_T = sorted([r for r in runs if r["variant"]=="node21_talebi_zeroinit" and r["input_size"] == 1024],
                           key=lambda r: r["target_size"])
    if bilin_1024_T:
        xs = [r["target_size"] for r in bilin_1024_T]
        ys = [sens_of(r, "node21") for r in bilin_1024_T]
        b.plot(xs, ys, marker="o", color=BILINEAR_COLOR, linewidth=2, label="Bilinear (1024→T)")
        for x, y in zip(xs, ys):
            b.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=BILINEAR_COLOR)
    if talebi_1024_T:
        xs = [r["target_size"] for r in talebi_1024_T]
        ys = [sens_of(r, "node21") for r in talebi_1024_T]
        b.plot(xs, ys, marker="s", color=TALEBI_COLOR, linewidth=2, label="Talebi (1024→T)")
        for x, y in zip(xs, ys):
            b.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=TALEBI_COLOR)
    b.set_xlabel("Target resolution T (input fixed at 1024)")
    b.set_ylabel("NODE21 test sensitivity")
    b.set_title("Native-1024 paired — both variants do 1024→T")
    b.grid(alpha=0.3); b.legend(loc="lower right", fontsize=9)
    if bilin_1024_T or talebi_1024_T:
        xs = sorted({r["target_size"] for r in bilin_1024_T + talebi_1024_T})
        b.set_xticks(xs); b.set_xticklabels([str(x) for x in xs])

    # Panel C: "does higher input help?" — compare bilinear at input=target vs bilinear 1024→target (at same target)
    c = axes[2]
    if bilin_XX and bilin_1024_T:
        # Match by target: bilin_XX[target=X, input=X] vs bilin_1024_T[target=X, input=1024]
        common_targets = sorted(set(r["target_size"] for r in bilin_XX) &
                                set(r["target_size"] for r in bilin_1024_T))
        if common_targets:
            xs = common_targets
            y_input_eq_tgt = [sens_of(next(r for r in bilin_XX if r["target_size"] == t), "node21") for t in xs]
            y_input_1024   = [sens_of(next(r for r in bilin_1024_T if r["target_size"] == t), "node21") for t in xs]
            c.plot(xs, y_input_eq_tgt, marker="o", color="#4C72B0", linewidth=2, label="Bilinear input=target")
            c.plot(xs, y_input_1024,   marker="s", color="#DD8452", linewidth=2, label="Bilinear input=1024→target")
            for x, y in zip(xs, y_input_eq_tgt):
                c.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
            for x, y in zip(xs, y_input_1024):
                c.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
            c.set_xticks(xs); c.set_xticklabels([str(x) for x in xs])
    c.set_xlabel("Target resolution T (px)")
    c.set_ylabel("NODE21 test sensitivity")
    c.set_title("Bilinear-only — does higher input res help?")
    c.grid(alpha=0.3); c.legend(loc="lower right", fontsize=9)

    fig.suptitle("v3 node21-only sweeps — sensitivity vs resolution", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "sens_vs_resolution")


# ============================================================
# Figure 3 — FROC curves at input=1024 (native-1024 paired)
# ============================================================
def fig_froc_native_1024(runs):
    fig, ax = plt.subplots(figsize=(9, 6))
    fp_grid = np.linspace(0, 4, 41)
    # Group by (variant, target)
    for r in sorted(runs, key=lambda r: (r["variant"], r["target_size"])):
        if r["input_size"] != 1024: continue
        if not r.get("froc", {}).get("node21"): continue
        pts = r["froc"]["node21"]["curve"]
        fps  = np.array([p["fp"] for p in pts])
        sens = np.array([p["sens"] for p in pts])
        if len(fps) < 2: continue
        interp = np.interp(fp_grid, fps, sens, left=sens[0], right=sens[-1])
        color = BILINEAR_COLOR if r["variant"] == "bilinear" else TALEBI_COLOR
        ls = {224: "-", 256: "--", 384: "-.", 512: ":"}.get(r["target_size"], "-")
        vlabel = "Bilinear" if r["variant"] == "bilinear" else "Talebi"
        ax.plot(fp_grid, interp, color=color, linestyle=ls, linewidth=1.8,
                label=f"{vlabel} 1024→{r['target_size']}  (AUC {r['froc']['node21']['auc_0_4fp']:.3f})")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("NODE21 sensitivity")
    ax.set_title("FROC curves at input=1024 — bilinear vs Talebi across targets")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "froc_native_1024")


# ============================================================
# Figure 4 — training curves per config
# ============================================================
def fig_training_curves(runs):
    """Two rows (bilinear, Talebi) x three cols (train loss, val loss, val sens).
    One line per (input, target) config within each variant panel."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Also need to read history from metrics.json (aggregate doesn't store per-epoch history)
    variant_rows = [("bilinear", BILINEAR_COLOR, 0),
                    ("node21_talebi_zeroinit", TALEBI_COLOR, 1)]

    # Colour by (input, target): use a colormap
    all_configs = sorted({(r["input_size"], r["target_size"]) for r in runs})
    cmap = plt.get_cmap("viridis")
    cfg_colors = {cfg: cmap(i / max(1, len(all_configs) - 1)) for i, cfg in enumerate(all_configs)}

    for variant, _, row in variant_rows:
        for col, (key, ylabel) in enumerate([("train_loss", "Train loss"),
                                              ("val_loss",   "Val loss"),
                                              ("sensitivity","Val sensitivity")]):
            ax = axes[row, col]
            variant_runs = [r for r in runs if r["variant"] == variant]
            for r in sorted(variant_runs, key=lambda r: (r["input_size"], r["target_size"])):
                # Read history from metrics.json
                run_dir = ROOT / r["checkpoint_path"].replace("\\", "/").rsplit("/", 1)[0]
                m = json.loads((run_dir / "metrics.json").read_text())
                curve = m["history"][key]
                xs = np.arange(1, len(curve) + 1)
                cfg = (r["input_size"], r["target_size"])
                ax.plot(xs, curve, color=cfg_colors[cfg], linewidth=1.5,
                        label=f"{r['input_size']}→{r['target_size']}")
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.set_title(f"{variant} — {ylabel}")
            ax.grid(alpha=0.3)
            if col == 0: ax.legend(loc="best", fontsize=7)

    fig.suptitle("v3 node21-only training curves", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "training_curves_v3")


def main():
    runs = load_v3_runs()
    print(f"Loaded {len(runs)} v3 node21-only runs")
    for r in runs:
        print(f"  {r['variant']:32s} i{r['input_size']:>4d}/t{r['target_size']:<4d}  NODE21 sens={sens_of(r, 'node21'):.3f}")

    print("\nGenerating plots:")
    fig_sens_by_config(runs)
    fig_sens_vs_resolution(runs)
    fig_froc_native_1024(runs)
    fig_training_curves(runs)
    print(f"\nAll plots -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
