"""Plot train_loss + val_loss + val_sensitivity per epoch for the four canonical
v2 variants at the reference config (ConvNeXt-pretrained, target=384, batch=8).

Reads each checkpoint's metrics.json history array. One figure with three panels.
Multiple seeds per variant show as mean +/- 1 std band across seeds.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "outputs" / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical v2 comparison (matches plot_v2_results.py palette + names)
VARIANTS = [
    ("bilinear",                    "Bilinear (baseline)",    "#4C72B0"),
    ("node21_talebi",               "Talebi (paper-faithful)", "#DD8452"),
    ("node21_talebi_zeroinit",      "Talebi (zero-init)",     "#55A868"),
    ("node21_talebi_zeroinit_attn", "Zero-init + attention",  "#C44E52"),
]


def load_histories():
    """Return {variant: [history dict, ...]}, one per seed, ConvNeXt-pretrained target=384 only."""
    grouped = defaultdict(list)
    for metrics_path in (ROOT / "outputs" / "v8").glob("*_v2_*/*/metrics.json"):
        m = json.loads(metrics_path.read_text())
        # Filter: default backbone (no _<backbone> suffix in dir name), target=384
        if not metrics_path.parent.name.endswith(f"s{m['seed']}"):                 # non-default backbone
            continue
        if m["heatmap_size"] * 4 != 384:                                            # target=384 only
            continue
        grouped[m["variant"]].append(m["history"] | {"seed": m["seed"], "epochs_trained": m["epochs_trained"]})
    return grouped


def pad_and_stack(curves, max_len):
    """Right-pad each 1D array to max_len with the last value (or NaN if you prefer);
    return (n_seeds, max_len) array."""
    arr = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
    return arr


def plot_panel(ax, grouped, key, ylabel, title, ylog=False):
    max_len = max(len(h[key]) for hs in grouped.values() for h in hs)
    xs = np.arange(1, max_len + 1)
    for var, label, color in VARIANTS:
        if var not in grouped: continue
        curves = [h[key] for h in grouped[var]]
        arr = pad_and_stack(curves, max_len)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)
        ax.plot(xs, mean, color=color, label=f"{label} (n={len(curves)})", linewidth=2)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main():
    grouped = load_histories()
    print(f"Loaded histories for {sum(len(v) for v in grouped.values())} runs across {len(grouped)} variants")
    for v, hs in grouped.items():
        print(f"  {v}: seeds {sorted(h['seed'] for h in hs)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_panel(axes[0], grouped, "train_loss",   "Train loss",        "Training loss")
    plot_panel(axes[1], grouped, "val_loss",     "Val loss",          "Validation loss")
    plot_panel(axes[2], grouped, "sensitivity",  "Val sensitivity",   "Validation sensitivity (peak-val early stop)")
    fig.suptitle("Training curves — v2 recipe, ConvNeXt-pretrained, target=384", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"training_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote training_curves.{{png,pdf}} -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
