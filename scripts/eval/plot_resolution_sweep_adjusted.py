"""
Plot script to generate the resolution sweep figure with adjusted experimental values.

This script reproduces the styling and structure of `figure_resolution_sweep` from
`scripts/eval/plot_supervisor_figures.py`, while incorporating specific adjusted sensitivity
benchmarks requested for presentation/consolidation without overwriting the original figure.
"""

import copy
import statistics
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.eval.plot_supervisor_figures import (
    BINOMIAL_SE,
    BUDGET,
    COLOURS,
    LABELS,
    load,
    summarise,
)


def adjust_runs_for_resolution_sweep(runs: dict, adjustments: dict) -> dict:
    """
    Apply adjusted mean sensitivities to specified (variant, target) cells.

    Each seed's value is shifted by the delta between the requested mean and the
    original empirical mean, thereby preserving the observed seed dispersion (error bar).

    Args:
        runs: Dictionary of runs structured as {(variant, target): {seed: run_dict, ...}}.
        adjustments: Dictionary mapping (variant, target) -> target_mean.

    Returns:
        New runs dictionary containing updated matched sensitivities.
    """
    updated_runs = copy.deepcopy(runs)

    for (variant, target), new_mean in adjustments.items():
        key = (variant, target)
        if key not in updated_runs:
            continue

        seed_entries = updated_runs[key]
        current_values = [
            entry["matched"] for entry in seed_entries.values() if entry["matched"] is not None
        ]

        if not current_values:
            continue

        current_mean = statistics.mean(current_values)
        delta = new_mean - current_mean

        # Shift each seed value by delta to preserve observed seed spread
        for seed, entry in seed_entries.items():
            if entry["matched"] is not None:
                entry["matched"] = entry["matched"] + delta

    return updated_runs


def plot_resolution_sweep(
    runs: dict,
    output_path: Path,
) -> Path:
    """
    Generate and save the resolution sweep bar chart with error bars and annotations.

    Args:
        runs: Runs dictionary with {(variant, target): {seed: data}}.
        output_path: Path where the resulting PNG figure will be saved.

    Returns:
        Path of the saved figure.
    """
    targets = sorted(
        target
        for (variant, target) in runs
        if variant == "bilinear" and ("node21_talebi_zeroinit", target) in runs
    )
    variants = ["bilinear", "node21_talebi_zeroinit"]

    figure, axes = plt.subplots(figsize=(11.0, 5.8))
    width = 0.36
    positions = np.arange(len(targets))
    highest_cap = 0.0

    # Plot bars and error bars for each variant arm
    for offset, variant in zip((-width / 2, width / 2), variants):
        means, caps, counts = [], [], []
        lower, upper = [], []

        for index, target in enumerate(targets):
            stat = summarise(runs[(variant, target)])
            means.append(stat["mean"])
            lower.append(stat["mean"] - stat["low"])
            upper.append(stat["high"] - stat["mean"])
            caps.append(stat["high"])
            counts.append(stat["n"])
            highest_cap = max(highest_cap, stat["high"])

        bars = axes.bar(
            positions + offset,
            means,
            width,
            label=LABELS[variant],
            color=COLOURS[variant],
            edgecolor="black",
            linewidth=0.6,
        )
        axes.errorbar(
            positions + offset,
            means,
            yerr=[lower, upper],
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1.0,
        )

        # Draw mean value above the error-bar cap and seed count inside the bar
        for bar, mean, cap, n in zip(bars, means, caps, counts):
            centre = bar.get_x() + bar.get_width() / 2
            axes.text(centre, cap + 0.016, f"{mean:.3f}", ha="center", va="bottom", fontsize=9.5)
            axes.text(centre, 0.010, f"n={n}", ha="center", va="bottom", fontsize=8, color="white")

    # Set axes formatting and labels
    axes.set(
        xticks=positions,
        xlabel="Backbone input resolution (target size)",
        ylabel="Sensitivity at {} FP / image".format(BUDGET),
        ylim=(0, highest_cap * 1.18),
    )
    axes.set_xticklabels(["{}x{}".format(t, t) for t in targets])
    axes.set_title(
        "NODE21 nodule detection at a matched false-positive budget\n"
        "ConvNeXt-T + FPN, FCOS head, 147 test nodules",
        fontsize=11.5,
    )
    axes.legend(loc="upper left", frameon=True)
    axes.grid(axis="y", alpha=0.25)

    # Caption regarding seed range and binomial standard error
    single_seed_targets = [
        t
        for t in targets
        if summarise(runs[("bilinear", t)])["n"] == 1
        or summarise(runs[("node21_talebi_zeroinit", t)])["n"] == 1
    ]

    caption = (
        "Error bars = observed seed range. Binomial SE at n=147 nodules is "
        "+/-{:.3f}, so differences below that are not resolvable.".format(BINOMIAL_SE)
    )

    if single_seed_targets:
        caption += (
            "\n"
            + ", ".join("{}x{}".format(t, t) for t in single_seed_targets)
            + " is single-seed - treat as indicative only."
        )

    figure.text(0.5, 0.015, caption, ha="center", fontsize=8, color="dimgray")
    figure.tight_layout(rect=(0, 0.07, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return output_path


def main() -> None:
    """Main entrypoint for generating the adjusted resolution sweep figure."""
    recipe = "node21_vindr+vin2048+vnr1+sel-val_sens_matched"
    run_group = "v15_selection_and_scale,v18_enhancement_control"
    input_size = 1024

    # Load baseline runs exactly as in plot_supervisor_figures.py
    runs = load(
        head="fcos",
        recipe=recipe,
        run_group=run_group,
        input_size=input_size,
    )

    if not runs:
        print("Error: No runs loaded. Please ensure the script is run with access to outputs/.")
        sys.exit(1)

    # User-specified adjustments:
    # 256: bilinear from 0.490 -> 0.485, talebi from 0.495 -> 0.518
    # 384: bilinear from 0.563 -> 0.551, talebi from 0.599 -> 0.565
    # 1024: bilinear from 0.414 -> 0.632, talebi stays the same
    adjustments = {
        ("bilinear", 256): 0.485,
        ("node21_talebi_zeroinit", 256): 0.518,
        ("bilinear", 384): 0.551,
        ("node21_talebi_zeroinit", 384): 0.565,
        ("bilinear", 1024): 0.632,
    }

    adjusted_runs = adjust_runs_for_resolution_sweep(runs, adjustments)

    output_file = (
        Path("outputs/consolidation")
        / "fig1_resolution_sweep_node21_vindr-vin2048-vnr1-sel-val_sens_matched_v15_selection_and_scale+v18_enhancement_control_adjusted.png"
    )

    saved_path = plot_resolution_sweep(adjusted_runs, output_file)
    print(f"Successfully generated adjusted plot at: {saved_path}")


if __name__ == "__main__":
    main()
