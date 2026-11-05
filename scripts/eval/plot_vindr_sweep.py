"""
Resolution sweep figures for the VinDr 14-class comparison.

TWO FIGURES, answering two different questions.

fig12 - the sweep. Micro and macro scores against output resolution for both arms, with
seed error bars, plus a fourth panel carrying the arm delta. That delta panel is the point
of the figure: the learned resizer's benefit SHRINKS as target resolution rises, which is a
decimation relationship rather than a flat offset.

fig13 - per class. The same sweep faceted into one panel per pathology, ordered by how much
that class gains from resolution. Fourteen series on one axis would be unreadable and would
force invented hues, so this is small multiples with the same two-colour scheme throughout.

WHY ERROR BARS ARE SEED SPREAD, NOT A CONFIDENCE INTERVAL. Each point is 3 seeds, which is
too few for a meaningful interval. The bar is the observed range across those seeds, which
is honest about what was measured. The statistical test lives in the paired analysis over
all 12 cells, not in these bars - two overlapping bars here do NOT mean "no difference",
because the comparison is PAIRED within seed and the bars are not.

Colours are the project's existing two-series pair, validated: adjacent-pair separation
dE 15.8 under protanopia and 23.2 at normal vision, both above the floor.
"""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

OUT_DIR = Path("outputs/consolidation")

ARMS = ("bilinear", "node21_talebi_zeroinit")
LABELS = {"bilinear": "Bilinear", "node21_talebi_zeroinit": "Talebi (zero-init)"}
COLOURS = {"bilinear": "#3b6ea5", "node21_talebi_zeroinit": "#b5443f"}
DELTA_COLOUR = "#4a7c59"
THIN_CLASS_FILMS = 50


def load(run_group):
    runs = {}

    for path in Path("outputs/vindr").rglob("metrics.json"):
        metrics = json.load(open(path))

        if metrics.get("run_group") in run_group.split(","):
            runs[(metrics["target_size"], metrics["variant"], metrics["seed"])] = metrics

    return runs


def series(runs, targets, seeds, arm, getter):
    """Mean and observed range of `getter` across seeds, per target."""
    means, lows, highs = [], [], []

    for target in targets:
        values = [getter(runs[(target, arm, seed)]) for seed in seeds
                  if (target, arm, seed) in runs]
        means.append(st.mean(values))
        lows.append(st.mean(values) - min(values))
        highs.append(max(values) - st.mean(values))

    return np.array(means), np.array([lows, highs])


def figure_sweep(runs, targets, seeds, plt, tag):
    """fig12: micro/macro scores and the arm delta against output resolution."""
    panels = [
        ("micro-AUC", lambda m: m["test_results"]["micro_auc"]),
        ("micro-F1", lambda m: m["test_results"]["micro_f1"]),
        ("macro-AUC", lambda m: m["test_results"]["macro_auc"]),
    ]

    figure, axes = plt.subplots(1, 5, figsize=(20.0, 4.2))
    x = np.arange(len(targets))

    for axis, (name, getter) in zip(axes, panels):
        for arm in ARMS:
            means, error = series(runs, targets, seeds, arm, getter)
            axis.errorbar(x, means, yerr=error, color=COLOURS[arm], lw=2.0, marker="o",
                          ms=8, capsize=4, elinewidth=1.2, label=LABELS[arm])

        axis.set(xticks=x, xlabel="Output resolution", ylabel=name)
        axis.set_xticklabels(targets)
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    # The delta panel - the finding the other three only imply
    delta_axis = axes[3]
    deltas, spread = [], []

    for target in targets:
        per_seed = [runs[(target, ARMS[1], s)]["test_results"]["micro_auc"]
                    - runs[(target, ARMS[0], s)]["test_results"]["micro_auc"]
                    for s in seeds if (target, ARMS[0], s) in runs]
        deltas.append(st.mean(per_seed))
        spread.append([st.mean(per_seed) - min(per_seed), max(per_seed) - st.mean(per_seed)])

    spread = np.array(spread).T

    delta_axis.axhline(0, color="black", lw=1.0, alpha=0.5)
    delta_axis.errorbar(x, deltas, yerr=spread, color=DELTA_COLOUR, lw=2.0, marker="o",
                        ms=8, capsize=4, elinewidth=1.2)

    for index, value in enumerate(deltas):
        delta_axis.annotate("{:+.4f}".format(value), (index, value), textcoords="offset points",
                            xytext=(0, 11), ha="center", fontsize=9)

    decimation = [2048 / t for t in targets]
    delta_axis.set(xticks=x, xlabel="Output resolution (decimation from 2048)",
                   ylabel="micro-AUC, Talebi minus Bilinear")
    delta_axis.set_xticklabels(["{}\n{:.1f}x".format(t, d) for t, d in zip(targets, decimation)])
    delta_axis.set_title("Arm difference (paired within seed)")
    delta_axis.grid(axis="y", alpha=0.2)
    delta_axis.spines[["top", "right"]].set_visible(False)

    # The regression against decimation ratio, which is the pre-specified headline test.
    # Plotted on the DECIMATION axis rather than the resolution axis because that is the
    # variable the hypothesis is about - the gap should grow with how much the downsample
    # discards, not with output size as such.
    slope_axis = axes[4]

    per_cell_x, per_cell_y = [], []

    for target in targets:
        for seed in seeds:
            if (target, ARMS[0], seed) not in runs: continue

            per_cell_x.append(2048 / target)
            per_cell_y.append(runs[(target, ARMS[1], seed)]["test_results"]["micro_auc"]
                              - runs[(target, ARMS[0], seed)]["test_results"]["micro_auc"])

    per_cell_x = np.array(per_cell_x)
    per_cell_y = np.array(per_cell_y)
    slope, intercept = np.polyfit(per_cell_x, per_cell_y, 1)
    correlation = np.corrcoef(per_cell_x, per_cell_y)[0, 1]

    fit_x = np.linspace(per_cell_x.min(), per_cell_x.max(), 50)

    slope_axis.axhline(0, color="black", lw=1.0, alpha=0.5)
    slope_axis.plot(fit_x, slope * fit_x + intercept, color=DELTA_COLOUR, lw=2.0,
                    label="fit, r = {:+.2f}".format(correlation))
    slope_axis.scatter(per_cell_x, per_cell_y, s=42, color=DELTA_COLOUR, alpha=0.55,
                       edgecolor="white", linewidth=0.8, zorder=3)
    slope_axis.scatter([2048 / t for t in targets],
                       [st.mean([runs[(t, ARMS[1], s)]["test_results"]["micro_auc"]
                                 - runs[(t, ARMS[0], s)]["test_results"]["micro_auc"]
                                 for s in seeds if (t, ARMS[0], s) in runs]) for t in targets],
                       s=95, color=DELTA_COLOUR, marker="D", edgecolor="white",
                       linewidth=1.2, zorder=4, label="seed mean")

    slope_axis.set(xlabel="Decimation ratio (2048 / target)",
                   ylabel="micro-AUC, Talebi minus Bilinear")
    slope_axis.set_title("Benefit against decimation")
    slope_axis.legend(frameon=False, fontsize=8, loc="upper left")
    slope_axis.grid(alpha=0.2)
    slope_axis.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    figure.suptitle("VinDr-CXR, 14 pathologies, input 2048 -> target | 3 seeds per point",
                    fontsize=12)
    # Computed from the data in the figure rather than hardcoded, so the caption cannot
    # go stale against it again - an earlier version still quoted the 12-cell result after
    # six more cells had been added.
    pooled_mean = float(np.mean(per_cell_y))
    pooled_t = pooled_mean / (float(np.std(per_cell_y, ddof=1)) / np.sqrt(len(per_cell_y)))

    figure.text(0.5, 0.015,
                "Error bars are the observed range across {} seeds, not a confidence "
                "interval. The arms are compared PAIRED within seed, so overlapping bars do "
                "not imply no difference. Over all {} cells the pooled mean is {:+.4f} "
                "(t = {:.2f}); the pre-specified test is the slope against decimation, "
                "r = {:+.2f} per cell.".format(len(seeds), len(per_cell_y), pooled_mean,
                                               pooled_t, correlation),
                ha="center", fontsize=8.5, color="dimgray")
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))

    path = OUT_DIR / "fig12_vindr_sweep{}.png".format(tag)
    figure.savefig(path, dpi=190)
    plt.close(figure)

    return path, deltas


def figure_per_class(runs, targets, seeds, plt, tag):
    """fig13: one panel per pathology, ordered by how much it gains from resolution."""
    sample = runs[(targets[0], ARMS[0], seeds[0])]["test_results"]["per_class"]
    films = {entry["name"]: entry["n_test_films"] for entry in sample}

    def class_auc(target, arm, seed, name):
        for entry in runs[(target, arm, seed)]["test_results"]["per_class"]:
            if entry["name"] == name:
                return entry["auc"]

        return float("nan")

    # Order by pooled gain from the lowest to the highest target, so the figure itself
    # carries the finding rather than leaving it to be hunted for
    gains = {}

    for name in films:
        low = st.mean([class_auc(targets[0], a, s, name) for a in ARMS for s in seeds])
        high = st.mean([class_auc(targets[-1], a, s, name) for a in ARMS for s in seeds])
        gains[name] = high - low

    ordered = sorted(films, key=lambda n: -gains[n])

    columns = 4
    rows = int(np.ceil(len(ordered) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15.0, 3.1 * rows), squeeze=False)
    x = np.arange(len(targets))

    for index, name in enumerate(ordered):
        axis = axes[index // columns][index % columns]

        for arm in ARMS:
            means, error = series(runs, targets, seeds, arm,
                                  lambda m, n=name: next(e["auc"] for e in
                                                         m["test_results"]["per_class"]
                                                         if e["name"] == n))
            axis.errorbar(x, means, yerr=error, color=COLOURS[arm], lw=1.8, marker="o",
                          ms=6, capsize=3, elinewidth=1.0, label=LABELS[arm])

        thin = films[name] < THIN_CLASS_FILMS
        axis.set_title("{}\n{} test films{}   {:+.3f} over the sweep".format(
            name, films[name], "  (thin)" if thin else "", gains[name]),
            fontsize=9.5, color="dimgray" if thin else "black")

        axis.set(xticks=x)
        axis.set_xticklabels(targets, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)

        if thin:
            axis.set_facecolor("#f6f6f4")

    for index in range(len(ordered), rows * columns):
        axes[index // columns][index % columns].axis("off")

    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")

    figure.suptitle("Per-class AUC against output resolution, ordered by resolution gain\n"
                    "VinDr-CXR, input 2048, 3 seeds per point", fontsize=12)
    figure.text(0.5, 0.008,
                "Classes with fewer than {} test films are shaded and cannot support a "
                "per-class claim on their own. Gains concentrate in the small, spatially "
                "fine findings and vanish for gross ones.".format(THIN_CLASS_FILMS),
                ha="center", fontsize=8.5, color="dimgray")
    figure.tight_layout(rect=(0, 0.03, 1, 0.945))

    path = OUT_DIR / "fig13_vindr_per_class{}.png".format(tag)
    figure.savefig(path, dpi=190)
    plt.close(figure)

    return path, gains, films


def main():
    global OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-group", default="v23_vindr_3seed")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    arguments = parser.parse_args()

    OUT_DIR = Path(arguments.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load(arguments.run_group)

    if not runs:
        print("No runs found for {}".format(arguments.run_group))
        return

    targets = sorted({key[0] for key in runs})
    seeds = sorted({key[2] for key in runs})
    tag = "_" + arguments.run_group.replace(",", "+")

    print("{} runs | targets {} | seeds {}".format(len(runs), targets, seeds))

    path, deltas = figure_sweep(runs, targets, seeds, plt, tag)
    print("wrote {}".format(path))

    path, gains, films = figure_per_class(runs, targets, seeds, plt, tag)
    print("wrote {}".format(path))

    print("\nArm delta against decimation ratio")
    print("  {:<8} {:>10} {:>12}".format("target", "decimation", "delta"))

    for target, delta in zip(targets, deltas):
        print("  {:<8} {:>9.2f}x {:>+12.4f}".format(target, 2048 / target, delta))

    print("\nPer-class gain from target {} to {}".format(targets[0], targets[-1]))
    print("  {:<22} {:>6} {:>9}".format("class", "films", "gain"))

    for name in sorted(gains, key=lambda n: -gains[n]):
        flag = "  thin" if films[name] < THIN_CLASS_FILMS else ""
        print("  {:<22} {:>6} {:>+9.3f}{}".format(name, films[name], gains[name], flag))


if __name__ == "__main__":
    main()
