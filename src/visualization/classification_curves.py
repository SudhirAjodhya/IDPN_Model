"""
Classification training curve visualizations for the Talebi Table 4 reproduction.

Per-run panels:
  - Loss curve (train + val on the same axis) - the overfit-detection view
  - Accuracy curve (train top-1, val top-1, val top-5)
  - Learning rate schedule (log scale)

Also provides confusion-matrix, per-seed phase comparison, and per-seed bar
chart helpers used by older Phase-A/B result JSONs. The v1 result JSONs remain
readable; new v2 JSONs simply carry extra keys (val_loss, macs, params).
"""
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_classification_curves(epoch_logs: list[dict], output_dir: Path,
                                phase_tag: str = "") -> None:
    """
    3-panel figure: Loss (train+val), Accuracy (top-1/top-5), Learning rate.

    epoch_logs may or may not carry val_loss (older JSONs pre-refactor lack it);
    the loss panel gracefully hides the val line when missing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs      = [d["epoch"] + 1 for d in epoch_logs]
    train_loss  = [d["train_loss"] for d in epoch_logs]
    val_loss    = [d.get("val_loss") for d in epoch_logs]
    has_val_loss = all(v is not None for v in val_loss)

    train_top1  = [d["train_top1"] for d in epoch_logs]
    val_top1    = [d["val_top1"] for d in epoch_logs]
    val_top5    = [d["val_top5"] for d in epoch_logs]
    lr          = [d["lr"] for d in epoch_logs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: loss curves (train vs val) - the overfitting-detection panel
    axes[0].plot(epochs, train_loss, "b-", label="Train Loss")
    if has_val_loss:
        axes[0].plot(epochs, val_loss, "r-", label="Val Loss")
        # Mark best val_loss epoch
        best_ep = epochs[int(np.argmin(val_loss))]
        best_vl = min(val_loss)
        axes[0].axvline(best_ep, color="r", linestyle=":", alpha=0.4,
                        label=f"Best Val Loss={best_vl:.3f} (ep {best_ep})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Val Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: accuracy (train top-1, val top-1, val top-5)
    best_val = max(val_top1)
    best_epoch = epochs[val_top1.index(best_val)]
    axes[1].plot(epochs, train_top1, "b-", alpha=0.5, label="Train Top-1")
    axes[1].plot(epochs, val_top1, "g-", label="Val Top-1")
    axes[1].plot(epochs, val_top5, "g--", alpha=0.4, label="Val Top-5")
    axes[1].axhline(best_val, color="g", linestyle=":", alpha=0.5,
                    label=f"Best Val Top-1={best_val:.2f}% (ep {best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Classification Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: LR schedule
    axes[2].plot(epochs, lr, "m-")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{phase_tag}" if phase_tag else ""
    plt.savefig(output_dir / f"training_curves{suffix}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: list[list[int]], output_dir: Path,
                           phase_tag: str = "") -> None:
    """Heatmap of the confusion matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm_arr, cmap="Blues")
    fig.colorbar(cax)

    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), va="center", ha="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {phase_tag}")

    plt.tight_layout()
    suffix = f"_{phase_tag}" if phase_tag else ""
    plt.savefig(output_dir / f"confusion_matrix{suffix}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Legacy helpers (v1 phase-A/B JSONs). Left intact so old plotting keeps working.
# ---------------------------------------------------------------------------

def save_phase_comparison(phase_a_logs: list[dict], phase_b_logs: list[dict],
                           output_dir: Path, seed: int = 0) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ep_a = [d["epoch"] + 1 for d in phase_a_logs]
    ep_b = [d["epoch"] + 1 for d in phase_b_logs]
    val_a = [d["val_top1"] for d in phase_a_logs]
    val_b = [d["val_top1"] for d in phase_b_logs]
    loss_a = [d["train_loss"] for d in phase_a_logs]
    loss_b = [d["train_loss"] for d in phase_b_logs]

    axes[0].plot(ep_a, val_a, "b-", label=f"Phase A (Bilinear 224): {val_a[-1]:.2f}%")
    axes[0].plot(ep_b, val_b, "r-", label=f"Phase B (Talebi 368→224): {val_b[-1]:.2f}%")
    delta = val_b[-1] - val_a[-1]
    axes[0].set_title(f"Val Top-1 Accuracy - Seed {seed}  (Δ = {delta:+.2f} pp)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep_a, loss_a, "b-", label="Phase A")
    axes[1].plot(ep_b, loss_b, "r-", label="Phase B")
    axes[1].set_title(f"Training Loss - Seed {seed}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"phase_comparison_seed{seed}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save_seed_comparison(results: dict[int, dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = sorted(results.keys())
    n = len(seeds)
    if n == 0:
        return

    a_top1 = [results[s]["A"]["final_val_top1"] for s in seeds]
    b_top1 = [results[s]["B"]["final_val_top1"] for s in seeds]
    deltas = [b - a for a, b in zip(a_top1, b_top1)]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, a_top1, width, label="Phase A (Bilinear 224)",
           color="#4A90D9", alpha=0.85)
    bars_b = ax.bar(x + width / 2, b_top1, width, label="Phase B (Talebi 368→224)",
                    color="#D94A4A", alpha=0.85)

    for bar, delta in zip(bars_b, deltas):
        ax.annotate(f"Δ {delta:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color="green" if delta > 0 else "red")

    mean_delta = sum(deltas) / len(deltas)
    ax.set_title(f"Talebi Reproduction - Mean Δ Top-1 = {mean_delta:+.2f} pp")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Val Top-1 Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "seed_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(result_dir: Path, output_dir: Optional[Path] = None) -> None:
    """Legacy convenience for v1 phase-A/B result dirs."""
    output_dir = output_dir or result_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_data: dict[str, dict[int, dict]] = {"A": {}, "B": {}}
    for f in sorted(result_dir.glob("phase*_seed*.json")):
        d = json.loads(f.read_text())
        if d.get("smoke", False):
            continue
        phase_data[d["phase"]][d["seed"]] = d

    for phase in ("A", "B"):
        for seed, data in phase_data[phase].items():
            save_classification_curves(
                data["epoch_logs"], output_dir,
                phase_tag=f"phase{phase}_seed{seed}",
            )
            if "cm" in data:
                save_confusion_matrix(
                    data["cm"], output_dir,
                    phase_tag=f"phase{phase}_seed{seed}",
                )

    common_seeds = sorted(set(phase_data["A"]) & set(phase_data["B"]))
    for seed in common_seeds:
        save_phase_comparison(
            phase_data["A"][seed]["epoch_logs"],
            phase_data["B"][seed]["epoch_logs"],
            output_dir, seed=seed,
        )

    if common_seeds:
        seed_results = {
            s: {"A": phase_data["A"][s], "B": phase_data["B"][s]}
            for s in common_seeds
        }
        save_seed_comparison(seed_results, output_dir)

    print(f"All plots saved to {output_dir}")
