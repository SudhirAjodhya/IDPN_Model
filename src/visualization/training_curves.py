import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_training_curves(history, output_dir):
    """
    Plot multiple graphs on one plot
    Graphs: Train vs Val loss, Sensitivity
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    
    # Train loss vs validation loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", alpha=0.3, label="Val (raw)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # sensitivity
    best_s = max(history["sensitivity"]) if history["sensitivity"] else 0
    axes[1].plot(epochs, history["sensitivity"], "g-", label="Val Sensitivity")
    axes[1].axhline(best_s, color="g", linestyle="--", alpha=0.5, label=f"Best={best_s:.3f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sensitivity")
    axes[1].set_title("Val Sensitivity")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if history.get("lr"):
        axes[2].plot(epochs, history["lr"], "m-")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR"); axes[2].set_title("Learning Rate")
        axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
