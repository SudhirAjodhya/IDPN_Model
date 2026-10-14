"""
Training diagnostics for the VinDr multi-class sweep: is a run overfitting, and when?

WHAT THE OVERFITTING SIGNATURE LOOKS LIKE HERE. The classic tell is train loss falling
while validation loss rises. Where validation loss is unavailable, the weaker substitute
is train loss falling while the validation RANKING metric goes flat or declines - weaker
because a ranking metric tolerates a model becoming overconfident, so it turns over later
than the loss does.

WHAT IS AND IS NOT AVAILABLE
  train_loss      every run
  val_loss        only runs started after it was added to scripts/train_vindr.py, so the
                  v21 sweep and the first runs of v22 do not have it. The panel is drawn
                  and annotated as missing rather than silently omitted, because "no curve"
                  and "flat curve" must not look the same.
  val micro-AUC   every run - this is the checkpoint selection metric
  train AUC       NOTHING computes it. It needs an eval pass over the 5,272 training films
                  per epoch, which nothing in this project does. A train-vs-val AUC pair
                  would be the cleanest diagnostic and it is simply not in the data.

The selected epoch is marked on every panel. Everything to the right of that marker was
trained but discarded - the checkpoint is the best-validation epoch, so a run that overfits
past its peak wastes wall clock without degrading the reported result.
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("outputs/consolidation")

LABELS = {"bilinear": "Bilinear", "node21_talebi_zeroinit": "Talebi (zero-init)"}
COLOURS = {"bilinear": "#3b6ea5", "node21_talebi_zeroinit": "#b5443f"}


def load(run_group):
    """Every run in a group, keyed by (target, variant)."""
    runs = {}

    for path in Path("outputs/vindr").rglob("metrics.json"):
        metrics = json.load(open(path))

        if run_group and metrics.get("run_group") not in run_group.split(","):
            continue

        runs[(metrics["target_size"], metrics["variant"], metrics["seed"])] = metrics

    return runs


def main():
    # Declared before the parser, because --out-dir's default READS this name and a
    # global declaration must precede every use of it in the function
    global OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-group", default="v21_vindr_multiclass",
                        help="one or several comma-separated run groups")
    parser.add_argument("--seed", type=int, default=None, help="restrict to one seed")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    arguments = parser.parse_args()

    OUT_DIR = Path(arguments.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load(arguments.run_group)

    if arguments.seed is not None:
        runs = {k: v for k, v in runs.items() if k[2] == arguments.seed}

    if not runs:
        print(f"No runs found for run-group {arguments.run_group}")
        return

    targets = sorted({key[0] for key in runs})
    seeds = sorted({key[2] for key in runs})

    # Rows are targets, columns are the three diagnostics
    figure, axes = plt.subplots(len(targets), 3, figsize=(16.5, 3.4 * len(targets)),
                                squeeze=False)

    has_any_val_loss = False

    for row, target in enumerate(targets):
        loss_axis, val_loss_axis, auc_axis = axes[row]

        for variant in ("bilinear", "node21_talebi_zeroinit"):
            for seed in seeds:
                metrics = runs.get((target, variant, seed))

                if metrics is None: continue

                history = metrics["history"]
                epochs = np.arange(len(history["train_loss"]))
                best = metrics["best_epoch"]
                style = "-" if seed == seeds[0] else "--"
                label = "{} s{}".format(LABELS[variant], seed)

                loss_axis.plot(epochs, history["train_loss"], style,
                               color=COLOURS[variant], lw=1.4, label=label)
                loss_axis.axvline(best, color=COLOURS[variant], alpha=0.3, lw=1.0, ls=":")

                val_loss = history.get("val_loss")

                if val_loss:
                    has_any_val_loss = True
                    val_loss_axis.plot(epochs, val_loss, style, color=COLOURS[variant],
                                       lw=1.4, label=label)
                    val_loss_axis.axvline(best, color=COLOURS[variant], alpha=0.3,
                                          lw=1.0, ls=":")

                auc_axis.plot(epochs, history["val_micro_auc"], style,
                              color=COLOURS[variant], lw=1.4, label=label)
                auc_axis.plot(best, history["val_micro_auc"][best], "o",
                              color=COLOURS[variant], ms=6)

        loss_axis.set(ylabel="target {}\n\ntrain loss".format(target), xlabel="epoch")
        loss_axis.set_title("Training loss" if row == 0 else "")
        loss_axis.grid(alpha=0.25)

        val_loss_axis.set(xlabel="epoch", ylabel="val loss")
        val_loss_axis.set_title("Validation loss" if row == 0 else "")
        val_loss_axis.grid(alpha=0.25)

        if not val_loss_axis.lines:
            val_loss_axis.text(0.5, 0.5, "not logged for these runs\n"
                                         "(added to train_vindr.py after they started)",
                               ha="center", va="center", fontsize=9, color="dimgray",
                               transform=val_loss_axis.transAxes)
            val_loss_axis.set_xticks([])
            val_loss_axis.set_yticks([])

        auc_axis.set(xlabel="epoch", ylabel="val micro-AUC")
        auc_axis.set_title("Validation micro-AUC (selection metric)" if row == 0 else "")
        auc_axis.grid(alpha=0.25)

        if row == 0:
            loss_axis.legend(fontsize=8, loc="upper right")

    caption = ("Dotted vertical line and circle mark the SELECTED epoch. Everything right "
               "of it was trained and discarded - the checkpoint is the best-validation "
               "epoch, so overfitting past the peak costs wall clock, not accuracy.\n"
               "Overfitting reads as train loss still falling while the validation curve "
               "flattens or turns over. Train-set AUC is not computed anywhere in this "
               "project, so a train-vs-val AUC pair cannot be drawn from banked runs.")

    if not has_any_val_loss:
        caption += ("\nVALIDATION LOSS IS ABSENT from every run shown - it is the stronger "
                    "signal and these runs predate it being logged.")

    figure.text(0.5, 0.005, caption, ha="center", fontsize=8.5, color="dimgray")
    figure.tight_layout(rect=(0, 0.045 if len(targets) > 2 else 0.09, 1, 1))

    tag = arguments.run_group.replace(",", "+")

    if arguments.seed is not None:
        tag += "_s{}".format(arguments.seed)

    path = OUT_DIR / "fig11_vindr_training_curves_{}.png".format(tag)
    figure.savefig(path, dpi=180)
    print("wrote {}".format(path))

    # The numeric read, so the figure does not have to be squinted at
    print("\nOverfitting check: change from the SELECTED epoch to the LAST epoch trained")
    print("  {:<7} {:<20} {:>5} {:>6} {:>7} {:>12} {:>12}".format(
        "target", "variant", "seed", "best", "epochs", "train loss", "val microAUC"))

    for target in targets:
        for variant in ("bilinear", "node21_talebi_zeroinit"):
            for seed in seeds:
                metrics = runs.get((target, variant, seed))

                if metrics is None: continue

                history = metrics["history"]
                best = metrics["best_epoch"]
                train = history["train_loss"]
                auc = history["val_micro_auc"]

                print("  {:<7} {:<20} {:>5} {:>6} {:>7} {:>+11.1f}% {:>+12.4f}".format(
                    target, LABELS[variant], seed, best, len(train),
                    100 * (train[-1] - train[best]) / train[best],
                    auc[-1] - auc[best]))

    print("\n  Train loss down with val micro-AUC down = overfitting past the selected "
          "epoch.\n  Both near zero = the run stopped close to its peak.")


if __name__ == "__main__":
    main()
