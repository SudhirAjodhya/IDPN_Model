"""
Train the multi-class VinDr detector for the resolution sweep, scored as multi-label.

SEPARATE FROM scripts/train.py ON PURPOSE. train.py is NODE21-shaped all the way through:
FROC on the fixed 517-film set, matched 0.25 FP/img checkpoint selection, LUNA16 size
stratification, residual attribution against size-matched control boxes. None of that
applies to a 14-class VinDr sweep, and threading a multi-label branch through every one of
those assumptions would put the project's banked nodule results at risk for no benefit.
This script shares the model, loss, data and split modules and nothing else.

WHAT IT TRAINS. A multi-class FCOS detector on VinDr's boxes - 14 classes, one channel
each. Boxes are kept rather than training a plain classifier because localised supervision
is what gives the resizer spatial gradient to learn from, and because it keeps the task
continuous with the nodule track.

WHAT IT REPORTS. Image-level multi-label scores. Each film is reduced to one score per
class by taking the max of sqrt(cls x centerness) over the grid, then scored as
multi-label classification: micro-F1 and micro-AUC as the headline, macro beside them to
expose rare-class collapse, and a per-class table. See
src/evaluation/vindr_multilabel_metrics.py for why micro is the one to quote.

THRESHOLD DISCIPLINE. The decision threshold that maximises micro-F1 is FITTED, so it is
fitted on validation and applied unchanged to test. Fitting it on test would be reading
the answer off the test set, which is the same class of error as selecting a checkpoint on
one metric and reporting another - a mistake this project has already made once and
corrected in v15.

CHECKPOINT SELECTION is on validation micro-AUC, which is threshold-free, so selection and
reporting cannot drift apart the way val_sens and val_sens_matched did.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.vindr import VinDrDataset, NUM_VINDR_CLASSES, MAX_OBJECTS, VINDR_ROOT
from src.data.vindr_split import build_vindr_multilabel_split, repeat_factors
from src.evaluation.fcos_targets import make_fcos_targets
from src.evaluation.vindr_multilabel_metrics import evaluate, format_report, image_scores
from src.losses.fcos import fcos_loss
from src.models.detector import NoduleDetector

ANNOTATIONS = VINDR_ROOT / "train.csv"
IMAGES = VINDR_ROOT / "train_png16"


def build_loaders(arguments):
    """Three VinDr loaders, with rare-class oversampling on train only."""
    cache_dir = VINDR_ROOT / f"train_png16_cache_i{arguments.input_size}"

    # The PNG16 archive was deleted once the resize caches were written, so the cache IS
    # the image source and the split must be restricted to what it holds. All 4,394
    # positive films are cached - verified - so every class keeps its full count; only the
    # negative pool is smaller, 4,394 instead of 10,606.
    if not cache_dir.exists():
        raise SystemExit(
            f"No VinDr cache at {cache_dir}. The PNG16 archive is gone, so this cache is "
            f"the only image source at input {arguments.input_size}. Warm it with "
            f"scripts/warm_vindr_cache.py --input-size {arguments.input_size}.")

    available = sorted(path.stem for path in cache_dir.glob("*.npy"))

    splits = build_vindr_multilabel_split(ANNOTATIONS, seed=arguments.split_seed,
                                          available_ids=available)

    def dataset(image_ids, augment):
        return VinDrDataset(
            annotations_csv=ANNOTATIONS, images_root=IMAGES, image_ids=image_ids,
            augment=augment, input_size=arguments.input_size, clahe=True,
            aug_profile=arguments.aug_profile, multiclass=True, cache_dir=cache_dir)

    train_set = dataset(splits["train"], augment=arguments.augment)
    val_set = dataset(splits["val"], augment=False)
    test_set = dataset(splits["test"], augment=False)

    # A held-IN diagnostic set: training films, scored the same way validation is. Without
    # it the only train-side number is the training objective, which is computed on
    # AUGMENTED films with gradients on, and is therefore not comparable to anything on the
    # validation curve. A train-vs-val AUC pair is the cleanest overfitting signal there is,
    # and it only means something if the two are measured identically.
    #
    # UNAUGMENTED on purpose. Scoring augmented training films would depress the train
    # curve by the augmentation noise and mask the very gap being looked for, so the only
    # difference from validation here is "the model trained on these films".
    #
    # Fixed subsample, size-matched to validation, drawn with a fixed seed so the curve is
    # comparable across epochs, arms and runs. Costs one validation-sized pass per epoch.
    train_eval_set = None

    if arguments.train_eval_films > 0:
        generator = np.random.default_rng(0)
        count = min(arguments.train_eval_films, len(splits["train"]))
        chosen = generator.choice(len(splits["train"]), size=count, replace=False)
        train_eval_set = dataset([splits["train"][i] for i in sorted(chosen)], augment=False)

    # Rare-class oversampling. Films are drawn with probability proportional to their
    # repeat factor, so a Pneumothorax film appears several times an epoch - each time
    # differently augmented, because VinDrDataset augments inside __getitem__. Train only:
    # val and test are drawn once each, unaugmented.
    sampler = None

    if arguments.oversample:
        ids_in_order = [record[0] for record in train_set.records]
        weights = repeat_factors(ANNOTATIONS, ids_in_order, power=arguments.oversample_power)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(round(float(weights.sum()))), replacement=True)
        print(f"[sampler] oversampling {weights.min():.2f}x to {weights.max():.2f}x, "
              f"epoch is {int(weights.sum())} draws from {len(ids_in_order)} films")

    loader_kwargs = dict(num_workers=arguments.num_workers, pin_memory=True,
                         persistent_workers=arguments.num_workers > 0)

    train_loader = DataLoader(train_set, batch_size=arguments.batch_size, sampler=sampler,
                              shuffle=sampler is None, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=arguments.batch_size, shuffle=False,
                            **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=arguments.batch_size, shuffle=False,
                             **loader_kwargs)

    train_eval_loader = None

    if train_eval_set is not None:
        train_eval_loader = DataLoader(train_eval_set, batch_size=arguments.batch_size,
                                       shuffle=False, **loader_kwargs)

    return (train_loader, val_loader, test_loader, train_eval_loader,
            {k: len(v) for k, v in splits.items()})


def run_epoch(model, loader, optimiser, device, heatmap_size, input_size, grad_accum):
    model.train()
    total, batches = 0.0, 0

    optimiser.zero_grad(set_to_none=True)

    for step, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(images)

        truth = make_fcos_targets(target, images.shape[0], heatmap_size, input_size,
                                  device, max_objects=MAX_OBJECTS,
                                  num_classes=NUM_VINDR_CLASSES)

        loss, _ = fcos_loss(prediction, truth)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

        total += float(loss.detach())
        batches += 1

    return total / max(batches, 1)


@torch.no_grad()
def validate(model, loader, device, heatmap_size, input_size):
    """
    One pass over validation returning image-level scores AND the task loss.

    WHY THE LOSS IS HERE. Selection runs on micro-AUC, so an earlier version of this
    script logged only the ranking metrics - and that leaves the classic overfitting
    signature invisible. Train loss falling while VAL LOSS rises is the standard tell, and
    on the NODE21 track it was what exposed the focal normaliser defect: the failing arm
    sat at val_loss 70-325 against 0.5-2.0 for the healthy one, which no sensitivity number
    showed anywhere near as plainly.

    Loss and scores share one forward pass, so this costs the target assignment only, not
    a second pass over the validation split.

    The two are complementary rather than redundant. A ranking metric can hold flat while
    the loss climbs, because a model that grows overconfident loses calibration without
    losing rank order. Neither alone tells the whole story.

    Returns:
        scores (N, C), labels (N, C), mean loss over batches
    """
    scores, labels, losses = [], [], []

    for batch in loader:
        images, target = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) \
            else (batch["image"], batch)
        images = images.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(images)

        truth = make_fcos_targets(target, images.shape[0], heatmap_size, input_size,
                                  device, max_objects=MAX_OBJECTS,
                                  num_classes=NUM_VINDR_CLASSES)

        loss, _ = fcos_loss(prediction, truth)
        losses.append(float(loss))

        confidence = (torch.sigmoid(prediction["classification"].float())
                      * torch.sigmoid(prediction["centerness"].float())).sqrt()

        scores.append(confidence.amax(dim=(2, 3)).cpu().numpy())
        labels.append(target["image_labels"].numpy())

    return (np.concatenate(scores), np.concatenate(labels),
            float(np.mean(losses)) if losses else float("nan"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="bilinear")
    parser.add_argument("--backbone", default="convnext_fpn")
    parser.add_argument("--input-size", type=int, default=2048,
                        help="VinDr is natively 2430-3408 px. At 1024 the dataloader's "
                             "INTER_AREA already performs a 2.8x anti-aliased decimation, "
                             "leaving the resizer under test a pre-solved problem.")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--select-smoothing", type=int, default=1,
                        help="Window, in epochs, of the trailing mean of validation "
                             "micro-AUC that checkpoints are selected on. 1 selects on the "
                             "raw per-epoch value, which is what every run banked before "
                             "this flag existed did. USE 5. "
                             "WHY: validation micro-AUC plateaus and then oscillates by "
                             "about +/-0.002, so taking the max over 40-80 plateau epochs "
                             "is an extreme-value draw. MEASURED across the 16 v21/v22 "
                             "runs it inflated the selected value by +0.0029 to +0.0082, "
                             "mean +0.0045, against a resizer effect of +0.0023 - the "
                             "artefact is twice the signal. It is also asymmetric: the arm "
                             "that trains longer gets more draws, and epochs trained "
                             "correlated with the arm delta at r = +0.55. Smoothing picks "
                             "the centre of the plateau rather than its luckiest epoch. "
                             "Changing this changes the SELECTION RULE, so it must be the "
                             "same for every run that will be compared.")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum validation micro-AUC gain that counts as an "
                             "improvement and resets patience. DEFAULTS TO 0.0, which "
                             "reproduces every run banked before this flag existed - "
                             "changing it changes the checkpoint SELECTION RULE, and two "
                             "arms selected under different rules are not comparable. "
                             "Set it for a whole run group or not at all. USE 0.0005, NOT "
                             "0.001 - SIMULATED on the real v22 seed-7 target-224 curve: "
                             "0.0 was still running at epoch 82 (best 70, val 0.9366); "
                             "0.0005 stops at 66 (best 63, val 0.9356); 0.001 stops at 28 "
                             "(best 27, val 0.9319). The 0.001 rule costs 0.0047 of "
                             "validation AUC, which is LARGER THAN THE ENTIRE RESIZER "
                             "EFFECT this sweep measures (max 0.0027 across four targets). "
                             "The stopping rule is a bigger lever than the variable under "
                             "study, which is exactly why it must not move between arms.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=0,
                        help="FIXED across arms. Changing it scores two arms on different "
                             "test films, which is not a comparison.")
    parser.add_argument("--aug-profile", default="v3")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--oversample", action="store_true",
                        help="draw rare-class films more often per epoch")
    parser.add_argument("--oversample-power", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--train-eval-films", type=int, default=1752,
                        help="Films from the TRAINING split, unaugmented, scored each "
                             "epoch exactly as validation is, to give a train-vs-val AUC "
                             "pair. Default is matched to the validation split size. 0 "
                             "disables it and reproduces runs made before this existed. "
                             "Purely observational - it changes nothing about training, "
                             "checkpoint selection or any reported number.")
    parser.add_argument("--run-group", default="vindr_multiclass")
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heatmap_size = arguments.target_size // 4

    stamp = datetime.now().strftime("%Y-%m-%d")
    leaf = (f"{arguments.variant}_fcos_vindr14_s{arguments.seed}"
            f"_i{arguments.input_size}_t{arguments.target_size}")
    output_dir = ROOT_DIR / "outputs" / "vindr" / f"{stamp}_{arguments.run_group}" / leaf
    output_dir.mkdir(parents=True, exist_ok=True)

    (train_loader, val_loader, test_loader, train_eval_loader,
     split_sizes) = build_loaders(arguments)

    model = NoduleDetector(variant=arguments.variant, head="fcos",
                           backbone=arguments.backbone,
                           target_size=arguments.target_size,
                           num_classes=NUM_VINDR_CLASSES).to(device)

    # Backbone on a reduced learning rate, matching the nodule track so the two sets of
    # results stay comparable on everything except the task.
    backbone_params = list(model.backbone_fpn.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimiser = torch.optim.AdamW([
        {"params": other_params, "lr": arguments.lr},
        {"params": backbone_params, "lr": arguments.lr * arguments.backbone_lr_factor},
    ], weight_decay=1e-4)

    print("=" * 78)
    print(f"VinDr 14-class sweep | {arguments.variant} | input {arguments.input_size} "
          f"-> target {arguments.target_size} | grid {heatmap_size}x{heatmap_size}")
    print(f"Run group: {arguments.run_group} | Seed: {arguments.seed} | "
          f"Split seed: {arguments.split_seed}")
    print(f"Films: {split_sizes['train']} train / {split_sizes['val']} val / "
          f"{split_sizes['test']} test")
    print(f"Output: {output_dir}")
    print("=" * 78)

    history = {"train_loss": [], "val_loss": [], "val_micro_auc": [], "val_micro_f1": [],
               "val_macro_auc": [], "val_threshold": [],
               # What the checkpoint was actually selected on. Recorded separately from
               # val_micro_auc so a later reader can see the rule, not just its result.
               "val_micro_auc_smoothed": [],
               # Held-IN diagnostics: training films scored like validation. The gap
               # between train_eval_micro_auc and val_micro_auc IS the overfitting measure.
               "train_eval_loss": [], "train_eval_micro_auc": []}
    best_auc, best_epoch, patience_counter = -1.0, -1, 0
    start = time.time()

    for epoch in range(arguments.epochs):
        train_loss = run_epoch(model, train_loader, optimiser, device, heatmap_size,
                               arguments.input_size, arguments.grad_accum)

        model.eval()
        scores, labels, val_loss = validate(model, val_loader, device, heatmap_size,
                                            arguments.input_size)
        val = evaluate(scores, labels)

        train_auc, train_eval_loss = float("nan"), float("nan")

        if train_eval_loader is not None:
            t_scores, t_labels, train_eval_loss = validate(
                model, train_eval_loader, device, heatmap_size, arguments.input_size)
            train_auc = evaluate(t_scores, t_labels)["micro_auc"]

        # SELECTION CRITERION: the trailing mean over `select_smoothing` epochs, so one
        # lucky epoch on the plateau cannot win the checkpoint. At window 1 this is the raw
        # per-epoch value and behaves exactly as every run banked before this existed.
        previous = history["val_micro_auc"][-(arguments.select_smoothing - 1):] \
            if arguments.select_smoothing > 1 else []
        selection_auc = float(np.mean(previous + [val["micro_auc"]]))

        history["val_micro_auc_smoothed"].append(selection_auc)

        improved = ""

        # min_delta governs only whether a gain counts as progress for PATIENCE, so a
        # genuine late improvement is still kept without buying another 15 epochs for it.
        if selection_auc > best_auc:
            if selection_auc > best_auc + arguments.min_delta:
                patience_counter = 0
            else:
                patience_counter += 1

            best_auc, best_epoch = selection_auc, epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            improved = " *" if patience_counter == 0 else " *(noise)"

        else:
            patience_counter += 1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_eval_loss"].append(train_eval_loss)
        history["train_eval_micro_auc"].append(train_auc)
        history["val_micro_auc"].append(val["micro_auc"])
        history["val_micro_f1"].append(val["micro_f1"])
        history["val_macro_auc"].append(val["macro_auc"])
        history["val_threshold"].append(val["threshold"])

        # Same shape as the nodule track's epoch line, so both logs read the same way
        # trAUC-vAUC is the overfitting gap: it widens as the model memorises films it
        # has already seen while the held-out curve stops moving.
        gap = "" if train_auc != train_auc else f" | gap: {train_auc - val['micro_auc']:+.4f}"

        print(f"Ep {epoch:02d}/{arguments.epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | trAUC: {train_auc:.4f} | "
              f"microAUC: {val['micro_auc']:.4f} | microF1: {val['micro_f1']:.4f} | "
              f"macroAUC: {val['macro_auc']:.4f}{gap}{improved}", flush=True)

        if patience_counter >= arguments.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"({arguments.patience} epochs without improvement)")
            break

    # Test, using the checkpoint selected on validation and the threshold fitted on it
    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    model.eval()

    val_scores, val_labels = image_scores(model, val_loader, device)
    fitted_threshold = evaluate(val_scores, val_labels)["threshold"]

    test_scores, test_labels = image_scores(model, test_loader, device)
    test = evaluate(test_scores, test_labels, threshold=fitted_threshold)

    criterion = ("val micro-AUC, {}-epoch trailing mean".format(arguments.select_smoothing)
                 if arguments.select_smoothing > 1 else "val micro-AUC")

    print(f"\nBest epoch {best_epoch} ({criterion} {best_auc:.4f})")
    print(f"Threshold {fitted_threshold:.3f} fitted on VALIDATION, applied to test\n")
    print(format_report(test))

    metrics = {
        "variant": arguments.variant, "head": "fcos", "dataset": "vindr14",
        "input_size": arguments.input_size, "target_size": arguments.target_size,
        "heatmap_size": heatmap_size, "backbone": arguments.backbone,
        "num_classes": NUM_VINDR_CLASSES, "seed": arguments.seed,
        "split_seed": arguments.split_seed, "run_group": arguments.run_group,
        "epochs_trained": len(history["train_loss"]), "best_epoch": best_epoch,
        "best_val_micro_auc": best_auc, "fitted_threshold": fitted_threshold,
        # THE SELECTION RULE, recorded with the run. Two arms chosen under different rules
        # are not comparable, and without this on disk there is no way to tell them apart
        # later - the same failure the NODE21 track guards against with `select_on`.
        "select_on": "val_micro_auc_smoothed{}".format(arguments.select_smoothing),
        "select_smoothing": arguments.select_smoothing,
        "min_delta": arguments.min_delta,
        "train_eval_films": arguments.train_eval_films,
        "oversample": arguments.oversample, "augment": arguments.augment,
        "batch_size": arguments.batch_size, "grad_accum": arguments.grad_accum,
        "lr": arguments.lr, "num_workers": arguments.num_workers,
        "split_sizes": split_sizes, "test_results": test, "history": history,
        "minutes": (time.time() - start) / 60.0,
    }

    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nTraining complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
