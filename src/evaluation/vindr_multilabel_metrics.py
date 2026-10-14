"""
Image-level multi-label scoring for the VinDr multi-class resolution sweep.

WHAT IS BEING MEASURED, AND WHY AT IMAGE LEVEL. The model is a multi-class FCOS detector
trained on VinDr's boxes - the boxes are kept because localised supervision is what gives
the resizer spatial gradient, and because it keeps the task continuous with the rest of
the thesis. But the question the sweep answers is "how does the resizer perform across
all fourteen pathologies", and a per-class FROC on a class with nineteen test films cannot
answer it. So the detector's output is reduced to one score per class per film, and scored
as multi-label classification.

HOW A DETECTION MAP BECOMES AN IMAGE SCORE. Max over the spatial grid of
sqrt(cls x centerness), per class. Max rather than mean because the question is "is this
finding present anywhere on the film", and a mean over a 56x56 grid would be dominated by
the background cells that make up almost all of it.

MICRO VS MACRO, AND WHY MICRO IS THE HEADLINE. Micro pools every (film, class) decision
into one contingency table before computing the score, so it is dominated by the common
classes and is backed by the full ~3,000-film test split. Macro averages the per-class
scores equally, so Pneumothorax at 19 test films carries the same weight as Aortic
enlargement at 613 and the result swings on a handful of films. Both are reported; micro
is the one to quote, macro is the one that exposes rare-class collapse.

THE TWO THIN CLASSES. Pneumothorax (19 test films) and Atelectasis (37) cannot support a
per-class number on their own - they are reported and explicitly flagged, the same way the
small size-bin is faded in the nodule figures. No split ratio fixes this and augmentation
does not either: augmented copies of one film are not independent films, so they raise the
row count without raising the sample size an interval is computed from.
"""

import numpy as np
import torch

from src.data.vindr import VINDR_CLASSES

# Below this many test films, a per-class score is reported but must not be quoted alone
THIN_CLASS_FILMS = 50


@torch.no_grad()
def image_scores(model, loader, device, autocast_dtype=torch.bfloat16):
    """
    Run the model over a loader and reduce each film to one score per class.

    Returns:
        scores (N, C) float32, labels (N, C) float32 multi-hot
    """
    all_scores, all_labels = [], []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images, target = batch[0], batch[1]
        else:
            images, target = batch["image"], batch

        images = images.to(device)

        with torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            output = model(images)

        # sqrt(cls x centerness) is the same confidence the detector decodes with, so the
        # image score is consistent with how boxes would have been ranked.
        confidence = (torch.sigmoid(output["classification"].float())
                      * torch.sigmoid(output["centerness"].float())).sqrt()

        # Max over the spatial grid: is this finding anywhere on the film
        per_class = confidence.amax(dim=(2, 3))                       # (B, C)

        all_scores.append(per_class.cpu().numpy())
        all_labels.append(target["image_labels"].numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def _auc(scores, labels):
    """ROC AUC via the rank identity. Returns nan when a class has one outcome only."""
    positives, negatives = labels == 1, labels == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks over ties, or tied scores bias the statistic
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)

    if counts.max() > 1:
        summed = np.zeros(len(unique))
        np.add.at(summed, inverse, ranks)
        ranks = (summed / counts)[inverse]

    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _counts_at(scores, labels, threshold):
    predicted = scores >= threshold

    return (int((predicted & (labels == 1)).sum()),
            int((predicted & (labels == 0)).sum()),
            int((~predicted & (labels == 1)).sum()))


def _f1(tp, fp, fn):
    if tp == 0:
        return 0.0

    precision, recall = tp / (tp + fp), tp / (tp + fn)

    return 2 * precision * recall / (precision + recall)


def evaluate(scores, labels, threshold=None):
    """
    Micro and macro multi-label scores, plus a per-class table.

    Args:
        scores: (N, C) model confidence per film per class
        labels: (N, C) multi-hot truth
        threshold: decision threshold. None fits the single threshold that maximises
                   micro-F1 on this set - which is a FITTED quantity and must therefore be
                   fitted on validation and applied to test, never fitted on test.

    Returns:
        dict of metrics, with "per_class" a list of dicts
    """
    if threshold is None:
        candidates = np.unique(np.round(scores, 3))
        threshold = max(candidates,
                        key=lambda t: _f1(*_counts_at(scores, labels, t)))

    tp, fp, fn = _counts_at(scores, labels, threshold)

    micro_f1 = _f1(tp, fp, fn)
    micro_auc = _auc(scores.ravel(), labels.ravel())

    per_class = []

    for index, name in enumerate(VINDR_CLASSES):
        class_tp, class_fp, class_fn = _counts_at(scores[:, index], labels[:, index], threshold)
        n_films = int((labels[:, index] == 1).sum())

        per_class.append({
            "name": name,
            "n_test_films": n_films,
            "auc": _auc(scores[:, index], labels[:, index]),
            "f1": _f1(class_tp, class_fp, class_fn),
            "recall": class_tp / max(class_tp + class_fn, 1),
            "precision": class_tp / max(class_tp + class_fp, 1),
            "thin": n_films < THIN_CLASS_FILMS,
        })

    macro_auc = float(np.nanmean([c["auc"] for c in per_class]))
    macro_f1 = float(np.mean([c["f1"] for c in per_class]))

    return {
        "threshold": float(threshold),
        "micro_f1": float(micro_f1),
        "micro_auc": float(micro_auc),
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "micro_precision": tp / max(tp + fp, 1),
        "micro_recall": tp / max(tp + fn, 1),
        "per_class": per_class,
    }


def format_report(metrics):
    """Per-class table with the thin classes marked, for the epoch log."""
    lines = ["  {:<22} {:>6} {:>7} {:>7} {:>8} {:>7}".format(
        "class", "films", "AUC", "F1", "recall", "prec")]

    for entry in sorted(metrics["per_class"], key=lambda c: -c["n_test_films"]):
        flag = "  <- thin, indicative only" if entry["thin"] else ""
        lines.append("  {:<22} {:>6} {:>7.3f} {:>7.3f} {:>8.3f} {:>7.3f}{}".format(
            entry["name"], entry["n_test_films"], entry["auc"], entry["f1"],
            entry["recall"], entry["precision"], flag))

    lines.append("")
    lines.append("  micro  F1 {:.4f}  AUC {:.4f}  precision {:.3f}  recall {:.3f}  @thr {:.3f}".format(
        metrics["micro_f1"], metrics["micro_auc"], metrics["micro_precision"],
        metrics["micro_recall"], metrics["threshold"]))
    lines.append("  macro  F1 {:.4f}  AUC {:.4f}   (macro weights the 19-film class like "
                 "the 613-film one)".format(metrics["macro_f1"], metrics["macro_auc"]))

    return "\n".join(lines)


def _demo():
    """Self-check: the metrics must behave correctly at the extremes."""
    rng = np.random.default_rng(0)
    n, c = 600, len(VINDR_CLASSES)

    labels = (rng.random((n, c)) < 0.2).astype(np.float32)

    # A perfect ranker scores exactly the truth
    perfect = evaluate(labels.copy(), labels)
    assert perfect["micro_auc"] > 0.999, perfect["micro_auc"]
    assert perfect["micro_f1"] > 0.999, perfect["micro_f1"]
    print("OK  perfect scores give micro F1 {:.3f}, AUC {:.3f}".format(
        perfect["micro_f1"], perfect["micro_auc"]))

    # Pure noise must land at chance, not above it
    noise = evaluate(rng.random((n, c)).astype(np.float32), labels)
    assert 0.45 < noise["micro_auc"] < 0.55, noise["micro_auc"]
    print("OK  random scores give micro AUC {:.3f} (chance)".format(noise["micro_auc"]))

    # Inverted scores must land BELOW chance - catches a sign error that random data hides
    inverted = evaluate(1.0 - labels, labels)
    assert inverted["micro_auc"] < 0.001, inverted["micro_auc"]
    print("OK  inverted scores give micro AUC {:.3f}".format(inverted["micro_auc"]))

    # THE REASON BOTH ARE REPORTED. A RARE class scored at chance while every other class
    # is scored well: micro barely notices, because the rare class is a tiny share of the
    # pooled decisions, while macro drops visibly. This is the realistic failure the sweep
    # must be able to see - Pneumothorax has 19 test films against Aortic enlargement's
    # 613, so a model that simply never predicts it would look fine on micro alone.
    #
    # Continuous scores, not binary: perfect 0/1 scores tie almost every pair, which makes
    # the AUC rank statistic uninformative and made an earlier version of this check
    # assert the wrong thing.
    rare_labels = (rng.random((n, c)) < 0.2).astype(np.float32)
    rare_labels[:, 12] = 0.0
    rare_labels[rng.choice(n, size=19, replace=False), 12] = 1.0        # 19 films, as in test

    good = (rare_labels * rng.uniform(0.6, 1.0, (n, c))
            + (1 - rare_labels) * rng.uniform(0.0, 0.4, (n, c)))
    good[:, 12] = rng.random(n)                                         # chance on the rare class

    mixed = evaluate(good.astype(np.float32), rare_labels)
    rare_auc = mixed["per_class"][12]["auc"]

    assert 0.3 < rare_auc < 0.7, f"rare class should score near chance, got {rare_auc:.3f}"
    assert mixed["macro_auc"] < mixed["micro_auc"], \
        "macro must punish a chance-level rare class harder than micro"
    assert mixed["micro_auc"] > 0.95, "micro should barely move - that is the point"
    assert mixed["per_class"][12]["thin"], "a 19-film class must be flagged thin"
    print("OK  rare class at chance (AUC {:.3f}): micro {:.3f} barely moves, "
          "macro {:.3f} exposes it".format(rare_auc, mixed["micro_auc"], mixed["macro_auc"]))

    # A class with no positives must not crash or silently score 0
    empty = labels.copy()
    empty[:, 3] = 0.0
    assert np.isnan(evaluate(rng.random((n, c)).astype(np.float32), empty)["per_class"][3]["auc"])
    print("OK  a class with no positive films returns nan rather than a fake score")


if __name__ == "__main__":
    _demo()
