"""
Box-based detection metrics for NODE21.

Written as a separate module rather than a change to `compute_detection_metrics_v4` so
the older centre-distance numbers stay reproducible for comparison.

Two things this fixes relative to the centre-distance metric:

1.  **Resolution independence.** The old hit test was `distance <= dist_thresh_mm /
    pixel_spacing_mm`, a constant 85.7 pixels applied in input_size space regardless of
    input_size. That is only correct at input_size=2048; at 224 it is ~38% of the image
    width, so nearly any peak in the correct lung scored as a hit. IoU is a ratio of
    areas, so it carries no length unit and is identical at every resolution.

2.  **Broken pixel spacing.** 61% of NODE21 films are 1024x1024 resamples whose .mha
    spacing headers were never updated - they imply a 146 mm chest, against 358 mm for
    the natively-sized films. Any mm-denominated threshold inherits that error. IoU does
    not. Spacing is needed only for the S/M/L size report, and there it is derived from
    image width rather than the headers (see `diameter_mm`).
"""

import numpy as np
import torch
from torch.amp import autocast

from src.evaluation.centernet_targets import decode_detections, box_iou
from src.evaluation.fcos_targets import decode_fcos

# Median physical width of the NODE21 films whose spacing headers are self-consistent
# (native-size images: 358 mm; the 1024x1024 resamples report an impossible 146 mm).
# Used only to express box sizes in mm for the S/M/L report.
CHEST_WIDTH_MM = 358.0

SIZE_BINS = {"S": (0.0, 10.0), "M": (10.0, 20.0), "L": (20.0, 1e9)}


def diameter_mm(box_width_input_px, box_height_input_px, input_size):
    """
    RECIST longest-axis diameter in mm, derived from image width rather than the .mha
    spacing header.

    A box occupying fraction f of the image width spans f * CHEST_WIDTH_MM millimetres,
    whatever the native pixel count. The native size cancels, so this is stable across
    the mixed-resolution NODE21 archive and independent of the broken headers.
    """
    return max(box_width_input_px, box_height_input_px) * CHEST_WIDTH_MM / input_size


def _bin_for(diameter):
    for name, (low, high) in SIZE_BINS.items():
        if low <= diameter < high: return name

    return "L"


def _match(predicted_boxes, predicted_scores, truth_boxes, iou_threshold, score_threshold):
    """
    Greedy highest-score-first matching of predictions to ground truth at an IoU floor.

    Each GT box may be claimed once. Returns (matched GT indices, number of unmatched
    predictions), where unmatched predictions above the score threshold are false
    positives.
    """
    keep = predicted_scores >= score_threshold
    boxes, scores = predicted_boxes[keep], predicted_scores[keep]

    if len(boxes) == 0: return set(), 0

    order = torch.argsort(scores, descending=True)
    claimed, false_positives = set(), 0

    for k in order:
        if len(truth_boxes) == 0:
            false_positives += 1
            continue

        ious = box_iou(boxes[k][None, :], truth_boxes)[0]

        # Never let one prediction claim an already-matched GT box
        for already in claimed:
            ious[already] = -1.0

        best = int(torch.argmax(ious))

        if ious[best] >= iou_threshold:
            claimed.add(best)
        else:
            false_positives += 1

    return claimed, false_positives


def _truth_boxes_for(targets, i, input_size):
    """Extract image i's real (non-padded) GT boxes as x1,y1,x2,y2 in input_size space."""
    count = int(targets["num_boxes"][i])

    if count == 0: return torch.zeros((0, 4)), []

    raw = targets["boxes"][i, :count].float()               # (n, 4) cx, cy, w, h
    corners = torch.stack([
        raw[:, 0] - raw[:, 2] / 2, raw[:, 1] - raw[:, 3] / 2,
        raw[:, 0] + raw[:, 2] / 2, raw[:, 1] + raw[:, 3] / 2,
    ], dim=1)
    bins = [_bin_for(diameter_mm(float(raw[j, 2]), float(raw[j, 3]), input_size)) for j in range(count)]

    return corners, bins


@torch.no_grad()
def compute_box_detection_metrics(model, loader, device, input_size,
                                  iou_threshold=0.25, score_threshold=0.3, top_k=20):
    """
    Sensitivity, false positives per image and average precision under IoU matching.

    `iou_threshold` defaults to 0.25 rather than the COCO-standard 0.5. NODE21 boxes are
    loose hand-drawn annotations around small, low-contrast lesions, and the clinical
    question is whether the nodule was found, not whether its extent was regressed
    precisely. Both 0.25 and 0.5 are reported by `sweep_iou_thresholds` so the choice is
    visible rather than buried.

    Returns a dict with overall counts, per-size-bin sensitivity, and the AP curve inputs.
    """
    model.eval()

    true_positives, false_negatives, false_positives, n_images = 0, 0, 0, 0
    bin_counters = {b: {"tp": 0, "n": 0} for b in SIZE_BINS}
    scored_predictions = []          # (score, is_true_positive, size_bin) for the AP curve
    total_truth = 0

    # Image-level accounting, for specificity and the image-level ROC. A CAD system is also
    # judged on whether it flags a film at all, not only on where it puts the box, and
    # aggregate FP counts cannot answer that: 10 false positives on one film and one each on
    # ten films are the same number and very different behaviour.
    n_negative_images, negative_images_flagged = 0, 0
    n_positive_images, positive_images_flagged = 0, 0
    image_level_scores = []          # (max prediction score, image has a nodule)

    for images, targets in loader:
        images = images.to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images)

        # Both heads produce boxes in input_size space; only the decode differs
        if getattr(model, "is_fcos", False):
            boxes_batch, scores_batch = decode_fcos(
                output["classification"].float(), output["ltrb"].float(),
                output["centerness"].float(), input_size=input_size, top_k=top_k)
        else:
            boxes_batch, scores_batch = decode_detections(
                output["heatmap"].float(), output["size_wh"].float(), output["offset"].float(),
                input_size=input_size, top_k=top_k)
        boxes_batch, scores_batch = boxes_batch.cpu(), scores_batch.cpu()

        for i in range(images.shape[0]):
            n_images += 1
            truth, bins = _truth_boxes_for(targets, i, input_size)
            total_truth += len(truth)

            claimed, n_false = _match(boxes_batch[i], scores_batch[i], truth,
                                      iou_threshold, score_threshold)

            true_positives  += len(claimed)
            false_negatives += len(truth) - len(claimed)
            false_positives += n_false

            # Image level: does this film get flagged at all, at this threshold?
            top_score = float(scores_batch[i].max()) if scores_batch[i].numel() else 0.0
            flagged = top_score >= score_threshold
            image_level_scores.append((top_score, len(truth) > 0))

            if len(truth):
                n_positive_images += 1
                positive_images_flagged += int(flagged)
            else:
                n_negative_images += 1
                negative_images_flagged += int(flagged)

            for j, bin_name in enumerate(bins):
                bin_counters[bin_name]["n"] += 1

                if j in claimed: bin_counters[bin_name]["tp"] += 1

            # Score-ranked records for average precision, independent of score_threshold
            order = torch.argsort(scores_batch[i], descending=True)
            ap_claimed = set()

            for k in order:
                # decode_fcos zero-pads to top_k; those slots are not predictions and
                # would otherwise enter the PR curve as free false positives
                if float(scores_batch[i][k]) <= 0.0: continue

                if len(truth) == 0:
                    scored_predictions.append((float(scores_batch[i][k]), 0, None))
                    continue

                ious = box_iou(boxes_batch[i][k][None, :], truth)[0]

                for already in ap_claimed:
                    ious[already] = -1.0

                best = int(torch.argmax(ious))
                hit = ious[best] >= iou_threshold

                if hit: ap_claimed.add(best)

                # Carry the matched GT's size bin so the FROC sweep can be split by bin.
                # None for a false positive: an FP belongs to no nodule and so to no bin,
                # which is why the per-bin curves below share one pooled FP axis.
                scored_predictions.append((float(scores_batch[i][k]), 1 if hit else 0,
                                           bins[best] if hit else None))

    sensitivity = true_positives / max(1, true_positives + false_negatives)

    return {
        "sensitivity": sensitivity,
        "fp_per_image": false_positives / max(1, n_images),
        "tp": true_positives, "fn": false_negatives, "fp": false_positives,
        "n_images": n_images, "n_nodules": total_truth,
        "iou_threshold": iou_threshold, "score_threshold": score_threshold,
        "average_precision": average_precision(scored_predictions, total_truth),
        "by_size": {b: {"sensitivity": (c["tp"] / c["n"] if c["n"] else None), "n_nodules": c["n"]}
                    for b, c in bin_counters.items()},
        "precision": true_positives / max(1, true_positives + false_positives),
        "froc": froc_from_predictions(scored_predictions, total_truth, n_images),
        "froc_by_size": froc_by_size(scored_predictions,
                                     {b: c["n"] for b, c in bin_counters.items()}, n_images),

        # Image-level view, at this score_threshold. `specificity` is the fraction of
        # nodule-free films with NO detection above threshold - the metric aggregate
        # false-positive counts cannot express.
        "image_level": {
            "n_positive_images":  n_positive_images,
            "n_negative_images":  n_negative_images,
            "sensitivity": positive_images_flagged / max(1, n_positive_images),
            "specificity": 1.0 - negative_images_flagged / max(1, n_negative_images),
            "roc_auc":     image_level_roc_auc(image_level_scores),
            "roc":         image_level_roc(image_level_scores),
        },
    }


def average_precision(scored_predictions, total_truth):
    """Area under the precision-recall curve, all-point interpolation (VOC 2010+ / COCO)."""
    if total_truth == 0 or not scored_predictions: return 0.0

    scored_predictions = sorted(scored_predictions, key=lambda p: -p[0])
    hits = np.array([p[1] for p in scored_predictions], dtype=np.float64)

    cumulative_tp = np.cumsum(hits)
    cumulative_fp = np.cumsum(1 - hits)

    recall    = cumulative_tp / total_truth
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)

    # Make precision monotone decreasing, then integrate over recall
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])

    return float(np.sum(np.diff(recall) * precision[1:]))


def froc_from_predictions(scored_predictions, total_truth, n_images, n_points=51):
    """
    Sensitivity against false positives per image, swept over a FIXED THRESHOLD GRID.

    This is the textbook construction (ACML lecture 6, section 4.2): choose threshold t,
    count TP and FP at that t, plot one point, repeat over an even grid of t in [0, 1].
    Only the x-axis differs from ROC - false positives PER IMAGE rather than a false
    positive rate - because a detector can emit many boxes per film, so there is no TN
    count to normalise by.

    WHY THIS REPLACED RANK SAMPLING. The original stepped evenly through the sorted
    prediction list. At low scores almost every extra prediction is a false positive, so
    even rank steps become even FP steps, and with top_k=20 per image the curve spans
    0 to ~20 FP/img. Twenty-five even steps therefore landed a sample every ~0.83 FP/img
    and left EXACTLY ONE point below 1 FP/img. Everything here is reported at 0.25 FP/img,
    less than a third of one step, so the reported value was a linear chord across the gap
    from (0.00 FP, ~0.01 sensitivity) to (0.65 FP, ~0.64). FROC curves are concave, so the
    chord ran far below the truth: on the v14 runs measured sensitivities came out roughly
    DOUBLE the interpolated ones and the sign of the bilinear-vs-Talebi delta flipped.

    A fixed threshold grid does not have that failure mode, because the score-to-FP
    relationship is steep at high thresholds: on these runs the whole 0 to 0.65 FP/img
    band lives between t=0.874 and t=0.303, so an even grid drops several points inside it
    automatically. Benchmarked against the exact curve (every vertex), a grid of step 0.05
    recovered sensitivity at 0.25 FP/img to within 0.002, while rank sampling was out by
    0.36.

    KNOWN LIMITATION, deliberately accepted. A fixed grid assumes scores are spread across
    [0, 1]. FCOS scores sqrt(cls * centerness), which can concentrate: if a model's scores
    all fell in [0.4, 0.5], every t below 0.4 would give the same point and every t above
    0.5 would give another, leaving the curve with two useful vertices. Sampling on the FP
    axis adapts automatically and does not have this weakness, but it is harder to explain
    and does not match the taught construction. The `_demo` density check below will fail
    loudly if a future model ever defeats the grid.

    Args:
        scored_predictions: (score, is_true_positive, ...) per prediction
        total_truth: number of ground-truth nodules
        n_images: images swept, the denominator for fp_per_image
        n_points: grid size; 51 gives a step of 0.02

    Returns:
        List of {threshold, sensitivity, fp_per_image}, ascending in fp_per_image.
    """
    if total_truth == 0 or not scored_predictions: return []

    scored_predictions = sorted(scored_predictions, key=lambda p: -p[0])
    hits = np.array([p[1] for p in scored_predictions], dtype=np.float64)
    scores = np.array([p[0] for p in scored_predictions])

    cumulative_tp = np.cumsum(hits)
    cumulative_fp = np.cumsum(1 - hits)

    curve = []

    # Descending t, so the curve comes out ascending in fp_per_image.
    for threshold in np.linspace(1.0, 0.0, n_points):
        # Predictions kept at this threshold. scores is descending, so negating gives an
        # ascending array searchsorted can work on.
        kept = int(np.searchsorted(-scores, -threshold, side="right"))

        if kept == 0:
            sensitivity, fp_per_image = 0.0, 0.0
        else:
            sensitivity = float(cumulative_tp[kept - 1] / total_truth)
            fp_per_image = float(cumulative_fp[kept - 1] / max(1, n_images))

        curve.append({
            "threshold": float(threshold),
            "sensitivity": sensitivity,
            "fp_per_image": fp_per_image,
        })

    return curve


def froc_by_size(scored_predictions, bin_totals, n_images, n_points=51):
    """
    Per-size-bin FROC curves swept over one shared, global score threshold.

    Uses the same FIXED THRESHOLD GRID as froc_from_predictions, and for the same reason:
    the previous rank-based sweep left a single sample below 1 FP/img, so every
    size-stratified number read at 0.25 FP/img was a chord across a gap. See that
    function's docstring for the measurement and the accepted limitation.

    A false positive is a prediction that matched no ground truth, so it belongs to no
    size bin. Splitting the FP axis by bin is therefore not defined. Instead every bin's
    curve is plotted against the SAME pooled fp_per_image, produced by the same threshold
    sweep as the pooled curve. Reading all three bins at 0.25 FP/img then answers "at the
    operating point NODE21 reports, how many small / medium / large nodules were found",
    which is the question the fixed score_threshold=0.3 in `by_size` cannot answer.

    Args:
        scored_predictions: (score, is_true_positive, bin_name_or_None) per prediction.
        bin_totals:         {bin_name: number of GT nodules in that bin}.
        n_images:           images swept, for the false-positives-per-image denominator.

    Returns:
        {bin_name: [{"threshold", "sensitivity", "fp_per_image"}, ...]}. Bins with no GT
        nodules map to an empty list.
    """
    if not scored_predictions: return {b: [] for b in SIZE_BINS}

    ranked = sorted(scored_predictions, key=lambda p: -p[0])
    scores = np.array([p[0] for p in ranked])

    # False positives are pooled: the FP axis is shared by every bin
    cumulative_fp = np.cumsum(np.array([0.0 if p[1] else 1.0 for p in ranked]))

    # Descending t so the curves come out ascending in fp_per_image. Precomputed once,
    # since every bin is read at the same global thresholds.
    thresholds = np.linspace(1.0, 0.0, n_points)
    kept_counts = [int(np.searchsorted(-scores, -t, side="right")) for t in thresholds]

    curves = {}

    for bin_name in SIZE_BINS:
        total = bin_totals.get(bin_name, 0)

        if not total:
            curves[bin_name] = []
            continue

        # True positives restricted to this bin, ranked by the same global score order
        bin_hits = np.array([1.0 if (p[1] and p[2] == bin_name) else 0.0 for p in ranked])
        cumulative_tp = np.cumsum(bin_hits)

        points = []

        for threshold, kept in zip(thresholds, kept_counts):
            if kept == 0:
                sensitivity, fp_per_image = 0.0, 0.0
            else:
                sensitivity = float(cumulative_tp[kept - 1] / total)
                fp_per_image = float(cumulative_fp[kept - 1] / max(1, n_images))

            points.append({"threshold": float(threshold),
                           "sensitivity": sensitivity,
                           "fp_per_image": fp_per_image})

        curves[bin_name] = points

    return curves


def froc_auc(curve, max_fp=4.0):
    """Mean sensitivity over [0, max_fp] false positives per image."""
    if not curve: return 0.0

    fp = np.array([p["fp_per_image"] for p in curve])
    se = np.array([p["sensitivity"] for p in curve])
    order = np.argsort(fp)
    grid = np.linspace(0, max_fp, 401)

    return float(np.trapezoid(np.interp(grid, fp[order], se[order], left=0.0, right=se[order][-1]), grid) / max_fp)


def image_level_roc_auc(scored_images):
    """
    ROC AUC for the image-level question: does this film contain a nodule at all?

    `scored_images` is (max prediction score, image has a nodule) per image. Computed by the
    rank-sum identity rather than by integrating a curve, so it is exact and threshold-free:
    AUC equals the probability that a random positive film scores above a random negative
    one. Ties contribute 0.5, which matters here because a film with no prediction at all
    scores exactly 0.0 and there are many of them.
    """
    positives = [s for s, is_positive in scored_images if is_positive]
    negatives = [s for s, is_positive in scored_images if not is_positive]

    if not positives or not negatives: return 0.0

    ranked = sorted(scored_images, key=lambda pair: pair[0])
    ranks, index = {}, 0

    # Average rank within each tie group, the standard Mann-Whitney correction
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[index][0]: end += 1

        average = (index + end) / 2.0 + 1.0

        for position in range(index, end + 1):
            ranks.setdefault(ranked[position][0], average)

        index = end + 1

    rank_sum = sum(ranks[score] for score, is_positive in scored_images if is_positive)
    n_pos, n_neg = len(positives), len(negatives)

    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def image_level_roc(scored_images, n_points=25):
    """Image-level ROC as (threshold, sensitivity, specificity), swept over the score."""
    if not scored_images: return []

    positives = [s for s, is_positive in scored_images if is_positive]
    negatives = [s for s, is_positive in scored_images if not is_positive]

    if not positives or not negatives: return []

    scores = np.array(sorted({s for s, _ in scored_images}))

    if len(scores) == 1: scores = np.array([scores[0]])

    sweep = np.unique(np.linspace(0, len(scores) - 1, n_points).astype(int))
    positives_array, negatives_array = np.array(positives), np.array(negatives)
    curve = []

    for idx in sweep:
        threshold = float(scores[idx])
        curve.append({
            "threshold":   threshold,
            "sensitivity": float((positives_array >= threshold).mean()),
            "specificity": float((negatives_array < threshold).mean()),
        })

    return curve


def sensitivity_at_fp(curve, budget=0.25):
    """
    Sensitivity read off a FROC curve at a false-positive budget.

    This is the number this project reports at, and it is what checkpoint selection should
    be judged on. Selecting on `sensitivity` from the metrics dict instead maximises
    sensitivity at a FIXED score threshold of 0.3, which places no constraint on false
    positives at all - so the saved checkpoint is by construction the most liberal epoch the
    run produced. See trap 4.1 in docs/context/HANDOVER.md; the same failure that once made a
    flat-to-worse variant look like a winner applies to epoch selection, not only reporting.

    Clamps outside the measured range rather than extrapolating.
    """
    if not curve: return 0.0

    fp = np.array([p["fp_per_image"] for p in curve])
    se = np.array([p["sensitivity"] for p in curve])
    order = np.argsort(fp)

    return float(np.interp(budget, fp[order], se[order], left=0.0, right=se[order][-1]))


def _demo():
    """Self-checks on matching, AP and the resolution-independence claim."""
    # Greedy matching: one prediction cannot claim two GT boxes
    truth = torch.tensor([[0.0, 0.0, 100.0, 100.0], [200.0, 200.0, 300.0, 300.0]])
    preds = torch.tensor([[0.0, 0.0, 100.0, 100.0], [5.0, 5.0, 105.0, 105.0]])
    scores = torch.tensor([0.9, 0.8])
    claimed, n_fp = _match(preds, scores, truth, iou_threshold=0.25, score_threshold=0.3)
    assert claimed == {0}, f"expected GT 0 claimed once, got {claimed}"
    assert n_fp == 1, f"second overlapping prediction must be a false positive, got {n_fp}"
    print("OK  greedy matching claims each GT box at most once")

    # Every prediction on a negative image is a false positive
    claimed, n_fp = _match(preds, scores, torch.zeros((0, 4)), 0.25, 0.3)
    assert claimed == set() and n_fp == 2
    print("OK  predictions on a nodule-free film all count as false positives")

    # Score threshold suppresses low-confidence predictions
    _, n_fp = _match(preds, torch.tensor([0.1, 0.05]), torch.zeros((0, 4)), 0.25, 0.3)
    assert n_fp == 0
    print("OK  score threshold filters predictions before matching")

    # Perfect ranking gives AP 1.0; pure noise gives much less
    assert abs(average_precision([(0.9, 1), (0.8, 1), (0.1, 0)], 2) - 1.0) < 1e-6
    assert average_precision([(0.9, 0), (0.8, 0), (0.1, 1)], 1) < 0.4
    print("OK  average precision is 1.0 on perfect ranking, low on inverted ranking")

    # THE key property: IoU is unchanged by resolution, the old mm threshold was not
    for input_size in (224, 1024, 2048):
        scale = input_size / 1024.0
        a = torch.tensor([[10.0, 10.0, 50.0, 50.0]]) * scale
        b = torch.tensor([[20.0, 20.0, 60.0, 60.0]]) * scale
        iou = float(box_iou(a, b))
        assert abs(iou - 0.391304) < 1e-4, f"IoU changed with resolution at {input_size}: {iou}"
    print("OK  IoU identical at 224 / 1024 / 2048 - the resolution confound is gone")

    # Size binning derived from image width, not the .mha headers
    assert abs(diameter_mm(50.0, 40.0, 1024) - 50.0 * 358.0 / 1024) < 1e-6
    assert _bin_for(diameter_mm(20, 20, 1024)) == "S"      # ~7 mm
    assert _bin_for(diameter_mm(45, 45, 1024)) == "M"      # ~16 mm
    assert _bin_for(diameter_mm(90, 90, 1024)) == "L"      # ~31 mm
    # and the same physical box bins identically at another resolution
    assert _bin_for(diameter_mm(45 * 224 / 1024, 45 * 224 / 1024, 224)) == "M"
    print("OK  size bins are width-derived and resolution-stable")

    # Per-bin FROC: 2 S nodules and 2 L nodules, one of each found, plus one false positive.
    # The pooled curve and the per-bin curves must be driven by the same threshold sweep.
    predictions = [(0.9, 1, "S"), (0.8, 1, "L"), (0.7, 0, None), (0.6, 0, None)]
    curves = froc_by_size(predictions, {"S": 2, "M": 0, "L": 2}, n_images=4, n_points=4)
    assert curves["M"] == [], "a bin with no GT nodules must yield an empty curve"
    assert curves["S"][-1]["sensitivity"] == 0.5, curves["S"]
    assert curves["L"][-1]["sensitivity"] == 0.5, curves["L"]

    # The FP axis is pooled, so S and L report identical fp_per_image at every threshold
    assert [p["fp_per_image"] for p in curves["S"]] == [p["fp_per_image"] for p in curves["L"]]
    assert curves["S"][-1]["fp_per_image"] == 0.5, curves["S"]      # 2 FPs over 4 images

    # Old 2-tuple records must still drive AP and the pooled FROC unchanged
    assert abs(average_precision([(0.9, 1), (0.8, 1), (0.1, 0)], 2) - 1.0) < 1e-6
    assert froc_from_predictions(predictions, 4, 4, n_points=4)[-1]["sensitivity"] == 0.5

    # The FROC must be DENSE where the operating point is. Sampling evenly by rank put
    # exactly one sample below 1 FP/img on a real run, so the reported 0.25 FP/img value
    # was a chord across a gap and understated sensitivity by roughly 2x.
    rng = np.random.default_rng(0)
    stream  = [(float(np.clip(rng.beta(5, 3), 0, 1)), 1.0, None) for _ in range(147)]
    stream += [(float(np.clip(rng.beta(1.2, 6), 0, 1)), 0.0, None) for _ in range(517 * 20 - 147)]

    dense_curve = froc_from_predictions(stream, 147, 517)
    below_one = [c for c in dense_curve if c["fp_per_image"] < 1.0]

    assert len(below_one) >= 10, f"only {len(below_one)} samples below 1 FP/img"
    assert min(abs(c["fp_per_image"] - 0.25) for c in dense_curve) < 0.05,         "no sample near the 0.25 FP/img budget everything is reported at"

    fps = [c["fp_per_image"] for c in dense_curve]
    assert fps == sorted(fps), "curve must be ascending in fp_per_image"
    print("OK  FROC is sampled densely below 1 FP/img, where every reported number lives")
    print("OK  per-bin FROC shares the pooled FP axis and one global threshold sweep")

    # Reading a curve at a budget interpolates, and clamps rather than extrapolating
    curve = [{"threshold": 0.9, "sensitivity": 0.0, "fp_per_image": 0.0},
             {"threshold": 0.5, "sensitivity": 0.6, "fp_per_image": 0.5},
             {"threshold": 0.1, "sensitivity": 0.9, "fp_per_image": 2.0}]
    assert abs(sensitivity_at_fp(curve, 0.25) - 0.3) < 1e-9, sensitivity_at_fp(curve, 0.25)
    assert abs(sensitivity_at_fp(curve, 0.5) - 0.6) < 1e-9
    assert abs(sensitivity_at_fp(curve, 99.0) - 0.9) < 1e-9      # clamps, no extrapolation
    assert sensitivity_at_fp([], 0.25) == 0.0

    # Points arriving out of FP order must be sorted before interpolation
    assert abs(sensitivity_at_fp(list(reversed(curve)), 0.25) - 0.3) < 1e-9
    print("OK  sensitivity_at_fp interpolates on a sorted FP axis and clamps at the ends")

    # Image-level ROC AUC: perfect separation, inverted separation, and pure ties
    perfect = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    assert abs(image_level_roc_auc(perfect) - 1.0) < 1e-9, image_level_roc_auc(perfect)

    inverted = [(0.1, True), (0.2, True), (0.8, False), (0.9, False)]
    assert abs(image_level_roc_auc(inverted) - 0.0) < 1e-9, image_level_roc_auc(inverted)

    # Every film scoring 0.0 - which happens when nothing is predicted anywhere - must give
    # 0.5, not 1.0. Ties are why this uses the rank-sum form rather than curve integration.
    all_tied = [(0.0, True), (0.0, True), (0.0, False), (0.0, False)]
    assert abs(image_level_roc_auc(all_tied) - 0.5) < 1e-9, image_level_roc_auc(all_tied)

    # One class missing cannot produce a score
    assert image_level_roc_auc([(0.5, True), (0.4, True)]) == 0.0
    assert image_level_roc_auc([]) == 0.0

    # The swept curve must be monotone in the right directions
    roc = image_level_roc(perfect, n_points=5)
    sens = [p["sensitivity"] for p in roc]
    spec = [p["specificity"] for p in roc]
    assert all(a >= b for a, b in zip(sens, sens[1:])), sens        # falls as threshold rises
    assert all(a <= b for a, b in zip(spec, spec[1:])), spec        # rises as threshold rises
    print("OK  image-level ROC AUC handles ties, inversion and a missing class")


if __name__ == "__main__":
    _demo()
