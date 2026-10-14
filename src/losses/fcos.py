"""
Loss for the single-level FCOS detector.

L = focal(classification) + GIoU(ltrb, weighted by centerness) + BCE(centerness)

GIoU rather than the L1 that CenterNet uses on width/height: L1 on edge distances treats
a 2-cell error the same on a 2-cell nodule as on a 30-cell one, which on NODE21 - where
boxes span 2.5 to 6 cells - would let large lesions dominate the gradient. GIoU is
scale-free, which matters more here than in a natural-image dataset.

Regression is weighted by the centerness TARGET, the reference FCOS choice: cells near a
box edge carry less weight than cells sitting on the object.
"""

import torch
import torch.nn.functional as F

from src.core.registry import LOSS_REGISTRY


# Grid side the normalisation floor is anchored to: target 224 over a stride-4 head.
# t224 is this project's resolution of interest, so anchoring here makes the floor below
# EXACTLY 1.0 there - a bit-for-bit no-op against every run banked before 2026-09-25 at
# that target - and a correction that grows with the grid above it.
REFERENCE_GRID_SIDE = 56


def _negative_batch_floor(grid_side):
    """
    Denominator floor for a batch holding no positive cells, in units of the t224 grid.

    WHY THIS EXISTS. The reference RetinaNet/FCOS normaliser is `num_positives`, floored
    at 1 so a batch with no objects cannot divide by zero. That convention assumes every
    training image contains an object, which is true of COCO and false here: with
    vindr_negative_ratio 1.0 and batch_size 2, over half of all batches are entirely
    nodule-free. On those batches the numerator sums over every cell while the denominator
    stays pinned at 1, so their gradient scales with GRID AREA while the nodule-bearing
    batches stay correctly normalised.

    MEASURED at identical per-cell error, batch of 2 clean films:

        target   grid     clean-batch loss    batch-with-nodule loss
        224      56x56        8.48                  1.35
        512      128x128     44.32                  1.10
        1024     256x256    177.30                  1.20

    So a nodule-free batch pulled 21x harder at target 1024 than at 224 purely from the
    normaliser. That suppresses detections progressively with resolution, and at 1024 it
    was strong enough to stop the bilinear arm training at all (val_loss 70-325 against
    the learned arm's 0.5-2.0 on the same setting).

    Scaling the floor with grid area removes the resolution dependence while leaving the
    per-batch-size behaviour untouched, so a larger batch still contributes proportionally
    more as it always did.
    """
    return (grid_side / REFERENCE_GRID_SIDE) ** 2


def _focal(prediction, target, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss (Lin et al. 2017), normalised by the positive count."""
    probability = torch.sigmoid(prediction.float())
    ce = F.binary_cross_entropy_with_logits(prediction.float(), target.float(), reduction="none")

    p_t = probability * target + (1 - probability) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce

    # Floor the positive count so nodule-free batches do not scale with the grid
    floor = _negative_batch_floor(target.shape[-1])

    return loss.sum() / target.sum().clamp(min=floor)


def _giou(predicted_ltrb, target_ltrb):
    """
    GIoU between two axis-aligned boxes expressed as distances to their shared cell centre.

    Both are anchored at the same point, so intersection and enclosure follow directly
    from element-wise min/max of the edge distances.
    """
    pl, pt, pr, pb = predicted_ltrb.unbind(dim=1)
    tl, tt, tr, tb = target_ltrb.unbind(dim=1)

    predicted_area = (pl + pr) * (pt + pb)
    target_area    = (tl + tr) * (tt + tb)

    intersect_w = torch.min(pl, tl) + torch.min(pr, tr)
    intersect_h = torch.min(pt, tt) + torch.min(pb, tb)
    intersection = intersect_w.clamp(min=0) * intersect_h.clamp(min=0)

    union = predicted_area + target_area - intersection
    iou = intersection / union.clamp(min=1e-7)

    enclose_w = torch.max(pl, tl) + torch.max(pr, tr)
    enclose_h = torch.max(pt, tt) + torch.max(pb, tb)
    enclose = (enclose_w * enclose_h).clamp(min=1e-7)

    return 1.0 - (iou - (enclose - union) / enclose)


@LOSS_REGISTRY.register("fcos")
def fcos_loss(prediction, target, lambda_regression=1.0, lambda_centerness=1.0):
    """
    Args:
        prediction: dict from FCOSHead - classification (B,1,H,W) logits, ltrb (B,4,H,W)
                    already positive, centerness (B,1,H,W) logits
        target: dict from make_fcos_targets

    Returns:
        (total loss, dict of components detached for logging)
    """
    classification_loss = _focal(prediction["classification"], target["classification"])

    positive = target["positive_mask"].view(-1) > 0
    n_positive = int(positive.sum())

    # A batch of nodule-free films has no box to regress; classification still applies
    if n_positive == 0:
        return classification_loss, {"cls": float(classification_loss.detach()),
                                     "giou": 0.0, "centerness": 0.0, "n_pos": 0}

    flatten = lambda tensor, channels: tensor.permute(0, 2, 3, 1).reshape(-1, channels)
    predicted_ltrb    = flatten(prediction["ltrb"].float(), 4)[positive]
    target_ltrb       = flatten(target["ltrb"], 4)[positive]
    centerness_target = flatten(target["centerness"], 1)[positive].squeeze(1)

    giou = _giou(predicted_ltrb, target_ltrb)
    regression_loss = (giou * centerness_target).sum() / centerness_target.sum().clamp(min=1e-6)

    predicted_centerness = flatten(prediction["centerness"].float(), 1)[positive].squeeze(1)
    centerness_loss = F.binary_cross_entropy_with_logits(predicted_centerness, centerness_target)

    total = classification_loss + lambda_regression * regression_loss + lambda_centerness * centerness_loss

    return total, {"cls": float(classification_loss.detach()),
                   "giou": float(regression_loss.detach()),
                   "centerness": float(centerness_loss.detach()),
                   "n_pos": n_positive}


def _demo():
    """Self-check: perfect predictions minimise each term; degraded ones raise it."""
    from src.evaluation.fcos_targets import make_fcos_targets

    input_size, heatmap_size = 1024, 128
    boxes = torch.zeros((2, 4, 4))
    boxes[0, 0] = torch.tensor([300.0, 400.0, 80.0, 60.0])
    boxes[1, 0] = torch.tensor([512.0, 512.0, 160.0, 100.0])
    counts = torch.tensor([1, 1])

    t = make_fcos_targets({"boxes": boxes, "num_boxes": counts}, 2, heatmap_size, input_size, "cpu")

    perfect = {
        "classification": torch.where(t["classification"] > 0, torch.tensor(8.0), torch.tensor(-8.0)),
        "ltrb": t["ltrb"].clone(),
        "centerness": torch.logit(t["centerness"].clamp(1e-4, 1 - 1e-4)),
    }
    _, parts = fcos_loss(perfect, t)

    assert parts["giou"] < 1e-5, f"GIoU should vanish on exact boxes, got {parts['giou']}"
    assert parts["cls"] < 1e-3, f"classification loss should vanish, got {parts['cls']}"
    print(f"OK  perfect prediction -> cls {parts['cls']:.2e}, giou {parts['giou']:.2e}, {parts['n_pos']} positive cells")

    wrong = dict(perfect)
    wrong["ltrb"] = t["ltrb"] * 2.0
    _, wrong_parts = fcos_loss(wrong, t)
    assert wrong_parts["giou"] > parts["giou"], "doubling every box must raise the GIoU loss"
    print(f"OK  doubled boxes -> giou {wrong_parts['giou']:.3f}")

    # Scale invariance: the same relative error costs the same on a small and a large box
    small = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    large = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    assert abs(float(_giou(small * 1.5, small)) - float(_giou(large * 1.5, large))) < 1e-5
    print("OK  GIoU is scale-free: identical cost at 1 cell and at 10 cells")

    # Supervision density is the whole point of this head
    assert parts["n_pos"] > 50, f"expected dense supervision, got {parts['n_pos']} cells"
    print(f"OK  {parts['n_pos']} supervised cells for 2 nodules (CenterNet would give 2)")

    empty_t = make_fcos_targets({"boxes": torch.zeros((1, 4, 4)), "num_boxes": torch.tensor([0])},
                                1, heatmap_size, input_size, "cpu")
    empty_p = {"classification": torch.full((1, 1, heatmap_size, heatmap_size), -8.0),
               "ltrb": torch.ones((1, 4, heatmap_size, heatmap_size)),
               "centerness": torch.zeros((1, 1, heatmap_size, heatmap_size))}
    _, empty_parts = fcos_loss(empty_p, empty_t)
    assert empty_parts["n_pos"] == 0 and empty_parts["giou"] == 0.0
    print("OK  nodule-free image contributes no regression loss and does not crash")

    # The normaliser floor must leave target 224 untouched and hold every other target to
    # the same clean-batch loss. Without it, 1024 pulled 21x harder than 224.
    assert abs(_negative_batch_floor(224 // 4) - 1.0) < 1e-9,         "floor must be exactly 1.0 at target 224 or banked runs at that target move"

    clean_losses = {}
    for target_size in (224, 256, 384, 512, 768, 1024):
        side = target_size // 4
        logits = torch.full((2, 1, side, side), -2.0)
        clean_losses[target_size] = float(_focal(logits, torch.zeros((2, 1, side, side))))

    spread = max(clean_losses.values()) / min(clean_losses.values())
    assert spread < 1.01, f"clean-batch loss still varies {spread:.1f}x across targets"
    print("OK  nodule-free batch costs {:.2f} at every target from 224 to 1024".format(
        clean_losses[224]))

    # A batch WITH nodules must be unaffected: its positive count clears the floor
    side = 256
    logits = torch.full((2, 1, side, side), -2.0)
    tgt = torch.zeros((2, 1, side, side))
    tgt[0, 0, :16, :16] = 1.0
    assert int(tgt.sum()) > _negative_batch_floor(side), "floor must not bind on real nodules"
    print("OK  a nodule-bearing batch clears the floor, so its normalisation is unchanged")


if __name__ == "__main__":
    _demo()
