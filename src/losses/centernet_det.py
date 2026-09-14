"""
Loss for the complete CenterNet detector.

L = focal(heatmap) + lambda_size * L1(size) + lambda_offset * L1(offset)

Weights follow Zhou, Wang & Krahenbuhl (2019): lambda_size = 0.1, lambda_offset = 1.0.
The size term is downweighted because it is measured in heatmap cells, so its raw
magnitude is an order above the offset term, which lives in [0, 1).
"""

import torch

from src.core.registry import LOSS_REGISTRY
from src.evaluation.centernet_targets import _gather_by_index


def _focal_heatmap(prediction, target, alpha=2.0, beta=4.0):
    """
    Penalty-reduced focal loss, identical in form to focal_v5 so the classification
    branch is trained exactly as in the heatmap-only runs and the two are comparable.
    """
    prediction = torch.sigmoid(prediction.float()).clamp(1e-6, 1 - 1e-6)
    target = target.float()

    positive_mask = target.ge(0.99).float()
    negative_mask = target.lt(0.99).float()

    positive_loss = -torch.log(prediction) * torch.pow(1 - prediction, alpha) * positive_mask
    negative_loss = -torch.log(1 - prediction) * torch.pow(prediction, alpha) \
                    * torch.pow(1 - target, beta) * negative_mask

    return (positive_loss.sum() + negative_loss.sum()) / positive_mask.sum().clamp(min=1)


def _masked_l1(prediction_map, target_values, index, mask):
    """
    L1 between predictions gathered at object centres and their targets.

    Normalised by the number of real objects, so padded slots contribute nothing to
    either the numerator or the denominator.
    """
    predicted = _gather_by_index(prediction_map, index)                 # (B, K, C)
    mask = mask.unsqueeze(2).expand_as(predicted).float()

    return (torch.abs(predicted * mask - target_values * mask).sum()) / (mask.sum() + 1e-4)


@LOSS_REGISTRY.register("centernet_det")
def centernet_detection_loss(prediction, target, lambda_size=0.1, lambda_offset=1.0):
    """
    Args:
        prediction: dict from CenterNetDetHead - heatmap (B,1,H,W), size_wh (B,2,H,W),
                    offset (B,2,H,W); heatmap is raw logits
        target: dict from make_centernet_targets - heatmap, size_wh, offset, index, mask

    Returns:
        (total loss, dict of the three components detached for logging)
    """
    heatmap_loss = _focal_heatmap(prediction["heatmap"], target["heatmap"])
    size_loss    = _masked_l1(prediction["size_wh"], target["size_wh"], target["index"], target["mask"])
    offset_loss  = _masked_l1(prediction["offset"],  target["offset"],  target["index"], target["mask"])

    total = heatmap_loss + lambda_size * size_loss + lambda_offset * offset_loss

    parts = {
        "heatmap": float(heatmap_loss.detach()),
        "size":    float(size_loss.detach()),
        "offset":  float(offset_loss.detach()),
    }

    return total, parts


def _demo():
    """Self-check: perfect predictions give ~0 regression loss; wrong ones give more."""
    from src.evaluation.centernet_targets import make_centernet_targets

    input_size, heatmap_size, max_objects = 1024, 64, 4
    boxes = torch.zeros((2, max_objects, 4))
    boxes[0, 0] = torch.tensor([300.0, 400.0, 80.0, 60.0])
    boxes[0, 1] = torch.tensor([700.0, 250.0, 40.0, 40.0])
    boxes[1, 0] = torch.tensor([512.0, 512.0, 120.0, 100.0])
    counts = torch.tensor([2, 1])

    built = make_centernet_targets({"boxes": boxes, "num_boxes": counts}, 2, heatmap_size,
                                   input_size, "cpu", max_objects=max_objects)

    # Dense maps holding exactly the target values at the object centres
    dense_size   = torch.zeros((2, 2, heatmap_size, heatmap_size))
    dense_offset = torch.zeros((2, 2, heatmap_size, heatmap_size))

    for b in range(2):
        for k in range(max_objects):
            if built["mask"][b, k] == 0: continue

            flat = int(built["index"][b, k])
            dense_size[b, :, flat // heatmap_size, flat % heatmap_size]   = built["size_wh"][b, k]
            dense_offset[b, :, flat // heatmap_size, flat % heatmap_size] = built["offset"][b, k]

    logits = torch.logit(built["heatmap"].clamp(1e-4, 1 - 1e-4))
    perfect = {"heatmap": logits, "size_wh": dense_size, "offset": dense_offset}
    _, parts = centernet_detection_loss(perfect, built)

    assert parts["size"] < 1e-4,   f"size loss should vanish on perfect prediction, got {parts['size']}"
    assert parts["offset"] < 1e-4, f"offset loss should vanish on perfect prediction, got {parts['offset']}"
    print(f"OK  perfect prediction -> size {parts['size']:.2e}, offset {parts['offset']:.2e}")

    wrong = {"heatmap": logits, "size_wh": dense_size * 2, "offset": dense_offset + 0.5}
    _, wrong_parts = centernet_detection_loss(wrong, built)

    assert wrong_parts["size"] > parts["size"],     "doubling sizes must raise the size loss"
    assert wrong_parts["offset"] > parts["offset"], "shifting offsets must raise the offset loss"
    print(f"OK  degraded prediction -> size {wrong_parts['size']:.3f}, offset {wrong_parts['offset']:.3f}")

    # Padded slots must not contribute: image 1 has one object, three padded
    only_padding = built["mask"][1, 1:].sum().item()
    assert only_padding == 0, "expected padded slots to be masked out"
    print("OK  padded object slots are masked out of the regression losses")


if __name__ == "__main__":
    _demo()
