"""
Ground-truth construction and decoding for the complete CenterNet detector.

The heatmap-only path (`src/evaluation/heatmap.py`) predicts object centres and nothing
else, which is one of CenterNet's three branches. This module supplies the targets and
the decode step for the full formulation of Zhou, Wang & Krahenbuhl, "Objects as Points"
(2019): a centre heatmap, a width/height regression, and a sub-pixel centre offset.

All box coordinates entering this module are (cx, cy, w, h) in INPUT_SIZE pixel space.
All box coordinates leaving it are (x1, y1, x2, y2), also in input_size pixel space.
"""

import math

import torch
import torch.nn.functional as F


def gaussian_radius(box_height, box_width, min_overlap=0.7):
    """
    CornerNet's adaptive Gaussian radius: the largest centre displacement that still
    leaves a box of this size overlapping the ground truth by at least `min_overlap` IoU.

    Using a size-adaptive radius rather than a fixed sigma matters here because nodule
    diameters span roughly 5-30 mm; a fixed sigma either smears small nodules or leaves
    large ones with a near-delta target.

    Args:
        box_height, box_width: box size in heatmap cells
        min_overlap: IoU the displaced box must still achieve

    Returns:
        radius in heatmap cells
    """
    a1 = 1
    b1 = box_height + box_width
    c1 = box_width * box_height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (box_height + box_width)
    c2 = (1 - min_overlap) * box_width * box_height
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_height + box_width)
    c3 = (min_overlap - 1) * box_width * box_height
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / (2 * a3)

    return max(0.0, min(r1, r2, r3))


def _draw_gaussian(heatmap, centre_x, centre_y, radius):
    """Splat an unnormalised Gaussian onto `heatmap` in place, keeping the max on overlap."""
    diameter = 2 * int(radius) + 1
    sigma = diameter / 6.0
    size = int(radius)

    height, width = heatmap.shape
    left, right = min(centre_x, size), min(width - centre_x, size + 1)
    top, bottom = min(centre_y, size), min(height - centre_y, size + 1)

    if right <= -left or bottom <= -top: return

    ys = torch.arange(-top, bottom, device=heatmap.device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(-left, right, device=heatmap.device, dtype=torch.float32).view(1, -1)
    blob = torch.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2 + 1e-8))

    region = heatmap[centre_y - top:centre_y + bottom, centre_x - left:centre_x + right]
    torch.maximum(region, blob, out=region)


def make_centernet_targets(targets, batch_size, heatmap_size, input_size, device,
                           max_objects=4, min_overlap=0.7):
    """
    Build the three CenterNet targets from padded per-image boxes.

    Args:
        targets: collated target dict carrying "boxes" (B, max_objects, 4) as (cx,cy,w,h)
                 in input_size space, and "num_boxes" (B,)
        batch_size: B
        heatmap_size: side of the square output grid
        input_size: side of the square model input, for the input->heatmap scale
        device: where to allocate
        max_objects: padding width of the regression targets
        min_overlap: IoU floor for the adaptive Gaussian radius

    Returns:
        dict with
          heatmap  (B, 1, H, W)  peak-normalised Gaussians at each centre
          size_wh  (B, K, 2)     box width/height in heatmap cells
          offset   (B, K, 2)     sub-pixel remainder of the centre
          index    (B, K)        flattened H*W position of each centre cell
          mask     (B, K)        1 for real objects, 0 for padding
    """
    scale = heatmap_size / input_size

    heatmap = torch.zeros((batch_size, 1, heatmap_size, heatmap_size), device=device)
    size_wh = torch.zeros((batch_size, max_objects, 2), device=device)
    offset  = torch.zeros((batch_size, max_objects, 2), device=device)
    index   = torch.zeros((batch_size, max_objects), dtype=torch.long, device=device)
    mask    = torch.zeros((batch_size, max_objects), device=device)

    boxes_batch = targets["boxes"]
    counts      = targets["num_boxes"]

    for i in range(batch_size):
        n_objects = int(counts[i])

        for j in range(min(n_objects, max_objects)):
            cx, cy, w, h = [float(v) * scale for v in boxes_batch[i, j]]

            if w <= 0 or h <= 0: continue

            grid_x, grid_y = int(cx), int(cy)

            # A centre landing outside the grid would corrupt the flattened index
            if not (0 <= grid_x < heatmap_size and 0 <= grid_y < heatmap_size): continue

            radius = max(1, int(gaussian_radius(math.ceil(h), math.ceil(w), min_overlap)))
            _draw_gaussian(heatmap[i, 0], grid_x, grid_y, radius)

            size_wh[i, j] = torch.tensor([w, h], device=device)
            offset[i, j]  = torch.tensor([cx - grid_x, cy - grid_y], device=device)
            index[i, j]   = grid_y * heatmap_size + grid_x
            mask[i, j]    = 1.0

    return {"heatmap": heatmap, "size_wh": size_wh, "offset": offset, "index": index, "mask": mask}


def _gather_by_index(feature_map, index):
    """Gather (B, C, H, W) predictions at flattened positions (B, K) -> (B, K, C)."""
    batch, channels = feature_map.shape[0], feature_map.shape[1]
    flat = feature_map.view(batch, channels, -1).permute(0, 2, 1)          # (B, H*W, C)

    return flat.gather(1, index.unsqueeze(2).expand(-1, -1, channels))


def decode_detections(heatmap, size_wh, offset, input_size, top_k=20, nms_kernel=3):
    """
    Turn the three prediction branches into boxes, the standard CenterNet decode.

    Peak extraction is a 3x3 max-pool equality test, which is the reference
    implementation's NMS - no IoU-based suppression is applied here.

    Args:
        heatmap: (B, 1, H, W) RAW logits
        size_wh: (B, 2, H, W) predicted width/height in heatmap cells
        offset:  (B, 2, H, W) predicted sub-pixel centre offsets
        input_size: side of the model input, for scaling boxes back to input space
        top_k: peaks retained per image before score thresholding
        nms_kernel: max-pool window for peak detection

    Returns:
        boxes  (B, top_k, 4) as x1, y1, x2, y2 in input_size pixel space
        scores (B, top_k)    sigmoid confidence, descending
    """
    batch, _, height, width = heatmap.shape
    scores_map = torch.sigmoid(heatmap)

    # Keep only local maxima
    pooled = F.max_pool2d(scores_map, nms_kernel, stride=1, padding=nms_kernel // 2)
    scores_map = scores_map * (pooled == scores_map).float()

    scores, index = scores_map.view(batch, -1).topk(min(top_k, height * width), dim=1)
    grid_y = (index // width).float()
    grid_x = (index % width).float()

    picked_offset = _gather_by_index(offset, index)                        # (B, K, 2)
    picked_size   = _gather_by_index(size_wh, index).clamp(min=0)          # (B, K, 2)

    centre_x = grid_x + picked_offset[:, :, 0]
    centre_y = grid_y + picked_offset[:, :, 1]
    box_w, box_h = picked_size[:, :, 0], picked_size[:, :, 1]

    # heatmap cells -> input_size pixels
    scale = input_size / width
    boxes = torch.stack([
        (centre_x - box_w / 2) * scale,
        (centre_y - box_h / 2) * scale,
        (centre_x + box_w / 2) * scale,
        (centre_y + box_h / 2) * scale,
    ], dim=2)

    return boxes, scores


def box_iou(boxes_a, boxes_b):
    """
    Pairwise IoU between (N, 4) and (M, 4) boxes in x1,y1,x2,y2 form -> (N, M).
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)

    top_left     = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    overlap = (bottom_right - top_left).clamp(min=0)
    intersection = overlap[:, :, 0] * overlap[:, :, 1]

    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-8)


def _demo():
    """Self-check: a target built from known boxes must decode back to those boxes."""
    torch.manual_seed(0)
    input_size, heatmap_size, max_objects = 1024, 128, 4

    truth = torch.zeros((2, max_objects, 4))
    truth[0, 0] = torch.tensor([300.0, 400.0, 80.0, 60.0])
    truth[0, 1] = torch.tensor([700.0, 250.0, 40.0, 40.0])
    truth[1, 0] = torch.tensor([512.0, 512.0, 120.0, 100.0])
    counts = torch.tensor([2, 1])

    built = make_centernet_targets({"boxes": truth, "num_boxes": counts},
                                   batch_size=2, heatmap_size=heatmap_size,
                                   input_size=input_size, device="cpu", max_objects=max_objects)

    assert built["mask"].sum().item() == 3, "expected 3 real objects"
    assert built["heatmap"].max().item() == 1.0, "peak must be exactly 1.0"

    # Paint the GT regression values into dense maps, then decode and compare
    dense_wh     = torch.zeros((2, 2, heatmap_size, heatmap_size))
    dense_offset = torch.zeros((2, 2, heatmap_size, heatmap_size))

    for b in range(2):
        for k in range(max_objects):
            if built["mask"][b, k] == 0: continue

            flat = int(built["index"][b, k])
            gy, gx = flat // heatmap_size, flat % heatmap_size
            dense_wh[b, :, gy, gx]     = built["size_wh"][b, k]
            dense_offset[b, :, gy, gx] = built["offset"][b, k]

    logits = torch.log(built["heatmap"].clamp(1e-4, 1 - 1e-4) / (1 - built["heatmap"].clamp(1e-4, 1 - 1e-4)))
    boxes, scores = decode_detections(logits, dense_wh, dense_offset, input_size, top_k=10)

    for b, n in enumerate(counts.tolist()):
        for k in range(n):
            cx, cy, w, h = truth[b, k]
            expected = torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            ious = box_iou(expected[None, :], boxes[b])
            assert ious.max() > 0.99, f"round-trip IoU {ious.max():.3f} for image {b} box {k}"

    # IoU sanity on known geometry
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert abs(box_iou(a, a).item() - 1.0) < 1e-6
    assert box_iou(a, torch.tensor([[20.0, 20.0, 30.0, 30.0]])).item() == 0.0
    assert abs(box_iou(a, torch.tensor([[0.0, 0.0, 10.0, 5.0]])).item() - 0.5) < 1e-6

    print("OK  targets round-trip through decode at IoU > 0.99")
    print("OK  box_iou correct on identity, disjoint and half-overlap")


if __name__ == "__main__":
    _demo()
