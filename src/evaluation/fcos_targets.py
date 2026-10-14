"""
Target assignment and decoding for a single-level FCOS detector.

Motivation. CenterNet supervises box regression at ONE pixel per object - the centre -
so the learned resizer receives gradient from a single cell per nodule regardless of how
large that nodule is. FCOS instead makes every cell inside the box a positive sample that
regresses its own distances to the four box edges, which multiplies the positive
supervision by the box area in cells: measured on NODE21, 6x at target=224 rising to 32x
at target=512.

Single-level: the existing head reads only the stage1 FPN feature, so there is no
multi-level assignment and no per-level size range. Boxes here are small and uniform
enough (2.5-6 cells across) that a second level would carry almost nothing.

No centre sampling. Reference FCOS restricts positives to a radius around the centre to
suppress low-quality edge samples, but a 2.46-cell box at target=224 would be reduced to
roughly one cell, which is the CenterNet behaviour this module exists to avoid.

All box coordinates entering and leaving are (x1, y1, x2, y2) in INPUT_SIZE pixel space.
"""

import torch
import torch.nn.functional as F


def _cell_centres(heatmap_size, device):
    """(H*W, 2) tensor of cell-centre coordinates in heatmap units."""
    coords = torch.arange(heatmap_size, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def make_fcos_targets(targets, batch_size, heatmap_size, input_size, device, max_objects=4,
                      num_classes=1):
    """
    Assign every cell to a ground-truth box or to background.

    A cell is positive when its centre falls inside a box. Cells inside more than one box
    are assigned to the SMALLEST box, the standard FCOS ambiguity rule - it keeps a large
    lesion from swallowing a small one that overlaps it.

    Args:
        targets: collated dict with "boxes" (B, max_objects, 4) as (cx,cy,w,h) in
                 input_size space and "num_boxes" (B,)
        heatmap_size: side of the square prediction grid
        input_size: side of the model input, for the input->heatmap scale
        num_classes: 1 for the NODE21 single-class track, 14 for the VinDr multi-class
                 sweep. At num_classes=1 every tensor below is shaped exactly as it was
                 before this argument existed, so no banked run changes.

    Returns:
        dict with
          classification (B, C, H, W)  1.0 at a positive cell, in ITS BOX'S class channel
          ltrb           (B, 4, H, W)  distances to left/top/right/bottom edges, in cells
          centerness     (B, 1, H, W)  sqrt of the l/r and t/b balance, 0 at negatives
          positive_mask  (B, 1, H, W)  1.0 where a box was assigned
    """
    scale = heatmap_size / input_size
    centres = _cell_centres(heatmap_size, device)                     # (N, 2)
    n_cells = centres.shape[0]

    classification = torch.zeros((batch_size, num_classes, heatmap_size, heatmap_size), device=device)
    ltrb           = torch.zeros((batch_size, 4, heatmap_size, heatmap_size), device=device)
    centerness     = torch.zeros((batch_size, 1, heatmap_size, heatmap_size), device=device)
    positive_mask  = torch.zeros((batch_size, 1, heatmap_size, heatmap_size), device=device)

    boxes_batch, counts = targets["boxes"], targets["num_boxes"]

    # Absent for the single-class track, where every box is generic foreground.
    classes_batch = targets.get("box_classes")

    for i in range(batch_size):
        n_objects = min(int(counts[i]), max_objects)

        if n_objects == 0: continue

        raw = boxes_batch[i, :n_objects].float().to(device) * scale    # (n, 4) cx,cy,w,h in cells
        corners = torch.stack([
            raw[:, 0] - raw[:, 2] / 2, raw[:, 1] - raw[:, 3] / 2,
            raw[:, 0] + raw[:, 2] / 2, raw[:, 1] + raw[:, 3] / 2,
        ], dim=1)                                                      # (n, 4)

        # Distances from every cell centre to every box edge: (N, n)
        left   = centres[:, 0:1] - corners[None, :, 0]
        top    = centres[:, 1:2] - corners[None, :, 1]
        right  = corners[None, :, 2] - centres[:, 0:1]
        bottom = corners[None, :, 3] - centres[:, 1:2]
        distances = torch.stack([left, top, right, bottom], dim=2)     # (N, n, 4)

        inside = distances.min(dim=2).values > 0                       # (N, n)

        if not inside.any(): continue

        # Smallest-box-wins on overlap
        areas = (raw[:, 2] * raw[:, 3]).clamp(min=1e-6)                # (n,)
        areas_per_cell = torch.where(inside, areas[None, :].expand(n_cells, -1),
                                     torch.full((n_cells, n_objects), float("inf"), device=device))
        chosen = areas_per_cell.argmin(dim=1)                          # (N,)
        is_positive = inside.any(dim=1)                                # (N,)

        chosen_distances = distances[torch.arange(n_cells, device=device), chosen]   # (N, 4)

        # Centerness downweights cells near the box edge, where regression is least reliable
        lr = chosen_distances[:, [0, 2]]
        tb = chosen_distances[:, [1, 3]]
        balance = (lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-6)) * \
                  (tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-6))
        cell_centerness = balance.clamp(min=0).sqrt()

        grid = lambda flat: flat.view(heatmap_size, heatmap_size)
        positive_mask[i, 0] = grid(is_positive.float())

        if num_classes == 1:
            classification[i, 0] = grid(is_positive.float())

        else:
            # A positive cell fires in the channel of the box it was assigned to, which
            # is the SMALLEST box covering it - the same ambiguity rule used above, so a
            # cell never claims two classes. Overlapping findings of different classes
            # (cardiomegaly over aortic enlargement is the common pair) are therefore
            # resolved to the smaller lesion rather than being double-counted.
            box_classes = classes_batch[i, :n_objects].long().to(device)
            cell_class = box_classes[chosen]                                   # (N,)

            for class_index in torch.unique(cell_class[is_positive]):
                fires = is_positive & (cell_class == class_index)
                classification[i, int(class_index)] = grid(fires.float())
        centerness[i, 0]     = grid(torch.where(is_positive, cell_centerness, torch.zeros_like(cell_centerness)))

        for c in range(4):
            ltrb[i, c] = grid(torch.where(is_positive, chosen_distances[:, c],
                                          torch.zeros_like(chosen_distances[:, c])))

    return {"classification": classification, "ltrb": ltrb,
            "centerness": centerness, "positive_mask": positive_mask}


def ltrb_to_boxes(ltrb, heatmap_size, input_size, device=None):
    """
    Convert (B, 4, H, W) edge distances into (B, H*W, 4) x1,y1,x2,y2 boxes in input space.
    """
    device = device or ltrb.device
    centres = _cell_centres(heatmap_size, device)                      # (N, 2)
    scale = input_size / heatmap_size

    flat = ltrb.permute(0, 2, 3, 1).reshape(ltrb.shape[0], -1, 4)      # (B, N, 4)
    cx, cy = centres[None, :, 0], centres[None, :, 1]

    return torch.stack([
        (cx - flat[:, :, 0]) * scale,
        (cy - flat[:, :, 1]) * scale,
        (cx + flat[:, :, 2]) * scale,
        (cy + flat[:, :, 3]) * scale,
    ], dim=2)


def _nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS. Kept local so the module does not depend on torchvision."""
    from src.evaluation.centernet_targets import box_iou

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        best = order[0]
        keep.append(int(best))

        if order.numel() == 1: break

        rest = order[1:]
        ious = box_iou(boxes[best][None, :], boxes[rest])[0]
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def decode_fcos(classification, ltrb, centerness, input_size, top_k=20,
                score_threshold=0.05, nms_iou=0.5):
    """
    Turn the three FCOS branches into boxes.

    Confidence is sqrt(cls * centerness), the reference combination: a cell must be both
    confident that an object is present AND well-centred on it to score highly.

    Args:
        classification: (B, 1, H, W) raw logits
        ltrb: (B, 4, H, W) POSITIVE edge distances in cells (already activated)
        centerness: (B, 1, H, W) raw logits

    Returns:
        boxes (B, top_k, 4) in input_size space, scores (B, top_k), both zero-padded
    """
    batch, _, heatmap_size, _ = classification.shape
    device = classification.device

    scores = (torch.sigmoid(classification) * torch.sigmoid(centerness)).sqrt()
    scores = scores.view(batch, -1)                                    # (B, N)
    all_boxes = ltrb_to_boxes(ltrb, heatmap_size, input_size, device)  # (B, N, 4)

    out_boxes  = torch.zeros((batch, top_k, 4), device=device)
    out_scores = torch.zeros((batch, top_k), device=device)

    for i in range(batch):
        candidate = scores[i] >= score_threshold

        if not candidate.any(): continue

        boxes_i, scores_i = all_boxes[i][candidate], scores[i][candidate]

        # Cap before NMS - at 128x128 nearly every cell can clear a low threshold early
        # in training, and greedy NMS over 16k boxes is quadratic.
        if scores_i.numel() > 1000:
            top = torch.topk(scores_i, 1000).indices
            boxes_i, scores_i = boxes_i[top], scores_i[top]

        keep = _nms(boxes_i, scores_i, nms_iou)[:top_k]
        out_boxes[i, :len(keep)]  = boxes_i[keep]
        out_scores[i, :len(keep)] = scores_i[keep]

    return out_boxes, out_scores


def _demo():
    """Self-check: known boxes must assign sensibly and decode back to themselves."""
    input_size, heatmap_size = 1024, 128

    boxes = torch.zeros((2, 4, 4))
    boxes[0, 0] = torch.tensor([300.0, 400.0, 80.0, 60.0])
    boxes[0, 1] = torch.tensor([700.0, 250.0, 120.0, 120.0])
    boxes[1, 0] = torch.tensor([512.0, 512.0, 160.0, 100.0])
    counts = torch.tensor([2, 1])

    t = make_fcos_targets({"boxes": boxes, "num_boxes": counts}, 2, heatmap_size, input_size, "cpu")

    # Supervision must be dense, not a single cell per object
    n_pos_img0 = int(t["positive_mask"][0].sum())
    expected = (80 * 120 / (1024 / heatmap_size) ** 2) + (120 * 120 / (1024 / heatmap_size) ** 2)
    assert n_pos_img0 > 10, f"expected dense positives, got {n_pos_img0}"
    assert abs(n_pos_img0 - expected) / expected < 0.25, f"{n_pos_img0} positives vs ~{expected:.0f} expected from box areas"
    print(f"OK  dense assignment: {n_pos_img0} positive cells for 2 boxes (~{expected:.0f} expected by area)")

    # An image with no boxes must be entirely background
    assert t["positive_mask"][1].sum() > 0
    empty = make_fcos_targets({"boxes": torch.zeros((1, 4, 4)), "num_boxes": torch.tensor([0])},
                              1, heatmap_size, input_size, "cpu")
    assert empty["positive_mask"].sum() == 0
    print("OK  nodule-free image assigns zero positives")

    # Centerness is 1 at the box centre and decays outward
    assert t["centerness"].max().item() > 0.95, f"peak centerness {t['centerness'].max():.3f}"
    positives = t["centerness"][0][t["positive_mask"][0] > 0]
    assert positives.min() < 0.6, "centerness should decay toward box edges"
    print(f"OK  centerness peaks at {t['centerness'].max():.3f}, decays to {positives.min():.3f}")

    # GT ltrb must decode back to the original boxes
    from src.evaluation.centernet_targets import box_iou
    decoded = ltrb_to_boxes(t["ltrb"], heatmap_size, input_size)
    mask = t["positive_mask"][0].view(-1) > 0
    truth = torch.tensor([[300 - 40, 400 - 30, 300 + 40, 400 + 30],
                          [700 - 60, 250 - 60, 700 + 60, 250 + 60]], dtype=torch.float32)
    ious = box_iou(truth, decoded[0][mask])
    assert ious.max(dim=1).values.min() > 0.99, f"round-trip IoU {ious.max(dim=1).values}"
    print("OK  every positive cell reconstructs its own box at IoU > 0.99")

    # Full decode path with confident predictions
    logits = torch.where(t["classification"] > 0, torch.tensor(4.0), torch.tensor(-4.0))
    cness  = torch.where(t["centerness"] > 0, torch.logit(t["centerness"].clamp(1e-3, 1 - 1e-3)), torch.tensor(-4.0))
    out_boxes, out_scores = decode_fcos(logits, t["ltrb"], cness, input_size, top_k=10)

    matched = box_iou(truth, out_boxes[0][out_scores[0] > 0])
    assert matched.max(dim=1).values.min() > 0.9, f"decode+NMS IoU {matched.max(dim=1).values}"
    print(f"OK  decode + NMS recovers both boxes ({int((out_scores[0] > 0).sum())} detections kept)")

    # ---- multi-class assignment, for the VinDr sweep ----
    # Two well-separated boxes of different classes must fire in their own channels and
    # nowhere else. The failure this catches is a cell claiming more than one class, which
    # would make the per-class scores meaningless while every total still looked right.
    multi = {"boxes": boxes.clone(), "num_boxes": counts.clone(),
             "box_classes": torch.zeros((2, 4), dtype=torch.long)}
    multi["box_classes"][0, 0] = 3        # first box  -> class 3
    multi["box_classes"][0, 1] = 11       # second box -> class 11

    m = make_fcos_targets(multi, 2, heatmap_size, input_size, "cpu", num_classes=14)

    assert m["classification"].shape == (2, 14, heatmap_size, heatmap_size)

    fired = [c for c in range(14) if m["classification"][0, c].sum() > 0]
    assert fired == [3, 11], f"expected classes 3 and 11 to fire, got {fired}"

    # No cell may be positive in two channels at once
    per_cell = m["classification"][0].sum(dim=0)
    assert per_cell.max() <= 1.0, f"a cell claimed {per_cell.max():.0f} classes"

    # Total positives must match the single-class run exactly - only the channel moved
    assert int(m["positive_mask"][0].sum()) == n_pos_img0,         "multi-class assignment changed WHICH cells are positive, not just their channel"
    print("OK  multi-class: boxes fire in channels 3 and 11, no cell claims two classes")

    # Class ids must survive into the loss unchanged in count
    assert abs(float(m["classification"].sum()) - float(t["classification"].sum())) < 1e-6
    print("OK  multi-class positive count identical to single-class ({:.0f} cells)".format(
        float(m["classification"].sum())))


if __name__ == "__main__":
    _demo()
