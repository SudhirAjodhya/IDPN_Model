import torch
import math

def render_gaussian_heatmap_direct(centers_x, centers_y, radiuses, has_nodule,
                                   heatmap_size, sigma_min=2.5, device="cpu"):
    """
    Renders Gaussian blobs centered at given coordinates directly on device.

    Shape (B, 1, H, W) to match CenterNetHead output. Without the channel dim,
    focal_v5(pred=(B,1,H,W), target=(B,H,W)) broadcasts to (B,B,H,W) - a
    cross-batch product that corrupts gradients (pred[i] gets supervised by
    target[j] for all j). This was the reason Node21-only sensitivity plateaued
    at ~0.28 vs matrix_v1's 0.46 on the same test set.
    """
    batch_size = len(centers_x)
    hm = torch.zeros((batch_size, 1, heatmap_size, heatmap_size), device=device)

    y = torch.arange(heatmap_size, device=device).view(-1, 1).float()
    x = torch.arange(heatmap_size, device=device).view(1, -1).float()

    for i in range(batch_size):
        if not has_nodule[i]:
            continue
        cx = centers_x[i]
        cy = centers_y[i]
        r = radiuses[i]
        sigma = max(r / 3.0, sigma_min)

        dist_sq = (x - cx)**2 + (y - cy)**2
        blob = torch.exp(-dist_sq / (2 * sigma**2))

        hm[i, 0] = torch.maximum(hm[i, 0], blob)

    return hm

def make_gt_heatmap(targets, batch_size, heatmap_size, sigma_min=2.5, input_size=2048, device="cpu"):
    scale = heatmap_size / input_size
    # After default_collate, center_2048 is a 2-tuple (cx_tensor, cy_tensor), each
    # of length B - NOT a per-sample list of pairs. Prior indexing target[i][0]/[1]
    # worked by accident at B=1, silently swapped cx/cy across samples at B=2, and
    # IndexErrors at B>=3. Match detection_metrics.py which had it right.
    cx_batch = targets["center_2048"][0]
    cy_batch = targets["center_2048"][1]
    cx, cy, radiuses, has_nodule = [], [], [], []
    for i in range(len(targets["has_nodule"])):
        hn = bool(targets["has_nodule"][i])
        has_nodule.append(hn)
        if hn:
            cx.append(cx_batch[i].item() * scale)
            cy.append(cy_batch[i].item() * scale)
            radiuses.append(targets["radius_px"][i].item() * scale)
        else:
            cx.append(0.0)
            cy.append(0.0)
            radiuses.append(0.0)
    
    hm = render_gaussian_heatmap_direct(
        cx, cy, radiuses, has_nodule,
        heatmap_size=heatmap_size,
        sigma_min=sigma_min,
        device=device,
    )
    # Normalize each item's Gaussian so the peak == 1.0. See directionB line 532.
    # hm is (B, 1, H, W) - normalize per-item over the spatial dims (2, 3).
    peak = hm.amax(dim=(2, 3), keepdim=True)
    hm = hm / (peak + 1e-8)
    return hm
