import math
import torch
import numpy as np
from torch.cuda.amp import autocast
from .detection_metrics import _nms_radius_for, _extract_peaks

@torch.no_grad()
def compute_froc_data_v5(model, loader, device, thresholds=None, dist_thresh_mm=15.0, pixel_spacing_mm=0.175, input_size=2048):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05).tolist()

    heatmap_size   = model.heatmap_size
    nms_radius     = _nms_radius_for(heatmap_size)
    dist_thresh_px = dist_thresh_mm / pixel_spacing_mm
    scale          = input_size / heatmap_size
    model.eval()
    all_images = []

    for images, targets in loader:
        images = images.to(device)
        with autocast(enabled=device.type == "cuda"):
            preds = torch.sigmoid(model(images)).cpu()

        for i in range(preds.shape[0]):
            pred_hm    = preds[i, 0]
            has_nodule = bool(targets["has_nodule"][i])
            gt_cx      = targets["center_2048"][0][i].item()
            gt_cy      = targets["center_2048"][1][i].item()
            source     = targets["source"][i] if "source" in targets else "unknown"

            peak_coords, peak_scores = _extract_peaks(pred_hm, score_thresh=0.01,
                                                       nms_radius=nms_radius)
            peaks = []
            for k in range(len(peak_scores)):
                py = peak_coords[k, 0].item() * scale
                px = peak_coords[k, 1].item() * scale
                dist = (math.sqrt((px - gt_cx)**2 + (py - gt_cy)**2)
                        if has_nodule else float('inf'))
                peaks.append({"score": peak_scores[k].item(), "dist": dist})

            all_images.append({"has_nodule": has_nodule, "source": source, "peaks": peaks})

    results = {"thresholds": [], "all": [], "jsrt": [], "node21": []}
    for thresh in thresholds:
        results["thresholds"].append(thresh)
        for sf in ("all", "jsrt", "node21"):
            subset = [img for img in all_images if sf == "all" or img["source"] == sf]
            if not subset:
                continue
            tp, fn, fp = 0, 0, 0
            for img in subset:
                above = [p for p in img["peaks"] if p["score"] >= thresh]
                if img["has_nodule"]:
                    matched = any(p["dist"] <= dist_thresh_px for p in above)
                    if matched:
                        tp += 1
                        fp += max(0, len(above) - 1)
                    else:
                        fn += 1
                        fp += len(above)
                else:
                    fp += len(above)
            n = len(subset)
            results[sf].append({
                "threshold": thresh,
                "sensitivity": tp / max(tp + fn, 1),
                "fp_per_image": fp / max(n, 1),
            })
    return results
