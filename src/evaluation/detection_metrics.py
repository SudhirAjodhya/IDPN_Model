import math
import torch
import torch.nn.functional as F
from torch.amp import autocast

def _nms_radius_for(heatmap_size):
    return 6 if heatmap_size > 128 else 3

def _extract_peaks(heatmap, score_thresh=0.3, nms_radius=3):
    kernel = 2 * nms_radius + 1
    hm = heatmap.unsqueeze(0).unsqueeze(0)
    local_max = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=nms_radius)
    is_peak = (heatmap == local_max.squeeze()) & (heatmap >= score_thresh)
    peak_coords = torch.nonzero(is_peak, as_tuple=False)
    peak_scores = heatmap[is_peak]
    return peak_coords, peak_scores

@torch.no_grad()
def compute_detection_metrics_v4(model, loader, device, score_thresh=0.3, dist_thresh_mm=15.0, pixel_spacing_mm=0.175, input_size=2048):
    model.eval()
    heatmap_size = model.heatmap_size
    nms_radius   = _nms_radius_for(heatmap_size)
    dist_thresh_px = dist_thresh_mm / pixel_spacing_mm
    scale = input_size / heatmap_size

    counters = {s: {"tp": 0, "fn": 0, "fp": 0, "n": 0} for s in ("all", "jsrt", "node21")}
    size_bins = {"small": (0, 10), "medium": (11, 20), "large": (21, 9999)}
    size_counters = {b: {"tp": 0, "fn": 0, "n_nod": 0} for b in size_bins}

    for images, targets in loader:
        images = images.to(device)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            preds = torch.sigmoid(model(images)).cpu()

        for i in range(preds.shape[0]):
            pred_hm    = preds[i, 0]
            has_nodule = bool(targets["has_nodule"][i])
            gt_cx      = targets["center_2048"][0][i].item()
            gt_cy      = targets["center_2048"][1][i].item()
            source     = targets["source"][i] if "source" in targets else "unknown"
            size_mm    = targets["size_mm"][i].item() if "size_mm" in targets else 0.0

            for src in ("all", source):
                if src in counters:
                    counters[src]["n"] += 1

            peak_coords, peak_scores = _extract_peaks(pred_hm, score_thresh, nms_radius)

            gt_matched = False
            detected   = False
            if len(peak_scores) > 0:
                order = torch.argsort(peak_scores, descending=True)
                for k in order:
                    py = peak_coords[k, 0].item() * scale
                    px = peak_coords[k, 1].item() * scale
                    dist = math.sqrt((px - gt_cx)**2 + (py - gt_cy)**2)
                    if has_nodule and not gt_matched and dist <= dist_thresh_px:
                        gt_matched = True
                        detected   = True
                        break

            if has_nodule:
                if detected:
                    for src in ("all", source):
                        if src in counters:
                             counters[src]["tp"] += 1
                else:
                    for src in ("all", source):
                        if src in counters:
                            counters[src]["fn"] += 1
                n_fp = max(0, len(peak_scores) - (1 if detected else 0))
                for src in ("all", source):
                    if src in counters:
                        counters[src]["fp"] += n_fp
            else:
                for src in ("all", source):
                    if src in counters:
                        counters[src]["fp"] += len(peak_scores)

            if source == "jsrt" and has_nodule and size_mm > 0:
                for bin_name, (lo, hi) in size_bins.items():
                    if lo <= size_mm <= hi:
                        size_counters[bin_name]["n_nod"] += 1
                        if detected:
                            size_counters[bin_name]["tp"] += 1
                        else:
                            size_counters[bin_name]["fn"] += 1

    results = {}
    for src, c in counters.items():
        if c["n"] == 0:
            continue
        results[src] = {
            "sensitivity":  c["tp"] / max(c["tp"] + c["fn"], 1),
            "fp_per_image": c["fp"] / max(c["n"], 1),
            "tp": c["tp"], "fn": c["fn"], "fp": c["fp"], "n_images": c["n"],
        }

    size_results = {}
    for bin_name, c in size_counters.items():
        if c["n_nod"] == 0:
            continue
        size_results[bin_name] = {
            "sensitivity": c["tp"] / max(c["tp"] + c["fn"], 1),
            "n_nodules": c["n_nod"],
        }
    results["jsrt_by_size"] = size_results
    return results
