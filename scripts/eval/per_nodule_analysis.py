"""
per_nodule_analysis.py
======================
Per-nodule analysis on saved checkpoints.

For each test nodule, extract the peak heatmap score at the GT location
for bilinear vs Talebi. Cross-compare which nodules each model catches/misses
to assess whether the mechanism space captures distinct features.

Usage:
    python scripts/eval/per_nodule_analysis.py
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.models.detector import NoduleDetector
from src.data.splits import build_three_way_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 2048
SCORE_THRESH = 0.3
HEATMAP_SIZE = 128

# Point to output directory - adjust group path as needed
OUTPUT_DIR = ROOT_DIR / "outputs"


def _extract_gt_peak_score(pred_hm, gt_cx, gt_cy, radius_px, heatmap_size=HEATMAP_SIZE):
    """Max score in a window of radius gt_r around the GT centre (heatmap coords)."""
    scale = heatmap_size / INPUT_SIZE
    gt_x = int(gt_cx * scale)
    gt_y = int(gt_cy * scale)
    gt_r = max(3, int(radius_px * scale))

    y0 = max(0, gt_y - gt_r)
    y1 = min(heatmap_size, gt_y + gt_r + 1)
    x0 = max(0, gt_x - gt_r)
    x1 = min(heatmap_size, gt_x + gt_r + 1)
    window = pred_hm[y0:y1, x0:x1]
    return window.max().item() if window.numel() > 0 else 0.0


def analyse_checkpoint(name, ckpt_path, test_loader):
    variant = "bilinear" if "bilinear" in name else "talebi"
    print(f"\n  Loading {name} from {ckpt_path}")
    if not ckpt_path.exists():
        print(f"    MISSING - skipping")
        return []

    model = NoduleDetector(downsampler_type=variant, backbone_type="convnext_tiny", head_type="centernet").to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    per_nodule = []
    with torch.no_grad():
        for images, targets in test_loader:
            for i in range(images.shape[0]):
                if not targets["has_nodule"][i]:
                    continue
                img = images[i:i+1].to(DEVICE)
                with autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=DEVICE.type == "cuda"):
                    pred = torch.sigmoid(model(img))[0, 0].cpu()

                gt_cx    = targets["center_2048"][0][i].item()
                gt_cy    = targets["center_2048"][1][i].item()
                radius   = targets["radius_px"][i].item()
                size_mm  = targets.get("size_mm", [0.0])[i]
                fname    = targets.get("filename", ["?"])[i]
                if hasattr(size_mm, "item"):
                    size_mm = size_mm.item()

                peak_score = _extract_gt_peak_score(pred, gt_cx, gt_cy, radius)
                caught = peak_score > SCORE_THRESH

                per_nodule.append({
                    "image_id":      fname,
                    "size_mm":       size_mm,
                    "gt_cx_2048":    gt_cx,
                    "gt_cy_2048":    gt_cy,
                    "gt_peak_score": peak_score,
                    "caught":        caught,
                })

    n = len(per_nodule)
    caught_n = sum(1 for p in per_nodule if p["caught"])
    print(f"    {caught_n}/{n} nodules caught (threshold={SCORE_THRESH})")
    return per_nodule


def cross_compare(results, seed):
    key_bl = f"bilinear_s{seed}"
    key_tl = f"talebi_s{seed}"
    if key_bl not in results or key_tl not in results:
        return
    bl = {p["image_id"]: p["caught"] for p in results[key_bl]}
    tl = {p["image_id"]: p["caught"] for p in results[key_tl]}
    both      = sum(1 for k in bl if bl.get(k) and tl.get(k))
    only_bl   = sum(1 for k in bl if bl.get(k) and not tl.get(k))
    only_tl   = sum(1 for k in bl if not bl.get(k) and tl.get(k))
    neither   = sum(1 for k in bl if not bl.get(k) and not tl.get(k))
    total     = both + only_bl + only_tl + neither
    print(f"\n  Seed {seed}: both={both} bilinear_only={only_bl} "
          f"talebi_only={only_tl} neither={neither}  (total={total})")


def main():
    print("Per-nodule analysis on checkpoints")

    # Build JSRT test loader (seed=42, test_seed=0 - same split as training)
    data_root = ROOT_DIR / "data_sets"
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=data_root / "JSRT" / "jsrt_metadata.csv",
        jsrt_image_dir=data_root / "JSRT" / "images" / "images",
        node21_csv_path=data_root / "node21" / "cxr_images" / "original_data" / "metadata.csv",
        node21_image_dir=data_root / "node21" / "cxr_images" / "original_data" / "images",
        seed=42,
        test_seed=0,
        target_size=512,
        augment_train=False,
    )
    jsrt_test = test_loaders["jsrt"]

    checkpoints = {}
    # Discover checkpoints if available
    for p in OUTPUT_DIR.glob("**/best_model.pth"):
        name = p.parent.name
        checkpoints[name] = p

    if not checkpoints:
        print(f"No best_model.pth checkpoints found in {OUTPUT_DIR}")
        return

    results = {}
    for name, ckpt_path in checkpoints.items():
        results[name] = analyse_checkpoint(name, ckpt_path, jsrt_test)

    out_path = OUTPUT_DIR / "results" / "per_nodule_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
