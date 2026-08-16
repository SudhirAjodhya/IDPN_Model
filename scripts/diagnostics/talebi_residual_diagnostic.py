"""
talebi_residual_diagnostic.py

Tests the hypothesis: does the paper-faithful Talebi residual concentrate on
edges (ribs, lung outlines, vessels) rather than nodules?

For each Talebi checkpoint x each test image:
  residual = |talebi_out - bilinear_out|
  edges    = Sobel(bilinear_out)
Reports (mean +/- std across images):
  edge_corr    = pearson(|residual|, |edges|)         high -> tracks edges
  nodule_ratio = mean|residual|_inside_nodule
                 / mean|residual|_outside            >1 -> attends to nodules
                                                     <1 -> avoids nodules
                                                     ~1 -> uniform / noise

Run:
  python scripts/talebi_residual_diagnostic.py
  python scripts/talebi_residual_diagnostic.py --max-images 30    # quick check
  python scripts/talebi_residual_diagnostic.py --demo             # self-check only
"""

import sys
from pathlib import Path
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import sobel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.detector import NoduleDetector
from src.data.splits import build_three_way_split

DATA_DIR = ROOT / "data_sets"
JSRT_IMAGE_DIR   = DATA_DIR / "JSRT" / "images" / "images"
JSRT_CSV_PATH    = DATA_DIR / "JSRT" / "jsrt_metadata.csv"
NODE21_IMAGE_DIR = DATA_DIR / "node21" / "cxr_images" / "original_data" / "images"
NODE21_CSV_PATH  = DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv"

INPUT_SIZE  = 2048
TARGET_SIZE = 384

CHECKPOINTS = [
    ("outputs/v8/2026-08-13_improved_v1/node21_talebi_centernet_focal_v5_s42/best_model.pth",  "node21_talebi",  42),
    ("outputs/v8/2026-08-13_improved_v1/node21_talebi_centernet_focal_v5_s7/best_model.pth",   "node21_talebi",   7),
    ("outputs/v8/2026-08-14_improved_v1/node21_talebi_centernet_focal_v5_s123/best_model.pth", "node21_talebi", 123),
]


# ============================================================
# metrics
# ============================================================

def sobel_mag(x_2d):
    gx = sobel(x_2d, axis=1)
    gy = sobel(x_2d, axis=0)
    return np.sqrt(gx * gx + gy * gy)


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# ============================================================
# per-checkpoint analysis
# ============================================================

def analyze_ckpt(ckpt_path, variant, seed, device, max_images=None):
    model = NoduleDetector(variant=variant, head="centernet", target_size=TARGET_SIZE).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=JSRT_CSV_PATH,
        jsrt_image_dir=JSRT_IMAGE_DIR,
        node21_csv_path=NODE21_CSV_PATH,
        node21_image_dir=NODE21_IMAGE_DIR,
        seed=seed,
        test_seed=0,
    )
    ds = test_loaders["combined"]                                          # ponytail: skip DataLoader/collate, index dataset directly
    scale = TARGET_SIZE / INPUT_SIZE

    edge_corrs, nodule_ratios, edge_ratios = [], [], []
    n_seen = 0

    with torch.no_grad():
        for i in range(len(ds)):
            if max_images is not None and n_seen >= max_images:
                break
            img, target = ds[i]
            img = img.unsqueeze(0).float().to(device)

            talebi = model.ldm(img).cpu().numpy()[0, 0]
            bilin  = F.interpolate(img, size=(TARGET_SIZE, TARGET_SIZE),
                                   mode="bilinear", align_corners=False).cpu().numpy()[0, 0]
            resid  = np.abs(talebi - bilin)
            edges  = sobel_mag(bilin)
            edge_corrs.append(pearson(resid, edges))

            if bool(target["has_nodule"]):
                cx = float(target["center_2048"][0]) * scale
                cy = float(target["center_2048"][1]) * scale
                r  = max(float(target["radius_px"]) * scale, 2.0)          # floor: nodules tiny at 384
                ys, xs = np.ogrid[:TARGET_SIZE, :TARGET_SIZE]
                mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (r * r)
                if mask.any() and (~mask).any():
                    r_in,  r_out = float(resid[mask].mean()), float(resid[~mask].mean())
                    e_in,  e_out = float(edges[mask].mean()), float(edges[~mask].mean())
                    if r_out > 0: nodule_ratios.append(r_in / r_out)
                    if e_out > 0: edge_ratios.append(e_in / e_out)                     # null model: is smoothness geometric?

            n_seen += 1

    return {
        "ckpt": str(ckpt_path),
        "variant": variant,
        "seed": seed,
        "n_images": n_seen,
        "n_positive": len(nodule_ratios),
        "edge_corr_mean": float(np.mean(edge_corrs)),
        "edge_corr_std":  float(np.std(edge_corrs)),
        "nodule_ratio_mean": float(np.mean(nodule_ratios)) if nodule_ratios else None,
        "nodule_ratio_std":  float(np.std(nodule_ratios))  if nodule_ratios else None,
        "edge_ratio_mean":   float(np.mean(edge_ratios))   if edge_ratios   else None,
        "edge_ratio_std":    float(np.std(edge_ratios))    if edge_ratios   else None,
    }


# ============================================================
# self-check
# ============================================================

def demo():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 64))
    b = rng.normal(size=(64, 64))
    assert abs(pearson(a, b)) < 0.2, f"pearson iid self-check failed: {pearson(a, b)}"
    assert abs(pearson(a, a) - 1.0) < 1e-9, "pearson identity self-check failed"
    assert sobel_mag(np.ones((32, 32))).max() < 1e-9, "sobel of constant should be 0"
    # sobel of a step should be non-zero
    step = np.zeros((32, 32)); step[:, 16:] = 1.0
    assert sobel_mag(step).max() > 0.5, "sobel of step should fire"
    print("demo() OK")


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", type=str, default="outputs/talebi_residual_diagnostic.json")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        demo(); return

    demo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    results = []
    for rel, variant, seed in CHECKPOINTS:
        ckpt = ROOT / rel
        if not ckpt.exists():
            print(f"[skip] {ckpt} missing"); continue
        print(f"[run ] {variant} s{seed}")
        r = analyze_ckpt(ckpt, variant, seed, device, max_images=args.max_images)
        nr = r["nodule_ratio_mean"]
        print(f"       edge_corr    = {r['edge_corr_mean']:+.3f} +/- {r['edge_corr_std']:.3f}")
        er = r["edge_ratio_mean"]
        print(f"       nodule_ratio = {nr:.3f} +/- {r['nodule_ratio_std']:.3f}"
              if nr is not None else "       nodule_ratio = n/a")
        print(f"       edge_ratio   = {er:.3f} +/- {r['edge_ratio_std']:.3f}   (null: geometric smoothness)"
              if er is not None else "       edge_ratio   = n/a")
        print(f"       (n={r['n_images']}, pos={r['n_positive']})")
        results.append(r)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # verdict
    if results:
        ecs = [r["edge_corr_mean"] for r in results]
        nrs = [r["nodule_ratio_mean"] for r in results if r["nodule_ratio_mean"] is not None]
        print("\n=== summary across checkpoints ===")
        print(f"edge_corr    mean={np.mean(ecs):+.3f}  range=[{min(ecs):+.3f}, {max(ecs):+.3f}]")
        ers = [r["edge_ratio_mean"] for r in results if r["edge_ratio_mean"] is not None]
        if nrs:
            print(f"nodule_ratio mean={np.mean(nrs):.3f}  range=[{min(nrs):.3f}, {max(nrs):.3f}]")
        if ers:
            print(f"edge_ratio   mean={np.mean(ers):.3f}  range=[{min(ers):.3f}, {max(ers):.3f}]")
            if nrs:
                nr_over_er = np.mean(nrs) / np.mean(ers)
                print(f"nodule_ratio / edge_ratio = {nr_over_er:.3f}   (>1: residual over-covers nodules vs baseline smoothness; <1: under-covers)")
        print("\ninterpretation:")
        print("  edge_ratio ~ nodule_ratio -> residual just tracks input smoothness (geometric, not Talebi-specific)")
        print("  edge_ratio ~ 1 but nodule_ratio < 1 -> residual actively avoids nodules")
        print("  nodule_ratio > 1.2 -> residual concentrates on nodules")


if __name__ == "__main__":
    main()
