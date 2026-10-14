"""Qualitative overlays for v3 native-1024 comparison.

Pairs: bilinear input=1024 target=224 vs Talebi zeroinit input=1024 target=224
(the strongest paired Talebi win, +0.089 on NODE21). Both variants get identical
native NODE21 input, do their own 1024->224 downsample, backbone at 224 features.

Same 2x3 panel layout as save_qualitative_overlays.py (v2 canonical version).
Outputs to outputs/results_v3_node21/plots/qualitative/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.detector import NoduleDetector
from src.data.splits import build_three_way_split

OUT = ROOT / "outputs" / "results_v3_node21" / "plots" / "qualitative"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_SIZE  = 1024                                                                  # native NODE21
TARGET_SIZE = 224
STRAT_CSV   = ROOT / "outputs" / "stratification" / "node21_nodule_sizes.csv"

BILINEAR_CKPT = ROOT / "outputs/v8/2026-08-17_v3_native_1024/bilinear_centernet_focal_v5_s42_i1024_t224/best_model.pth"
ZEROINIT_CKPT = ROOT / "outputs/v8/2026-08-17_v3_native_1024/node21_talebi_zeroinit_centernet_focal_v5_s42_i1024_t224/best_model.pth"


def load_model(variant, ckpt):
    m = NoduleDetector(variant=variant, head="centernet", target_size=TARGET_SIZE, backbone="convnext_fpn")
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


def pick_representative_images(test_ds, k=6):
    strat = pd.read_csv(STRAT_CSV)
    strat_by_stem = {Path(f).stem: b for f, b in zip(strat["img_name"], strat["bin"])}
    positives = []
    for idx in range(len(test_ds)):
        _, target = test_ds[idx]
        if not target.get("has_nodule", False): continue
        stem = Path(target["filename"]).stem
        bin_ = strat_by_stem.get(stem, "?")
        positives.append((idx, bin_, target["filename"]))
    picks = []
    for want_bin in ("S", "M", "L"):
        picks.extend([p for p in positives if p[1] == want_bin][:max(1, k // 3)])
    return picks[:k]


def overlay_hm(ax, base_img, hm, alpha=0.35, cmap="hot"):
    ax.imshow(base_img, cmap="gray", vmin=0, vmax=1)
    if hm is not None:
        hm_up = F.interpolate(torch.from_numpy(hm)[None, None].float(),
                              size=base_img.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(hm_up, cmap=cmap, alpha=alpha, vmin=0, vmax=hm_up.max() if hm_up.max() > 0 else 1)


def draw_gt_box(ax, target, from_input_size, to_size):
    scale = to_size / from_input_size
    cx = target["center_2048"][0] * scale
    cy = target["center_2048"][1] * scale
    r  = max(target["radius_px"] * scale, 3.0)
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="lime", linewidth=2))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading models...")
    bilin_model = load_model("bilinear", BILINEAR_CKPT).to(device)
    zero_model  = load_model("node21_talebi_zeroinit", ZEROINIT_CKPT).to(device)

    print(f"building test split at input_size={INPUT_SIZE}, datasets=node21...")
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path   = ROOT / "data_sets/JSRT/jsrt_metadata.csv",
        jsrt_image_dir  = ROOT / "data_sets/JSRT/images/images",
        node21_csv_path = ROOT / "data_sets/node21/cxr_images/original_data/metadata.csv",
        node21_image_dir= ROOT / "data_sets/node21/cxr_images/original_data/images",
        seed=42, test_seed=0, datasets="node21", input_size=INPUT_SIZE,
    )
    test_ds = test_loaders["node21"]

    picks = pick_representative_images(test_ds, k=6)
    print(f"picked {len(picks)} images: " + ", ".join(f"{fn}[{b}]" for _, b, fn in picks))

    with torch.no_grad():
        for i, (idx, bin_, fname) in enumerate(picks):
            img_t, target = test_ds[idx]
            img_t = img_t.unsqueeze(0).float().to(device)

            zero_ldm  = zero_model.ldm(img_t).cpu().numpy()[0, 0]
            bilin_ldm = F.interpolate(img_t, size=(TARGET_SIZE, TARGET_SIZE),
                                       mode="bilinear", align_corners=False).cpu().numpy()[0, 0]
            residual = np.abs(zero_ldm - bilin_ldm)

            bilin_pred = torch.sigmoid(bilin_model(img_t)).cpu().numpy()[0, 0]
            zero_pred  = torch.sigmoid(zero_model(img_t)).cpu().numpy()[0, 0]

            input_np = img_t.cpu().numpy()[0, 0]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            (a_in, a_bi, a_ze), (a_re, a_bp, a_zp) = axes

            a_in.imshow(input_np, cmap="gray", vmin=0, vmax=1)
            draw_gt_box(a_in, target, INPUT_SIZE, INPUT_SIZE)
            a_in.set_title(f"Input ({INPUT_SIZE}x{INPUT_SIZE}) + GT nodule\n{fname}  [{bin_}]")

            a_bi.imshow(bilin_ldm, cmap="gray", vmin=0, vmax=1)
            draw_gt_box(a_bi, target, INPUT_SIZE, TARGET_SIZE)
            a_bi.set_title(f"Bilinear 1024→{TARGET_SIZE}")

            a_ze.imshow(zero_ldm, cmap="gray", vmin=0, vmax=1)
            draw_gt_box(a_ze, target, INPUT_SIZE, TARGET_SIZE)
            a_ze.set_title(f"Talebi zero-init 1024→{TARGET_SIZE}")

            im = a_re.imshow(residual, cmap="magma")
            draw_gt_box(a_re, target, INPUT_SIZE, TARGET_SIZE)
            a_re.set_title("|Talebi - Bilinear|  residual magnitude")
            plt.colorbar(im, ax=a_re, fraction=0.046, pad=0.04)

            overlay_hm(a_bp, bilin_ldm, bilin_pred, alpha=0.45)
            draw_gt_box(a_bp, target, INPUT_SIZE, TARGET_SIZE)
            a_bp.set_title("Bilinear model — prediction heatmap")

            overlay_hm(a_zp, zero_ldm, zero_pred, alpha=0.45)
            draw_gt_box(a_zp, target, INPUT_SIZE, TARGET_SIZE)
            a_zp.set_title("Zero-init Talebi model — prediction heatmap")

            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f"v3 native-1024 comparison  —  {fname}  (bin {bin_})", fontsize=13)
            fig.tight_layout()
            out_stem = f"overlay_{i:02d}_{bin_}_{Path(fname).stem}"
            fig.savefig(OUT / f"{out_stem}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  [{i+1}/{len(picks)}] wrote {out_stem}.png")

    print(f"\nAll overlays -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
