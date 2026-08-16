"""
Sanity-check stratify_nodules output
    
5 random nodules, 2 rows each column:
    top:  original CXR + bounding box
    bottom: same CXR + bounding box + diameter line highlighted
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data_sets" / "node21" / "cxr_images" / "original_data"
_STRAT_CSV = _ROOT / "outputs" / "stratification" / "node21_nodule_sizes.csv"


def load_mha(path: Path) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).squeeze().astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else img


def draw_box(ax, x, y, w, h, color="lime"):
    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"))


def draw_diameter(ax, x, y, w, h, color="red"):
    if w >= h:
        ax.plot([x, x + w], [y + h / 2, y + h / 2], color=color, lw=2)
    else:
        ax.plot([x + w / 2, x + w / 2], [y, y + h], color=color, lw=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strat-csv", type=Path, default=_STRAT_CSV)
    ap.add_argument("--node21-images", type=Path, default=_DATA / "images")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.strat_csv).sample(args.n, random_state=args.seed)

    fig, axes = plt.subplots(2, args.n, figsize=(4 * args.n, 8))
    fig.suptitle("Stratification sanity check - top: bbox  |  bottom: bbox + diameter", fontsize=12)

    for col, (_, row) in enumerate(df.iterrows()):
        img = load_mha(args.node21_images / row["img_name"])
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]
        label = f"{row['bin']}  {row['diameter_mm']:.1f} mm"

        for r in (0, 1):
            ax = axes[r, col]
            ax.imshow(img, cmap="gray", origin="upper")
            ax.axis("off")
            draw_box(ax, x, y, w, h)
            if r == 1:
                draw_diameter(ax, x, y, w, h)
            if r == 0:
                ax.set_title(row["img_name"], fontsize=7)
            else:
                ax.set_title(label, fontsize=8, color="red")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
