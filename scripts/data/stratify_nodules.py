"""Emit per-nodule longest-diameter (mm) + LUNA16 size bin for Node21.
Reads pixel spacing from each MHA via SimpleITK (the training loader
discards it). Run once, review counts, commit the CSV."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk

from src.data.nodule_sizes import (SMALL_NODULE_SIZE_MAX, MEDIUM_NODULE_SIZE_MAX, BINS, bin_assignment, longest_diameter_mm,)


def spacing_mm(mha_path: Path) -> float:
    """
    Longest-in-plane spacing from the MHA header, in mm. Node21 CXRs are 2-D
    SimpleITK returns (sx, sy)  take x since Node21 is isotropic in practice. 
    ponytail: assumes isotropic, if that assumption breaks, use max(sx, sy) to match the max(w,h) diameter convention
    """

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(mha_path))
    reader.ReadImageInformation()
    return float(reader.GetSpacing()[0])


_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data_sets" / "node21" / "cxr_images" / "original_data"


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--node21-csv", type=Path, default=_DATA / "metadata.csv")
    argument_parser.add_argument("--node21-images", type=Path, default=_DATA / "images", help="Directory of .mha files referenced by img_name column")
    argument_parser.add_argument("--out-dir", type=Path, default=Path("outputs/stratification"))
    arguments = argument_parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    # read csv
    df = pd.read_csv(arguments.node21_csv, index_col=0)
    df = df[df["label"] == 1].copy() # read only positive rows, since negative rows have no nodules to parse in

    # Cache spacing per image one read per unique .mha not per row
    unique_images = df["img_name"].unique()
    spacings = {name: spacing_mm(arguments.node21_images / name) for name in unique_images}

    # Adding spacing, diameter and binnigs
    df["spacing_mm"] = df["img_name"].map(spacings)
    df["diameter_mm"] = [longest_diameter_mm(w, h, s) for w, h, s in zip(df["width"], df["height"], df["spacing_mm"])]
    df["bin"] = df["diameter_mm"].map(bin_assignment)

    # out csv format
    out = df[["img_name", "x", "y", "width", "height", "spacing_mm", "diameter_mm", "bin"]]
    out.to_csv(arguments.out_dir / "node21_nodule_sizes.csv", index=False)

    # count cxrs per noduel bin
    counts = Counter(df["bin"])
    print(f"Node21 positives: n={len(df)}  " +"  ".join(f"{bin}={counts.get(bin, 0)}" for bin in BINS))


    # Plot of hist binnig for nodule stratifications
    fig, axes = plt.subplots(figsize=(8, 4))
    axes.hist(df["diameter_mm"], bins=40, color="steelblue", alpha=0.8)

    for x in (SMALL_NODULE_SIZE_MAX, MEDIUM_NODULE_SIZE_MAX):
        axes.axvline(x, ls="--", color="k", lw=0.8)

    axes.set(xlabel="Longest-diameter (mm)", ylabel="Count", title=f"Node21 nodule diameters - LUNA16 cutoffs (n={len(df)})")
    fig.tight_layout()
    fig.savefig(arguments.out_dir / "node21_diameter_hist.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()