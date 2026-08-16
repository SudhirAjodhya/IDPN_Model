"""Post-hoc size-stratified evaluation for a trained checkpoint on Node21 test split.

Reads the bin lookup from `outputs/stratification/node21_nodule_sizes.csv`
(produced by scripts/stratify_nodules.py). Emits per-bin + macro + micro
sensitivity as JSON and prints a compact table.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import build_three_way_split
from src.models.detector import NoduleDetector
from src.evaluation.size_stratified import (
    compute_size_stratified_sensitivity, _load_bin_lookup,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
DATA_DIR   = ROOT_DIR / "data_sets"

DEFAULTS = {
    "jsrt_csv":       DATA_DIR / "JSRT" / "jsrt_metadata.csv",
    "jsrt_images":    DATA_DIR / "JSRT" / "images" / "images",
    "node21_csv":     DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv",
    "node21_images":  DATA_DIR / "node21" / "cxr_images" / "original_data" / "images",
    "stratification": ROOT_DIR / "outputs" / "stratification" / "node21_nodule_sizes.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",     type=Path, required=True)
    ap.add_argument("--variant",        type=str,  required=True,
                    help="Downsampler variant name (e.g. bilinear, node21_talebi)")
    ap.add_argument("--head",           type=str,  default="centernet")
    ap.add_argument("--stratification", type=Path, default=DEFAULTS["stratification"])
    ap.add_argument("--test-seed",      type=int,  default=0)
    ap.add_argument("--seed",           type=int,  default=42)
    ap.add_argument("--batch-size",     type=int,  default=2)
    ap.add_argument("--input-size",     type=int,  default=2048)
    ap.add_argument("--out",            type=Path, default=None,
                    help="JSON output path (default: sibling of checkpoint)")
    args = ap.parse_args()

    if not args.stratification.exists():
        raise FileNotFoundError(
            f"Stratification CSV not found at {args.stratification}. "
            "Run scripts/stratify_nodules.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loaders, split_info = build_three_way_split(
        jsrt_csv_path=DEFAULTS["jsrt_csv"], jsrt_image_dir=DEFAULTS["jsrt_images"],
        node21_csv_path=DEFAULTS["node21_csv"], node21_image_dir=DEFAULTS["node21_images"],
        seed=args.seed, test_seed=args.test_seed, datasets="node21",
    )
    test_dl = DataLoader(test_loaders["node21"], batch_size=args.batch_size,
                         shuffle=False, num_workers=2)

    model = NoduleDetector(variant=args.variant, head=args.head).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    bin_lookup = _load_bin_lookup(str(args.stratification))
    result = compute_size_stratified_sensitivity(model, test_dl, bin_lookup, device,
                                                 input_size=args.input_size)
    result["meta"] = {
        "checkpoint": str(args.checkpoint),
        "variant":    args.variant,
        "date":       date.today().isoformat(),
        "split_info": split_info,
        "stratification_csv": str(args.stratification),
    }

    out_path = args.out or args.checkpoint.with_name(f"size_stratified_{args.variant}.json")
    out_path.write_text(json.dumps(result, indent=2))

    per = result["per_bin"]
    print(f"\nSize-stratified sensitivity - {args.variant}")
    print(f"  S (<10mm):    {per['S']['sensitivity']!s:>8}  (n={per['S']['n_nodules']})")
    print(f"  M (10-20mm):  {per['M']['sensitivity']!s:>8}  (n={per['M']['n_nodules']})")
    print(f"  L (>=20mm):   {per['L']['sensitivity']!s:>8}  (n={per['L']['n_nodules']})")
    print(f"  macro:        {result['macro']}")
    print(f"  micro:        {result['micro']}")
    print(f"  unbinned:     {result['unbinned']} nodules not found in stratification CSV")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
