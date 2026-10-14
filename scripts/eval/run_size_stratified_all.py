"""Loop-driver for eval_size_stratified over every v2 (+ backbone matrix) checkpoint.

For each checkpoint under outputs/v8/*_v2_*/, load the sibling metrics.json to
recover variant/target_size/backbone, run compute_size_stratified_sensitivity
on the NODE21 test split, and write size_stratified.json next to the checkpoint.

Skips checkpoints that already have a fresh size_stratified.json (idempotent).

Runtime: ~30-45s per checkpoint (single forward pass over 521 NODE21 test images).
~15 checkpoints -> ~10 min on GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.splits import build_three_way_split
from src.models.detector import NoduleDetector
from src.evaluation.size_stratified import (
    compute_size_stratified_sensitivity, _load_bin_lookup,
)

DATA_DIR = ROOT / "data_sets"
STRAT_CSV = ROOT / "outputs" / "stratification" / "node21_nodule_sizes.csv"

def infer_backbone_from_dirname(dirname: str) -> str:
    """Match train.py's output_dir naming: _<backbone> suffix when non-default."""
    for bb in ("convnext_fpn_scratch", "resnet50_fpn_scratch", "resnet50_fpn"):
        if dirname.endswith(f"_{bb}"):
            return bb
    return "convnext_fpn"                                                     # default


def main():
    assert STRAT_CSV.exists(), f"Missing {STRAT_CSV}. Run scripts/data/stratify_nodules.py first."
    bin_lookup = _load_bin_lookup(str(STRAT_CSV))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enumerate every checkpoint under v8/*_v2_*/ (skip improved_v1 legacy dirs)
    checkpoints = sorted((ROOT / "outputs" / "v8").glob("*_v?_*/*/best_model.pth"))    # v2, v3, ...
    print(f"Found {len(checkpoints)} v2/v3 checkpoints")

    # Build test loader ONCE (NODE21 split is fixed by test_seed=0, ignores training seed)
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=DATA_DIR / "JSRT" / "jsrt_metadata.csv",
        jsrt_image_dir=DATA_DIR / "JSRT" / "images" / "images",
        node21_csv_path=DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv",
        node21_image_dir=DATA_DIR / "node21" / "cxr_images" / "original_data" / "images",
        seed=42, test_seed=0, datasets="node21",
    )
    test_dl = DataLoader(test_loaders["node21"], batch_size=2, shuffle=False, num_workers=2)

    for ckpt in checkpoints:
        out_json = ckpt.with_name("size_stratified.json")
        if out_json.exists():
            print(f"[skip] {ckpt.parent.name} (already has size_stratified.json)")
            continue

        metrics_json = ckpt.with_name("metrics.json")
        if not metrics_json.exists():
            print(f"[skip] {ckpt.parent.name} (no metrics.json — likely failed run)"); continue
        meta = json.loads(metrics_json.read_text())
        variant = meta["variant"]
        head    = meta["head"]
        backbone = infer_backbone_from_dirname(ckpt.parent.name)

        # target_size is not stored in metrics.json historically; derive from heatmap_size (target = 4 * heatmap)
        target_size = meta["heatmap_size"] * 4

        print(f"[run ] {ckpt.parent.name}  variant={variant} target={target_size} backbone={backbone}")
        model = NoduleDetector(variant=variant, head=head, target_size=target_size, backbone=backbone).to(device)
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        except Exception as e:
            print(f"       [fail] load_state_dict: {e}"); continue
        model.eval()

        # v2 runs used input_size=2048 (default). If we ever add resolution-sweep runs
        # with different input_size, extend metrics.json to record it and read here.
        result = compute_size_stratified_sensitivity(model, test_dl, bin_lookup, device, input_size=2048)
        result["meta"] = {
            "checkpoint":  str(ckpt.relative_to(ROOT)),
            "variant":     variant,
            "head":        head,
            "backbone":    backbone,
            "target_size": target_size,
            "seed":        meta.get("seed"),
            "run_group":   meta.get("run_group"),
        }
        out_json.write_text(json.dumps(result, indent=2))
        per = result["per_bin"]
        print(f"       S={per['S']['sensitivity']:.3f} (n={per['S']['n_nodules']})  "
              f"M={per['M']['sensitivity']:.3f} (n={per['M']['n_nodules']})  "
              f"L={per['L']['sensitivity']:.3f} (n={per['L']['n_nodules']})  "
              f"macro={result['macro']:.3f}  micro={result['micro']:.3f}")


if __name__ == "__main__":
    main()
