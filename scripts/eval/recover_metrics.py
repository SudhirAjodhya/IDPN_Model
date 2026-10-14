"""
Rebuild metrics.json for runs whose training finished but whose post-processing crashed.

Three separate optional steps - the LDM comparison figure, the FROC sweep and the
attribution pass - have each raised before metrics.json was written, leaving a directory
with best_model.pth and no results. train.py now wraps all three, but runs already on disk
still need recovering, and retraining them would waste hours for numbers that are fully
determined by the saved checkpoint.

Configuration is read back from the directory name, which encodes
    {variant}_{head}_{loss}_s{seed}[_{backbone}][_i{in}_t{out}][_recipe tags]

Usage:
    python scripts/eval/recover_metrics.py --run-group v8_adaptive_resizers
    python scripts/eval/recover_metrics.py --run-group v8_adaptive_resizers --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

import scripts.train as T
from src.data.splits import build_three_way_split
from src.evaluation.node21_detection_metrics import compute_box_detection_metrics, froc_auc
from src.models.detector import NoduleDetector

NAME = re.compile(
    r"^(?P<variant>.+?)_(?P<head>centernet_det|fcos|centernet_hr|centernet)_(?P<loss>focal_v5|focal|mse|qfl)"
    r"_s(?P<seed>\d+)(?P<rest>.*)$"
)


def parse_run_dir(name):
    """Recover the configuration a run was launched with from its directory name."""
    m = NAME.match(name)

    if not m: return None

    rest = m.group("rest")
    resolution = re.search(r"_i(\d+)_t(\d+)", rest)

    config = {
        "variant": m.group("variant"),
        "head": m.group("head"),
        "loss": m.group("loss"),
        "seed": int(m.group("seed")),
        "input_size": int(resolution.group(1)) if resolution else 2048,
        "target_size": int(resolution.group(2)) if resolution else 512,
        "letterbox": "_lb" in rest,
        "freeze_backbone_epochs": int(f.group(1)) if (f := re.search(r"_fz(\d+)", rest)) else 0,
        "aug_profile": a.group(1) if (a := re.search(r"_aug(v\d)", rest)) else "v3",
        "backbone": "convnext_fpn",
    }

    for known in ("resnet50_fpn_scratch", "resnet50_fpn", "convnext_fpn_scratch"):
        if known in rest: config["backbone"] = known

    return config


def recover(run_dir, device, dry_run=False):
    name = os.path.basename(os.path.normpath(run_dir))
    config = parse_run_dir(name)

    if config is None:
        return f"SKIP {name}: cannot parse configuration from the directory name"

    checkpoint = os.path.join(run_dir, "best_model.pth")

    if not os.path.exists(checkpoint):
        return f"SKIP {name}: no best_model.pth, training never completed"

    if dry_run:
        return f"WOULD RECOVER {name}: {config['variant']}/{config['head']} s{config['seed']} " \
               f"{config['input_size']}->{config['target_size']}" + (" letterbox" if config["letterbox"] else "")

    model = NoduleDetector(variant=config["variant"], head=config["head"],
                           target_size=config["target_size"], backbone=config["backbone"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    _, _, test_loaders, split_info = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        seed=config["seed"], test_seed=0, target_size=config["target_size"],
        augment_train=False, datasets="node21", input_size=config["input_size"],
        val_frac=0.20, aug_profile=config["aug_profile"], group_by_image=True,
        letterbox=config["letterbox"])

    loader = DataLoader(test_loaders["node21"], batch_size=4, shuffle=False, num_workers=0)

    box_metrics = {}
    for iou in (0.25, 0.5):
        m = compute_box_detection_metrics(model, loader, device=device,
                                          input_size=config["input_size"], iou_threshold=iou)
        m["froc_auc"] = froc_auc(m["froc"])
        box_metrics[f"iou{iou}"] = m

    # Attribution is best-effort: bilinear has no residual, so its enrichment is undefined
    attribution = None

    try:
        from src.evaluation.residual_attribution_hook import run_attribution
        attribution = run_attribution(model, loader, device, input_size=config["input_size"],
                                      target_size=config["target_size"], head=config["head"])
    except Exception as exc:
        attribution = {"error": str(exc)}

    primary = box_metrics["iou0.25"]
    payload = {
        "recovered": True,
        "recovered_note": "rebuilt from best_model.pth after post-processing crashed; "
                          "training history and val curves are NOT recoverable",
        "attribution": attribution,
        "box_metrics": box_metrics,
        "test_results": {"combined": primary, "node21": primary, "jsrt": {}},
        "split_info": split_info,
        **config,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=4)

    enrich = (attribution or {}).get("residual_enrichment")
    enrich_text = "n/a" if enrich is None else f"{enrich:.3f}"

    return (f"RECOVERED {name}: sens {primary['sensitivity']:.3f} at {primary['fp_per_image']:.2f} FP, "
            f"AP {primary['average_precision']:.3f}, FROC {primary['froc_auc']:.3f}, enrich {enrich_text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orphans = [os.path.dirname(p) for p in
               glob.glob(f"outputs/**/*_{args.run_group}/*/best_model.pth", recursive=True)
               if not os.path.exists(os.path.join(os.path.dirname(p), "metrics.json"))]

    if not orphans:
        print(f"no orphaned runs in '{args.run_group}' - every completed run has metrics.json")
        return

    print(f"found {len(orphans)} run(s) with a checkpoint but no metrics.json\n")

    for run_dir in sorted(orphans):
        print(" ", recover(run_dir, device, args.dry_run))


if __name__ == "__main__":
    main()
