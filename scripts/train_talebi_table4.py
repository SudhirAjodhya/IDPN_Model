"""
CLI entrypoint for the Talebi & Milanfar (ICCV 2021) Table 4 reproduction.

Usage:
  python scripts/train_talebi_table4.py --smoke
  python scripts/train_talebi_table4.py --backbone resnet50 --config bilinear_224 --seed 0 --epochs 40
  python scripts/train_talebi_table4.py --backbone resnet50 --config proposed_368 --seed 0 --epochs 40
  python scripts/train_talebi_table4.py --summarize

--smoke runs the smallest bilinear config + smallest proposed config
(for both default backbones) at 2 epochs, batch 8, to validate the pipeline.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.experiments.talebi_table4 import (
    DEFAULT_BACKBONES, CONFIGS_BY_LABEL, get_config,
)
from src.training.classification_trainer import DATA_ROOT, train_one
from src.utils.results_aggregation_talebi_table4 import summarize


def parse_args():
    p = argparse.ArgumentParser(
        description="Talebi Table 4 reproduction on ImageNette"
    )
    p.add_argument("--backbone", choices=DEFAULT_BACKBONES,
                   help="Torchvision backbone name")
    p.add_argument("--config", choices=sorted(CONFIGS_BY_LABEL),
                   help="Config label from src.experiments.talebi_table4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--smoke", action="store_true",
                   help="Quick pipeline check across both backbones (2ep, batch 8)")
    p.add_argument("--summarize", action="store_true",
                   help="Aggregate all Table 4 JSONs into TABLE4.md + plots")
    return p.parse_args()


def _device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        p = torch.cuda.get_device_properties(dev)
        print(f"cuda: {p.name}  vram={p.total_memory / 1e9:.1f} GB")
    return dev


def _smoke(data_root: Path, device: torch.device):
    """Run bilinear_224 then proposed_224 for each default backbone at 2ep/b8."""
    for bb in DEFAULT_BACKBONES:
        print(f"\n=== SMOKE: {bb} bilinear_224 (2ep, b8) ===")
        train_one(backbone_name=bb, config=get_config("bilinear_224"),
                  seed=0, epochs=2, data_root=data_root, device=device,
                  smoke=True)
        print(f"\n=== SMOKE: {bb} proposed_224 (2ep, b8, loads smoke bilinear_224) ===")
        train_one(backbone_name=bb, config=get_config("proposed_224"),
                  seed=0, epochs=2, data_root=data_root, device=device,
                  smoke=True)
    print("\n=== SMOKE OK ===")


def main():
    args = parse_args()

    if args.summarize:
        summarize()
        return

    device = _device()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"ImageNette not found at {data_root}. "
            f"Expected {{train,val}}/*/... layout. Pass --data-root PATH."
        )

    if args.smoke:
        _smoke(data_root, device)
        return

    if args.backbone is None or args.config is None:
        sys.exit("--backbone and --config required (or use --smoke / --summarize)")

    train_one(
        backbone_name=args.backbone,
        config=get_config(args.config),
        seed=args.seed,
        epochs=args.epochs,
        data_root=data_root,
        device=device,
    )


if __name__ == "__main__":
    main()
