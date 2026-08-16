"""
train_talebi_reproduce.py
=========================
CLI entrypoint for the Talebi & Milanfar (ICCV 2021) paper reproduction
on ImageNette (10-class ImageNet subset).

Usage:
  python scripts/train_talebi_reproduce.py --smoke
  python scripts/train_talebi_reproduce.py --phase A --seed 0 --epochs 60
  python scripts/train_talebi_reproduce.py --phase B --seed 0 --epochs 60
  python scripts/train_talebi_reproduce.py --summarize

Success criterion: mean Delta top-1 (Talebi - Bilinear) over 3 seeds is
positive, larger than the seed range, and within an order of magnitude of
the paper's +1.7 pp.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.training.classification_trainer import train_one, DATA_ROOT
from src.utils.results_aggregation_talebi_reproduce import summarize


def parse_args():
    p = argparse.ArgumentParser(
        description="Talebi & Milanfar (ICCV 2021) paper reproduction on ImageNette"
    )
    p.add_argument("--phase", choices=["A", "B"],
                   help="Training phase: A (bilinear baseline) or B (Talebi resizer)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--epochs", type=int, default=60,
                   help="Number of training epochs (default: 60)")
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT),
                   help="Path to ImageNette dataset root")
    p.add_argument("--smoke", action="store_true",
                   help="Quick end-to-end sanity: Phase A (2 ep) then Phase B (2 ep)")
    p.add_argument("--summarize", action="store_true",
                   help="Aggregate all phase*_seed*.json into SUMMARY.md")
    return p.parse_args()


def main():
    args = parse_args()

    if args.summarize:
        summarize()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"cuda: {torch.cuda.get_device_name(0)}  "
              f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"ImageNette not found at {data_root}. "
            f"Expected {{train,val}}/*/... layout. Pass --data-root PATH."
        )

    if args.smoke:
        print("=== SMOKE: Phase A (2 ep, batch 8) ===")
        train_one(phase="A", seed=0, epochs=2, data_root=data_root,
                  device=device, smoke=True)
        print("\n=== SMOKE: Phase B (2 ep, batch 8, loads smoke Phase A ckpt) ===")
        train_one(phase="B", seed=0, epochs=2, data_root=data_root,
                  device=device, smoke=True)
        print("\n=== SMOKE OK ===")
        return

    if args.phase is None:
        print("Error: --phase A|B is required (or use --smoke / --summarize)",
              file=sys.stderr)
        sys.exit(1)

    train_one(phase=args.phase, seed=args.seed, epochs=args.epochs,
              data_root=data_root, device=device)


if __name__ == "__main__":
    main()