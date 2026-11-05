"""
Populate the VinDr post-CLAHE, post-resize cache before a sweep runs.

Why this exists: uncached, a VinDr film costs ~370 ms per epoch (PNG/DICOM decode 47%,
CLAHE at native resolution 33%, normalisation 14%), against NODE21's 2.7 ms. That left
the GPU idle roughly 80% of the time and put a pooled run at ~19 min/epoch. Cached, a
film loads in ~14 ms.

The cache also supersedes the intermediate PNG16 archive: it is smaller (18 GB at input
1024 for the 8,788-film subset, against 124 GB for the full PNG set) and skips CLAHE as
well as decode. Films are read straight from the original DICOM when no PNG is present.

It works by iterating the real VinDrDataset through a DataLoader, so the cached pixels
are byte-identical to what training would have computed. There is no second
implementation of the load path to drift out of sync.

Run it once per (input_size, letterbox, subset) combination the sweep will use.

Usage:
    python -m scripts.warm_vindr_cache --input-size 1024 --negative-ratio 1.0
    python -m scripts.warm_vindr_cache --input-size 2048 --negative-ratio 1.0 --workers 12
"""

import argparse
import shutil
import time
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.vindr import VINDR_ROOT, VinDrDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=VINDR_ROOT)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--negative-ratio", type=float, default=None,
                        help="box-free films kept per box-bearing film; omit to keep all")
    parser.add_argument("--letterbox", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    # augment=False: the cache holds the pre-augmentation film. Augmentation runs after
    # the cache read in __getitem__, so every epoch still sees a different draw.
    dataset = VinDrDataset(
        annotations_csv=arguments.root / "train.csv",
        images_root=arguments.root / "train_png16",
        dicom_root=arguments.root / "train",
        input_size=arguments.input_size,
        letterbox=arguments.letterbox,
        negative_ratio=arguments.negative_ratio,
        augment=False,
    )

    if dataset.cache_dir is None:
        raise SystemExit("Cache is disabled for this configuration (insufficient disk). "
                         "Free space or lower --input-size.")

    already = len(list(dataset.cache_dir.glob("*.npy"))) if dataset.cache_dir.exists() else 0
    print(f"Warming {dataset.cache_dir} | {len(dataset)} films, {already} already cached")

    # batch_size=None yields single samples without collation, which is all that is needed
    # to trigger the cache write inside __getitem__ and avoids stacking cost.
    loader = DataLoader(dataset, batch_size=None, num_workers=arguments.workers, shuffle=False)

    started = time.perf_counter()

    for index, _ in enumerate(loader, start=1):
        if index % 500 == 0 or index == len(dataset):
            rate = index / (time.perf_counter() - started)
            remaining = (len(dataset) - index) / rate / 60
            print(f"  {index}/{len(dataset)}  {rate:.1f} films/s  ~{remaining:.1f} min left",
                  flush=True)

    cached_bytes = sum(path.stat().st_size for path in dataset.cache_dir.glob("*.npy"))
    elapsed = (time.perf_counter() - started) / 60

    print(f"Done in {elapsed:.1f} min. Cache {cached_bytes / 1e9:.1f} GB at {dataset.cache_dir}")
    print(f"D: free {shutil.disk_usage(arguments.root.drive or arguments.root).free / 1e9:.0f} GB")


if __name__ == "__main__":
    main()
