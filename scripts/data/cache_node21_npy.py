"""One-time Node21 MHA -> npy cache to unblock GPU on the training loop.

Reads every unique image referenced by the Node21 metadata CSV, decodes with
SimpleITK, normalises to float32 [0,1] (matching NODE21Dataset), and writes
`<image_dir>_npy/<name>.npy`. Loader auto-detects the sibling _npy dir.

Idempotent: skips images that already have a cached npy. Parallelised across
CPU cores. Run once, then delete the _npy tree if you ever change the
normalisation convention."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def convert_one(mha_path: Path, out_path: Path) -> str:
    if out_path.exists():
        return "skipped"
    img_sitk = sitk.ReadImage(str(mha_path))
    img_np = sitk.GetArrayFromImage(img_sitk).squeeze().astype(np.float32)
    if img_np.max() > img_np.min():
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    else:
        img_np = img_np - img_np.min()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, img_np, allow_pickle=False)
    return "converted"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node21-csv",    type=Path, required=True)
    ap.add_argument("--node21-images", type=Path, required=True)
    ap.add_argument("--out-dir",       type=Path, default=None,
                    help="Default: sibling '_npy' dir next to --node21-images")
    ap.add_argument("--workers",       type=int, default=6)
    args = ap.parse_args()

    out_dir = args.out_dir or args.node21_images.with_name(args.node21_images.name + "_npy")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.node21_csv, index_col=0)
    unique_names = sorted(df["img_name"].unique())
    print(f"Converting {len(unique_names)} unique MHAs -> {out_dir} using {args.workers} workers")

    tasks = [(args.node21_images / n, out_dir / f"{Path(n).stem}.npy") for n in unique_names]
    counts = {"converted": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_one, m, o): m.name for m, o in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                counts[fut.result()] += 1
            except Exception as e:
                counts["failed"] += 1
                print(f"  [fail] {futures[fut]}: {e}")
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  converted={counts['converted']}  "
                      f"skipped={counts['skipped']}  failed={counts['failed']}")

    print(f"Done. converted={counts['converted']}  skipped={counts['skipped']}  failed={counts['failed']}")
    print(f"Loader will auto-detect: {out_dir}")


if __name__ == "__main__":
    main()
