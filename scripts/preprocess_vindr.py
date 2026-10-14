"""
Convert the VinDr-CXR DICOM archive to losslessly compressed 16-bit PNG.

Why this exists: 37% of the archive is JPEG2000-encoded and decodes at a median of
651 ms per film against 28 ms for the uncompressed majority. Weighted, that is roughly
65 minutes of pure decode per epoch on a single core, or 3-7 extra hours per training
run at 23-50 epochs. Converting once costs the same 65 minutes and every subsequent run
reads it for free.

Nothing here discards image information:
  - Native resolution is preserved. No resize, so the full decimation ratio to target
    resolution (roughly 12x at target 224) survives for the resizer to be tested on.
  - PNG16 is losslessly compressed.
  - Raw stored pixel values are kept. No VOI LUT and no windowing is applied, since
    both are lossy and the dataset classes min-max normalise per image anyway.
  - MONOCHROME1 films are inverted, which is a bijective mapping.

Bit depth varies across the archive (12-bit 65%, 14-bit 22%, 16-bit 10%, 10-bit 2.5%),
so the true depth is recorded in the sidecar manifest rather than being baked into the
pixels.

Usage:
    python -m scripts.preprocess_vindr --source D:/.../train --out D:/.../train_png16
    python -m scripts.preprocess_vindr --source ... --out ... --limit 20      # pilot
    python -m scripts.preprocess_vindr --source ... --out ... --verify 25     # check
"""

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pydicom

# One implementation, shared with the dataset's own DICOM fallback path, so the two
# cannot drift apart on polarity or bit-depth handling.
from src.data.vindr import load_dicom_pixels

# NOTE: this script is now optional. scripts/warm_vindr_cache.py builds the per-input-size
# cache straight from DICOM, and that cache is both smaller than this PNG archive (18 GB at
# input 1024 for the 8,788-film subset, against 124 GB here) and skips CLAHE as well as
# decode. Keep this script for regenerating a full-resolution archive when one is wanted.


def convert_one(job: tuple[Path, Path]) -> dict | None:
    """
    Convert a single film, skipping work if the output already exists.

    Args:
        job: (source .dicom path, destination .png path)

    Returns:
        A manifest row, or None if the film could not be decoded.
    """
    dicom_path, png_path = job

    # Resuming a partial run must not re-decode what is already on disk.
    if png_path.exists():
        try:
            existing = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)

            if existing is not None:
                # Header-only read. It skips the expensive pixel decode, which is the
                # whole point of the cache, but still records the true bit depth and
                # polarity, so a resumed or piloted run leaves a manifest
                # indistinguishable from one clean pass.
                header = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
                photometric = str(header.PhotometricInterpretation)

                return {"image_id": dicom_path.stem,
                        "rows": existing.shape[0], "cols": existing.shape[1],
                        "bits_stored": int(getattr(header, "BitsStored", 16)),
                        "photometric": photometric,
                        "inverted": int(photometric == "MONOCHROME1")}

        except Exception:
            pass

    try:
        pixels, metadata = load_dicom_pixels(dicom_path)

    except Exception as error:
        sys.stderr.write(f"FAILED {dicom_path.name}: {error}\n")
        return None

    # Write to a temporary name and rename, so an interrupted run never leaves a
    # truncated PNG that a later pass would happily treat as complete.
    temporary_path = png_path.with_suffix(f".tmp{png_path.suffix}")
    cv2.imwrite(str(temporary_path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    temporary_path.replace(png_path)

    metadata["image_id"] = dicom_path.stem

    return metadata


def verify_roundtrip(source_directory: Path, output_directory: Path, count: int) -> int:
    """
    Re-decode a random sample of films and assert the PNGs match them exactly.

    This is the check that the conversion is genuinely lossless rather than assumed
    to be. Returns the number of mismatches found.

    Args:
        source_directory: directory of original .dicom files
        output_directory: directory of converted .png files
        count: how many films to sample

    Returns:
        Number of films whose PNG did not match a fresh decode.
    """
    rng = np.random.default_rng(0)
    png_paths = sorted(output_directory.glob("*.png"))

    if not png_paths:
        print("Nothing to verify: no PNGs found.")
        return 0

    sample = rng.choice(png_paths, size=min(count, len(png_paths)), replace=False)
    mismatches = 0

    for png_path in sample:
        expected, _ = load_dicom_pixels(source_directory / f"{Path(png_path).stem}.dicom")
        actual = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)

        if actual is None or actual.shape != expected.shape or not np.array_equal(actual, expected):
            sys.stderr.write(f"MISMATCH {Path(png_path).name}\n")
            mismatches += 1

    print(f"Verified {len(sample)} films: {len(sample) - mismatches} exact, {mismatches} mismatched.")

    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="directory of .dicom files")
    parser.add_argument("--out", required=True, type=Path, help="destination for .png files")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="convert only N films (pilot run)")
    parser.add_argument("--verify", type=int, default=0, help="verify N converted films and exit")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)

    if arguments.verify:
        sys.exit(1 if verify_roundtrip(arguments.source, arguments.out, arguments.verify) else 0)

    dicom_paths = sorted(arguments.source.glob("*.dicom"))

    if arguments.limit:
        dicom_paths = dicom_paths[:arguments.limit]

    jobs = [(path, arguments.out / f"{path.stem}.png") for path in dicom_paths]
    print(f"Converting {len(jobs)} films with {arguments.workers} workers -> {arguments.out}")

    manifest_rows = []

    # imap_unordered keeps memory flat and lets progress reflect real completion order.
    with Pool(arguments.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(convert_one, jobs, chunksize=8), start=1):
            if row is not None:
                manifest_rows.append(row)

            if index % 250 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)}", flush=True)

    # The manifest records the true bit depth and polarity per film, which the pixels
    # themselves no longer carry once written as PNG.
    manifest_path = arguments.out.parent / f"{arguments.out.name}_manifest.csv"
    fields = ["image_id", "rows", "cols", "bits_stored", "photometric", "inverted"]

    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: r["image_id"]))

    converted_bytes = sum(path.stat().st_size for path in arguments.out.glob("*.png"))
    print(f"Done. {len(manifest_rows)} films, {converted_bytes / 1e9:.1f} GB, manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
