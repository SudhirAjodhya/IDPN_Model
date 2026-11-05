"""
What the resize step costs a nodule, in pixels, before any learning happens.

This is the motivating figure: a lesion that is plainly visible on the native film is
reduced to a handful of pixels by the time the detector's backbone sees it. One column
per working resolution, each crop shown at the same display size with nearest-neighbour
interpolation so the surviving pixel grid is literally countable.

Both datasets are supported, because they make the point at different scales:

  node21   .mha films, 1024-4280 px, nodule boxes from metadata.csv
  vindr    .dicom films, 2128-3408 px, Nodule/Mass boxes from train.csv

Physical size comes from the file's own header, SimpleITK spacing for NODE21 and
PixelSpacing for VinDr, so the millimetre figures are read rather than assumed.

CAVEAT ON NODE21 SPACING. 922 of its 1,476 nodule annotations sit on films that were
downscaled without their spacing being updated, which puts the film's physical width
below 250 mm - anatomically impossible for a chest. Those films are excluded from
ranking and their millimetre figures are suppressed. VinDr carries original DICOM
spacing throughout and has no such problem.

The resize mirrors the resizer's own geometric skip exactly: `F.interpolate` with
`mode="bilinear"`, `align_corners=False` and `antialias=False` (see
src/models/downsamplers/talebi.py), and squashes per axis rather than letterboxing,
which is what both datasets do by default. Using a different kernel here would
illustrate a resize the detector never performs.

Film selection. `--rank` scores candidate lesions by how visible they are on the native
film, so the illustrated case is one a reader can actually see rather than one taken on
faith:

    visibility = |mean(box) - mean(surrounding ring)| / std(ring)

Pass `--film` and `--box` to pin a specific case once one has been chosen.

Usage:
    python -m scripts.eval.plot_decimation_cost --rank
    python -m scripts.eval.plot_decimation_cost --film n0189
    python -m scripts.eval.plot_decimation_cost --dataset vindr --rank
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

NODE21_IMAGE_DIR = ROOT_DIR / "data_sets" / "node21" / "cxr_images" / "original_data" / "images"
NODE21_CSV_PATH = ROOT_DIR / "data_sets" / "node21" / "cxr_images" / "original_data" / "metadata.csv"

from src.data.vindr import VINDR_ROOT                       # noqa: E402
VINDR_IMAGE_DIR = VINDR_ROOT / "train"
VINDR_CSV_PATH = VINDR_ROOT / "train.csv"

OUT_DIR = Path("outputs/consolidation")

RING_MARGIN = 2.0           # ring extends to 2x the box half-width, excluding the box itself
MIN_NATIVE_WIDTH = 60       # below this the lesion is not legible on the page at any zoom
MAX_NATIVE_WIDTH = 500      # above this it is a mass filling a lung field, not a nodule
MIN_PLAUSIBLE_FILM_MM = 250.0


# ============================================================================
# LOADING - one pair of functions per dataset, both returning the same shapes
# ============================================================================

def node21_candidates():
    """Every label==1 annotation, as (film, x, y, width, height)."""
    frame = pd.read_csv(NODE21_CSV_PATH)
    frame = frame[frame["label"] == 1]

    return [(row.img_name.replace(".mha", ""), int(row.x), int(row.y),
             int(row.width), int(row.height)) for row in frame.itertuples()]


def load_node21(film):
    """Native film as (H, W) float32 in [0, 1], plus (spacing_y, spacing_x) in mm."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(NODE21_IMAGE_DIR / (film + ".mha")))
    spacing_x, spacing_y = image.GetSpacing()[0], image.GetSpacing()[1]

    array = sitk.GetArrayFromImage(image).squeeze().astype(np.float32)

    if array.max() > array.min():
        array = (array - array.min()) / (array.max() - array.min())

    return array, (spacing_y, spacing_x)


def vindr_candidates():
    """Every Nodule/Mass box, as (image_id, x, y, width, height)."""
    frame = pd.read_csv(VINDR_CSV_PATH)
    frame = frame[(frame["class_name"] == "Nodule/Mass") & frame["x_min"].notna()]

    return [(row.image_id, int(row.x_min), int(row.y_min),
             int(row.x_max - row.x_min), int(row.y_max - row.y_min))
            for row in frame.itertuples()]


def load_vindr(image_id):
    """
    Native DICOM as (H, W) float32 in [0, 1], plus (spacing_y, spacing_x) in mm.

    Polarity handling matches src/data/vindr.py: MONOCHROME1 stores minimum as white and
    is reflected about the film's own range, so both polarities come out lungs-dark.
    """
    import pydicom

    dataset = pydicom.dcmread(str(VINDR_IMAGE_DIR / (image_id + ".dicom")))
    pixels = dataset.pixel_array.astype(np.float32)

    if str(dataset.PhotometricInterpretation) == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels

    if pixels.max() > pixels.min():
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())

    spacing = getattr(dataset, "PixelSpacing", None) or getattr(dataset, "ImagerPixelSpacing", None)
    spacing = (float(spacing[0]), float(spacing[1])) if spacing is not None else (None, None)

    return pixels, spacing


DATASETS = {
    "node21": (node21_candidates, load_node21),
    "vindr": (vindr_candidates, load_vindr),
}


# ============================================================================
# MEASUREMENT
# ============================================================================

def visibility(image, x, y, width, height):
    """
    Local contrast of the lesion against the tissue immediately around it.

    Returns:
        float, |mean inside - mean of ring| in units of the ring's own std. Higher is a
        lesion that stands out from its surroundings on the native film.
    """
    cx, cy = x + width / 2.0, y + height / 2.0
    half = max(width, height) / 2.0

    outer = int(round(half * RING_MARGIN))
    y0, y1 = int(max(cy - outer, 0)), int(min(cy + outer, image.shape[0]))
    x0, x1 = int(max(cx - outer, 0)), int(min(cx + outer, image.shape[1]))

    patch = image[y0:y1, x0:x1]

    if patch.size == 0:
        return 0.0

    # Mask off the box itself so the ring measures only the surrounding tissue
    mask = np.ones(patch.shape, dtype=bool)
    mask[int(max(y - y0, 0)):int(max(y + height - y0, 0)),
         int(max(x - x0, 0)):int(max(x + width - x0, 0))] = False

    inside, ring = patch[~mask], patch[mask]

    if inside.size == 0 or ring.size < 32 or ring.std() < 1e-6:
        return 0.0

    return float(abs(inside.mean() - ring.mean()) / ring.std())


def lesion_mm(width, height, spacing):
    """Longest side of the box in millimetres, or None when spacing is unusable."""
    if spacing[0] is None:
        return None

    return max(height * spacing[0], width * spacing[1])


def spacing_is_plausible(shape, spacing):
    """
    Whether the header's spacing can be believed.

    A chest film is roughly 300-450 mm across. A stored spacing that implies much less
    than that means the film was downscaled without the spacing being updated, which is
    true of most NODE21 films at 1024 px.
    """
    if spacing[0] is None:
        return False

    return shape[1] * spacing[1] >= MIN_PLAUSIBLE_FILM_MM


def rank_candidates(dataset, limit, max_scan, targets,
                    min_width=MIN_NATIVE_WIDTH, max_width=MAX_NATIVE_WIDTH):
    """Score candidate lesions on their native films and print the most visible ones."""
    candidates, loader = DATASETS[dataset]

    # Largest boxes first, so a truncated scan still sees the figure-worthy cases
    pool = [c for c in candidates() if min_width <= min(c[3], c[4])
            and max(c[3], c[4]) <= max_width]
    pool.sort(key=lambda c: -max(c[3], c[4]))
    pool = pool[:max_scan]

    scored = []

    for film, x, y, width, height in pool:
        try:
            image, spacing = loader(film)
        except Exception:
            continue

        if not spacing_is_plausible(image.shape, spacing):
            continue

        scored.append((visibility(image, x, y, width, height), film, x, y, width, height,
                       image.shape, spacing))

    scored.sort(reverse=True)

    header = "\n  {:<34} {:>12} {:>8} {:>8} {:>9}".format(
        "film", "native", "box px", "mm", "contrast")
    print(header + "".join("{:>10}".format("at " + str(t)) for t in targets))

    for score, film, x, y, width, height, shape, spacing in scored[:limit]:
        row = "  {:<34} {:>5} x {:<4} {:>8} {:>8.1f} {:>9.2f}".format(
            film, shape[1], shape[0], width, lesion_mm(width, height, spacing), score)

        print(row + "".join("{:>10.1f}".format(width * t / shape[1]) for t in targets))

    return scored


# ============================================================================
# FIGURE
# ============================================================================

def bilinear(image, size):
    """The resizer's geometric skip, exactly as talebi.py performs it."""
    tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear",
                            align_corners=False, antialias=False)

    return resized[0, 0].numpy()


def crop_around(image, box, native_shape, pad=1.6):
    """
    Crop centred on the lesion, in whatever resolution `image` is at.

    Scaling is per axis, because a non-square film squashed to a square input does not
    keep its aspect ratio and a single scale would place the crop off the lesion.

    Returns:
        (crop, (cx, cy, width, height)) with the box in the crop's own coordinates
    """
    x, y, width, height = box
    scale_y = image.shape[0] / float(native_shape[0])
    scale_x = image.shape[1] / float(native_shape[1])

    cx, cy = (x + width / 2.0) * scale_x, (y + height / 2.0) * scale_y
    box_width, box_height = width * scale_x, height * scale_y
    half = max(box_width, box_height) * pad

    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, image.shape[0]))
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, image.shape[1]))

    return image[y0:y1, x0:x1], (cx - x0, cy - y0, box_width, box_height)


def main():
    global OUT_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="node21")
    parser.add_argument("--rank", action="store_true", help="score candidate films and exit")
    parser.add_argument("--limit", type=int, default=25, help="how many candidates to print")
    parser.add_argument("--max-scan", type=int, default=200,
                        help="how many films to decode when ranking")
    parser.add_argument("--min-width", type=int, default=MIN_NATIVE_WIDTH,
                        help="smallest native box width to consider when ranking")
    parser.add_argument("--max-width", type=int, default=MAX_NATIVE_WIDTH,
                        help="largest native box width to consider when ranking")
    parser.add_argument("--film", default=None, help="film stem or VinDr image_id")
    parser.add_argument("--box", type=int, default=-1,
                        help="which lesion on that film; -1 picks the most visible one")
    parser.add_argument("--targets", type=int, nargs="*", default=[512, 224])
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    arguments = parser.parse_args()

    if arguments.rank:
        rank_candidates(arguments.dataset, arguments.limit, arguments.max_scan,
                        arguments.targets, arguments.min_width, arguments.max_width)
        return

    OUT_DIR = Path(arguments.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    candidates, loader = DATASETS[arguments.dataset]

    # Resolve the film, either the one asked for or the most visible lesion in the set
    if arguments.film is None:
        scored = rank_candidates(arguments.dataset, 5, arguments.max_scan, arguments.targets,
                                 arguments.min_width, arguments.max_width)
        _, film, x, y, width, height, _, _ = scored[0]
    else:
        film = arguments.film.replace(".mha", "").replace(".dicom", "")
        matches = [c for c in candidates() if c[0] == film]

        if not matches:
            raise SystemExit("no annotated lesion on film {}".format(film))

        if arguments.box >= 0:
            film, x, y, width, height = matches[arguments.box]
        else:
            # Several readers annotate the same film in VinDr, and they disagree on
            # extent. Without an explicit choice, take the box that stands out most.
            probe, _ = loader(film)
            film, x, y, width, height = max(matches, key=lambda c: visibility(probe, *c[1:]))

    image, spacing = loader(film)
    box = (x, y, width, height)

    millimetres = lesion_mm(width, height, spacing)
    trusted = spacing_is_plausible(image.shape, spacing)

    columns = 2 + len(arguments.targets)
    figure, axes = plt.subplots(1, columns, figsize=(3.05 * columns, 3.5))

    # Panel one: the whole film, with the lesion marked
    axes[0].imshow(image, cmap="gray")
    axes[0].add_patch(patches.Rectangle((x, y), width, height, fill=False,
                                        edgecolor="#FF8C00", lw=1.6))
    axes[0].set_title("Full film\n{} $\\times$ {} px{}".format(
        image.shape[1], image.shape[0],
        ", {:.3f} mm/px".format(spacing[1]) if trusted else ""), fontsize=10)

    # Remaining panels: the same crop at native resolution and at each working resolution
    panels = [(None, image)] + [(size, bilinear(image, size)) for size in arguments.targets]

    for column, (size, source) in enumerate(panels, start=1):
        crop, (cx, cy, box_width, box_height) = crop_around(source, box, image.shape)

        if size is None:
            label = "Native"
            detail = "nodule {} px{}".format(
                width, " ({:.1f} mm)".format(millimetres) if trusted else "")
        else:
            label = "Bilinear to {}".format(size)
            detail = "nodule {:.1f} px{}".format(
                box_width, " ({:.2f} mm/px)".format(spacing[1] * image.shape[1] / size)
                if trusted else "")

        axis = axes[column]
        axis.imshow(crop, cmap="gray", interpolation="nearest")
        axis.add_patch(patches.Rectangle((cx - box_width / 2, cy - box_height / 2),
                                         box_width, box_height, fill=False,
                                         edgecolor="#FF8C00", lw=1.6, ls="--"))
        axis.set_title("{}\n{}".format(label, detail), fontsize=10)

    for axis in axes:
        axis.set_xticks([]); axis.set_yticks([])

    figure.tight_layout()

    path = OUT_DIR / "fig01_decimation_cost_{}_{}.png".format(arguments.dataset, film[:12])
    figure.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote {}".format(path))

    print("\n  native film      {} x {} px".format(image.shape[1], image.shape[0]))
    print("  nodule, native   {} px wide, contrast {:.2f}{}".format(
        width, visibility(image, x, y, width, height),
        ", {:.1f} mm".format(millimetres) if trusted else ", spacing not trustworthy"))

    for size in arguments.targets:
        print("  nodule at {:<6}  {:.1f} px wide ({:.1f}x decimation)".format(
            size, width * size / image.shape[1], image.shape[1] / size))


if __name__ == "__main__":
    main()
