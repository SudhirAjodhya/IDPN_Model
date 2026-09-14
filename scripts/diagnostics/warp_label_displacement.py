"""
How far does the learned warp move a nodule away from the cell its label marks?

`fcos_targets.py:71` places supervision by a pure linear rescale, `boxes * (heatmap/input)`.
That is exact for any linear resize - bilinear and the whole additive-residual family, where
output pixel i always samples input coordinate (i + 0.5) * S / T - 0.5. It is NOT exact for
`WarpResizer`, which samples through `identity + tanh(offsets) * max_offset`, a non-linear
and image-dependent coordinate map. The nodule's pixels move; the label does not move with
them.

This script measures the size of that mismatch, and it needs no inversion of the warp.
`grid_sample`'s grid is a BACKWARD map: `grid[i, j]` is the input coordinate that output
cell (i, j) reads from. So take the cell the label calls positive - the linear one,
`centre * scale` - look up what input coordinate that cell actually sampled, and measure how
far it is from the nodule centre the label meant.

    displacement = | grid[linear_cell] - true_nodule_centre |

Reported in target cells and as a fraction of each nodule's own diameter, because a 5-cell
error on a 40-cell mass is trivial and the same error on a 10-cell nodule is fatal.

Reading the result:
    ratio well under ~0.25   the label still lands solidly inside the lesion; the coordinate
                             mismatch is not load-bearing and the warp null stands clean
    ratio around 0.5 or more supervision is landing off the lesion for a large share of
                             nodules, the warp arm has never been scored fairly, and every
                             warp number needs an asterisk until targets are built through
                             the grid

Pure diagnostic: loads a checkpoint, runs one forward pass, writes nothing.

Usage:
    python scripts/diagnostics/warp_label_displacement.py --run-dir outputs/v8/<group>/<run>
    python scripts/diagnostics/warp_label_displacement.py --all
    python scripts/diagnostics/warp_label_displacement.py --demo
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# build_three_way_split and NoduleDetector are imported inside run_one: they pull in
# SimpleITK and the full data stack, which --demo does not need and which is not present
# on every machine that might want to check the grid arithmetic.
try:
    from src.models.downsamplers.adaptive import WarpResizer

except ImportError:
    # src/models/__init__.py imports the backbones, and those need torchvision. The grid
    # arithmetic below does not. The failing chain still gets as far as registering the
    # downsamplers, so the module is already in sys.modules - take it from there rather
    # than re-executing it, which would double-register every variant.
    _adaptive = sys.modules.get("src.models.downsamplers.adaptive")

    if _adaptive is None: raise

    WarpResizer = _adaptive.WarpResizer

ROOT_DIR         = Path(__file__).resolve().parents[2]
DATA_DIR         = ROOT_DIR / "data_sets"
JSRT_IMAGE_DIR   = DATA_DIR / "JSRT" / "images" / "images"
JSRT_CSV_PATH    = DATA_DIR / "JSRT" / "jsrt_metadata.csv"
NODE21_IMAGE_DIR = DATA_DIR / "node21" / "cxr_images" / "original_data" / "images"
NODE21_CSV_PATH  = DATA_DIR / "node21" / "cxr_images" / "original_data" / "metadata.csv"


def find_warp(model):
    """The WarpResizer inside a detector, or None if this variant has no warp."""
    for module in model.modules():
        if isinstance(module, WarpResizer): return module

    return None


def sampling_grid(warp, images):
    """
    The backward sampling grid the warp would use for these images, (B, T, T, 2) as (x, y)
    in grid_sample's [-1, 1] convention. Mirrors WarpResizer.forward exactly.
    """
    batch_size = images.shape[0]

    control = F.interpolate(images, size=(warp.control_size,) * 2,
                            mode="bilinear", align_corners=False)
    offsets = warp.offset_net(control)
    offsets = F.interpolate(offsets, size=(warp.target_size,) * 2,
                            mode="bilinear", align_corners=False)
    offsets = torch.tanh(offsets).permute(0, 2, 3, 1) * warp.max_offset

    return warp._identity_grid(batch_size, images.device, images.dtype) + offsets


def normalised_to_pixels(value, size):
    """grid_sample [-1, 1] back to input pixels, align_corners=False."""
    return (value + 1.0) * size / 2.0 - 0.5


@torch.no_grad()
def measure(model, loader, device, input_size, target_size):
    """Per-nodule displacement between the labelled cell and the content it samples."""
    warp = find_warp(model)

    if warp is None: return None

    model.eval()
    scale = target_size / input_size
    displacements_cells, diameters_cells, ratios = [], [], []
    offset_magnitudes = []

    for images, targets in loader:
        images = images.to(device)
        grid = sampling_grid(warp, images).float().cpu()               # (B, T, T, 2)

        # Raw displacement of the whole field, for context on how hard the warp is working
        identity = warp._identity_grid(images.shape[0], torch.device("cpu"), torch.float32)
        field = (grid - identity).abs()
        offset_magnitudes.append((field.norm(dim=-1) * (target_size / 2.0)).flatten().numpy())

        for i in range(images.shape[0]):
            n_objects = int(targets["num_boxes"][i])

            if n_objects == 0: continue

            boxes = targets["boxes"][i, :n_objects].float()             # (n, 4) cx,cy,w,h input px

            for j in range(n_objects):
                centre_x, centre_y, width, height = [float(v) for v in boxes[j]]

                # The cell the linear label rule marks positive
                cell_x = int(np.clip(round(centre_x * scale - 0.5), 0, target_size - 1))
                cell_y = int(np.clip(round(centre_y * scale - 0.5), 0, target_size - 1))

                # What input coordinate that cell actually samples
                sampled = grid[i, cell_y, cell_x]                       # (x, y) normalised
                sampled_x = normalised_to_pixels(float(sampled[0]), input_size)
                sampled_y = normalised_to_pixels(float(sampled[1]), input_size)

                displacement_px = float(np.hypot(sampled_x - centre_x, sampled_y - centre_y))
                displacement_cells = displacement_px * scale
                diameter_cells = max(width, height) * scale

                displacements_cells.append(displacement_cells)
                diameters_cells.append(diameter_cells)

                if diameter_cells > 0: ratios.append(displacement_cells / diameter_cells)

    if not displacements_cells: return None

    displacements = np.array(displacements_cells)
    diameters     = np.array(diameters_cells)
    ratios        = np.array(ratios)
    field_all     = np.concatenate(offset_magnitudes)

    return {
        "n_nodules":            len(displacements),
        "displacement_cells":   {"mean": float(displacements.mean()),
                                 "median": float(np.median(displacements)),
                                 "p95": float(np.percentile(displacements, 95)),
                                 "max": float(displacements.max())},
        "nodule_diameter_cells": {"mean": float(diameters.mean()),
                                  "median": float(np.median(diameters))},
        "ratio_to_diameter":    {"mean": float(ratios.mean()),
                                 "median": float(np.median(ratios)),
                                 "p95": float(np.percentile(ratios, 95)),
                                 "frac_over_half": float((ratios > 0.5).mean())},
        "field_displacement_cells": {"mean": float(field_all.mean()),
                                     "max": float(field_all.max())},
        # The bound the warp could reach if the offset head saturated everywhere
        "max_possible_cells":   warp.max_offset * target_size / 2.0,
    }


def report(name, result):
    if result is None:
        print(f"  {name}: no WarpResizer in this model (linear resize, labels are exact)")
        return

    d, r = result["displacement_cells"], result["ratio_to_diameter"]
    print(f"\n  {name}")
    print(f"    nodules                     {result['n_nodules']}")
    print(f"    nodule diameter (cells)     median {result['nodule_diameter_cells']['median']:.1f}")
    print(f"    label displacement (cells)  median {d['median']:.2f}  mean {d['mean']:.2f}  "
          f"p95 {d['p95']:.2f}  max {d['max']:.2f}")
    print(f"    displacement / diameter     median {r['median']:.3f}  mean {r['mean']:.3f}  "
          f"p95 {r['p95']:.3f}")
    print(f"    nodules displaced > 0.5 dia {r['frac_over_half'] * 100:.1f}%")
    print(f"    whole-field displacement    mean {result['field_displacement_cells']['mean']:.2f} "
          f"cells, max {result['field_displacement_cells']['max']:.2f} "
          f"(bound {result['max_possible_cells']:.1f})")

    median_ratio = r["median"]

    if median_ratio < 0.25:
        print(f"    => small relative to nodule size; the coordinate mismatch is unlikely to")
        print(f"       be load-bearing, and the warp null stands as measured")
    elif median_ratio < 0.5:
        print(f"    => MATERIAL. Supervision is landing near the lesion edge for half the")
        print(f"       nodules. Worth building targets through the grid and retraining.")
    else:
        print(f"    => SEVERE. The median nodule is labelled more than half a diameter from")
        print(f"       where its pixels actually are. The warp arm has not been scored fairly.")


def run_one(run_dir, device):
    from src.data.splits import build_three_way_split
    from src.models.detector import NoduleDetector

    run = json.load(open(os.path.join(run_dir, "metrics.json")))

    model = NoduleDetector(variant=run["variant"], head=run["head"],
                           target_size=run["target_size"],
                           backbone=run.get("backbone", "convnext_fpn")).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"),
                                     map_location=device))

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=JSRT_CSV_PATH, jsrt_image_dir=JSRT_IMAGE_DIR,
        node21_csv_path=NODE21_CSV_PATH, node21_image_dir=NODE21_IMAGE_DIR,
        seed=run["seed"], test_seed=run.get("test_seed", 0),
        target_size=run["target_size"], augment_train=False, datasets="node21",
        input_size=run["input_size"], val_frac=0.20,
        aug_profile=run.get("aug_profile", "v3"), group_by_image=True,
        letterbox=bool(run.get("letterbox")))

    loader = DataLoader(test_loaders["node21"], batch_size=4, shuffle=False, num_workers=0)

    return measure(model, loader, device, run["input_size"], run["target_size"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--all",     action="store_true",
                        help="Every fcos warp run on disk.")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arguments.all:
        run_dirs = sorted({os.path.dirname(p) for p in
                           glob.glob("outputs/**/metrics.json", recursive=True)
                           if "warp" in os.path.basename(os.path.dirname(p))
                           and os.path.exists(os.path.join(os.path.dirname(p), "best_model.pth"))})
    elif arguments.run_dir:
        run_dirs = [str(arguments.run_dir)]
    else:
        parser.error("pass --run-dir or --all")

    print(f"measuring label displacement on {len(run_dirs)} run(s), device {device}")

    for run_dir in run_dirs:
        try:
            report(os.path.basename(run_dir), run_one(run_dir, device))
        except Exception as exc:
            print(f"  {os.path.basename(run_dir)}: FAILED - {exc}")

    print()


def _demo():
    """Assert-based self-check with a warp whose displacement is known analytically."""
    target_size, input_size = 384, 1024
    warp = WarpResizer(target_size=target_size, max_offset=0.05)
    warp.eval()

    images = torch.zeros(1, 1, input_size, input_size)

    # Zero-init offset head -> exact identity grid -> zero displacement everywhere
    grid = sampling_grid(warp, images)
    identity = warp._identity_grid(1, images.device, images.dtype)
    assert torch.allclose(grid, identity, atol=1e-6), "zero-init warp must be the identity grid"

    # Saturate the offset head so tanh -> 1 and the shift is exactly max_offset in x and y
    with torch.no_grad():
        warp.offset_net[-1].bias.fill_(50.0)

    grid = sampling_grid(warp, images)
    shift_normalised = float((grid - identity)[0, target_size // 2, target_size // 2, 0])
    assert abs(shift_normalised - 0.05) < 1e-4, shift_normalised

    # A normalised shift of max_offset is max_offset * T / 2 target cells
    expected_cells = 0.05 * target_size / 2.0
    shift_cells = shift_normalised * target_size / 2.0
    assert abs(shift_cells - expected_cells) < 1e-3, (shift_cells, expected_cells)

    # And the same shift converted through input pixels must agree
    centre = input_size / 2.0
    sampled = normalised_to_pixels(
        float(grid[0, target_size // 2, target_size // 2, 0]), input_size)
    identity_px = normalised_to_pixels(
        float(identity[0, target_size // 2, target_size // 2, 0]), input_size)
    assert abs((sampled - identity_px) * target_size / input_size - expected_cells) < 1e-2

    # A model with no warp must be reported as such rather than crashing
    class _Plain(torch.nn.Module):
        def modules(self): return iter([self])
    assert find_warp(_Plain()) is None

    print(f"ok  (saturated warp shifts {expected_cells:.1f} target cells at t{target_size})")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        main()
