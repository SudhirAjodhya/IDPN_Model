"""
Visual comparison of what two resizers actually hand the backbone.

The whole thesis question is whether a learned resizer preserves nodule information that
bilinear discards. Every measurement of that so far has been a scalar - sensitivity,
residual enrichment - and none of them show WHERE the two differ or WHAT the difference
looks like. This makes that visible.

Panels produced:

  Per film (--n-films of them, nodule-bearing, drawn from the fixed NODE21 test split)
    input at input_size with GT boxes | resizer A output | resizer B output
    | signed difference B-A | nodule crops from A, B and the difference

  Aggregate
    mean |B-A| over all sampled films, which shows whether the difference is structured
      or diffuse
    pixel intensity histograms for A and B, which shows whether B alters dynamic range
      rather than placing detail
    difference magnitude inside GT boxes against matched control boxes - the visual form
      of the residual enrichment number, where 1.0 means the resizer treats a nodule like
      any other patch

Note on interpretation: when A is bilinear, B - A IS the learned residual, because the
Talebi family computes bilinear(x) + residual(x). The signed difference panel is therefore
a direct picture of what the residual adds and where.

Usage:
    python -m scripts.eval.compare_resizer_outputs --a <bilinear_run> --b <talebi_run>
    python -m scripts.eval.compare_resizer_outputs --a ... --b ... --n-films 6
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.splits import build_three_way_split          # noqa: E402
from src.models.detector import NoduleDetector             # noqa: E402
import scripts.train as T                                  # noqa: E402

DPI = 300
CONTROLS_PER_BOX = 8          # matches scripts/diagnostics/residual_attribution.py


def load_model(run_dir: Path, device):
    """Rebuild a run's model from its metrics.json and checkpoint."""
    metrics = json.load(open(run_dir / "metrics.json"))

    model = NoduleDetector(variant=metrics["variant"], head=metrics["head"],
                           target_size=metrics["target_size"],
                           backbone=metrics.get("backbone", "convnext_fpn")).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    model.eval()

    return model, metrics


@torch.no_grad()
def resizer_output(model, image, device):
    """Return the resizer's output for one image as a (H, W) float32 array."""
    with autocast(device_type=device.type, dtype=torch.bfloat16,
                  enabled=device.type == "cuda"):
        output = model.get_ldm_output(image.to(device))

    return output.float().cpu().squeeze().numpy()


def scale_boxes(boxes_cxcywh, num_boxes, input_size, target_size):
    """Convert (cx, cy, w, h) boxes from input_size space to target_size corners."""
    scale = target_size / input_size
    corners = []

    for j in range(int(num_boxes)):
        cx, cy, w, h = (float(v) * scale for v in boxes_cxcywh[j])
        corners.append((cx - w / 2, cy - h / 2, w, h))

    return corners


def control_boxes(box, size, rng, n=CONTROLS_PER_BOX):
    """
    Same-sized boxes at random non-overlapping positions in the same film.

    This is the matched null the enrichment ratio is scored against: it controls for box
    size and for the film's own intensity statistics, so a ratio above 1.0 means the
    resizer treats nodules differently from ordinary patches of identical size.
    """
    x, y, w, h = box
    out = []

    for _ in range(n * 12):
        if len(out) >= n:
            break

        cx = rng.uniform(0, max(1, size - w))
        cy = rng.uniform(0, max(1, size - h))

        # Reject anything overlapping the true box, or an already-placed control.
        if abs(cx - x) < w and abs(cy - y) < h:
            continue

        if any(abs(cx - ox) < w and abs(cy - oy) < h for ox, oy, _, _ in out):
            continue

        out.append((cx, cy, w, h))

    return out


def patch_mean(array, box):
    """Mean absolute value of `array` inside an (x, y, w, h) box, clipped to bounds."""
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(array.shape[1], x + max(1, w)), min(array.shape[0], y + max(1, h))

    if x1 <= x0 or y1 <= y0:
        return None

    return float(np.abs(array[y0:y1, x0:x1]).mean())


def figure_per_film(records, label_a, label_b, out_path):
    """One row per film: input, both resizer outputs, signed difference, nodule crops."""
    n = len(records)
    figure, axes = plt.subplots(n, 6, figsize=(21, 3.4 * n), squeeze=False)

    for row, rec in enumerate(records):
        limit = np.abs(rec["diff"]).max()

        axes[row][0].imshow(rec["input"], cmap="gray")
        axes[row][0].set_title(f"input {rec['input'].shape[0]}px" if row == 0 else "", fontsize=10)

        for (bx, by, bw, bh) in rec["boxes_input"]:
            axes[row][0].add_patch(patches.Rectangle((bx, by), bw, bh, fill=False,
                                                     edgecolor="lime", linewidth=1.2))

        # SHARED intensity scale across A and B. Letting imshow autoscale each panel to its
        # own min/max makes a resizer that widens its output range look flatter than the
        # baseline, which is a display artifact and the opposite of what is happening.
        shared_min = min(rec["a"].min(), rec["b"].min())
        shared_max = max(rec["a"].max(), rec["b"].max())

        axes[row][1].imshow(rec["a"], cmap="gray", vmin=shared_min, vmax=shared_max)
        axes[row][1].set_title(f"{label_a}  [{rec['a'].min():.2f}, {rec['a'].max():.2f}]"
                               if row == 0 else f"[{rec['a'].min():.2f}, {rec['a'].max():.2f}]",
                               fontsize=9)

        axes[row][2].imshow(rec["b"], cmap="gray", vmin=shared_min, vmax=shared_max)
        axes[row][2].set_title(f"{label_b}  [{rec['b'].min():.2f}, {rec['b'].max():.2f}]"
                               if row == 0 else f"[{rec['b'].min():.2f}, {rec['b'].max():.2f}]",
                               fontsize=9)

        # Diverging map on a symmetric scale, so added and removed signal are readable
        # separately rather than collapsed by an absolute value.
        im = axes[row][3].imshow(rec["diff"], cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[row][3].set_title(f"{label_b} - {label_a}" if row == 0 else "", fontsize=10)
        plt.colorbar(im, ax=axes[row][3], fraction=0.046)

        for (bx, by, bw, bh) in rec["boxes_target"]:
            for column in (1, 2, 3):
                axes[row][column].add_patch(patches.Rectangle(
                    (bx, by), bw, bh, fill=False, edgecolor="lime", linewidth=1.0))

        if rec["crop"] is not None:
            crop_a, crop_b, crop_d = rec["crop"]
            # Shared scale here too, for the same reason as the full panels above.
            axes[row][4].imshow(np.hstack([crop_a, crop_b]), cmap="gray",
                                vmin=min(crop_a.min(), crop_b.min()),
                                vmax=max(crop_a.max(), crop_b.max()))
            axes[row][4].set_title(f"nodule: {label_a} | {label_b}" if row == 0 else "", fontsize=9)

            crop_limit = np.abs(crop_d).max() or 1e-8
            axes[row][5].imshow(crop_d, cmap="RdBu_r", vmin=-crop_limit, vmax=crop_limit)
            axes[row][5].set_title("nodule difference" if row == 0 else "", fontsize=9)
        else:
            axes[row][4].axis("off")
            axes[row][5].axis("off")

        for column in range(6):
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])

        axes[row][0].set_ylabel(rec["name"][:18], fontsize=8)

    figure.suptitle(f"What the backbone receives: {label_a} vs {label_b}", fontsize=13)
    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def figure_aggregate(records, label_a, label_b, out_path):
    """Mean difference map, intensity histograms, and the nodule-vs-control comparison."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    mean_abs = np.mean([np.abs(r["diff"]) for r in records], axis=0)
    im = axes[0].imshow(mean_abs, cmap="magma")
    axes[0].set_title(f"mean |{label_b} - {label_a}|  (n={len(records)} films)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    values_a = np.concatenate([r["a"].ravel() for r in records])
    values_b = np.concatenate([r["b"].ravel() for r in records])
    bins = np.linspace(min(values_a.min(), values_b.min()),
                       max(values_a.max(), values_b.max()), 120)

    axes[1].hist(values_a, bins=bins, alpha=0.55, label=label_a, density=True)
    axes[1].hist(values_b, bins=bins, alpha=0.55, label=label_b, density=True)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("pixel intensity"); axes[1].set_ylabel("density (log)")
    axes[1].set_title("intensity distribution of the resized film")
    axes[1].legend(fontsize=9)

    nodule_values = [v for r in records for v in r["nodule_diff"]]
    control_values = [v for r in records for v in r["control_diff"]]

    if nodule_values and control_values:
        enrichment = float(np.mean(nodule_values) / np.mean(control_values))
        edges = np.linspace(0, max(max(nodule_values), max(control_values)), 40)

        axes[2].hist(control_values, bins=edges, alpha=0.6, density=True,
                     label=f"matched controls (n={len(control_values)})")
        axes[2].hist(nodule_values, bins=edges, alpha=0.6, density=True,
                     label=f"GT nodules (n={len(nodule_values)})")
        axes[2].axvline(np.mean(control_values), color="C0", linestyle="--", linewidth=1)
        axes[2].axvline(np.mean(nodule_values), color="C1", linestyle="--", linewidth=1)
        axes[2].set_xlabel(f"mean |{label_b} - {label_a}| within box")
        axes[2].set_title(f"residual enrichment = {enrichment:.3f}\n"
                          f"(1.0 = nodules treated like any patch)")
        axes[2].legend(fontsize=8)
    else:
        axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)

    return mean_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, type=Path, help="baseline run (usually bilinear)")
    parser.add_argument("--b", required=True, type=Path, help="learned-resizer run")
    parser.add_argument("--n-films", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("outputs/results"))
    parser.add_argument("--seed", type=int, default=0, help="control-box placement seed")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, metrics_a = load_model(arguments.a, device)
    model_b, metrics_b = load_model(arguments.b, device)

    assert metrics_a["input_size"] == metrics_b["input_size"], "runs must share input_size"
    assert metrics_a["target_size"] == metrics_b["target_size"], "runs must share target_size"

    input_size, target_size = metrics_a["input_size"], metrics_a["target_size"]
    label_a, label_b = metrics_a["variant"], metrics_b["variant"]

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=input_size, seed=42, test_seed=0,
        val_frac=0.20, group_by_image=True)

    loader = DataLoader(test_loaders["node21"], batch_size=1, shuffle=False, num_workers=0)
    rng = np.random.default_rng(arguments.seed)
    records = []

    for images, targets in loader:
        if int(targets["num_boxes"][0]) == 0:
            continue

        out_a = resizer_output(model_a, images, device)
        out_b = resizer_output(model_b, images, device)
        difference = out_b - out_a

        boxes_target = scale_boxes(targets["boxes"][0], targets["num_boxes"][0],
                                   input_size, target_size)
        boxes_input = scale_boxes(targets["boxes"][0], targets["num_boxes"][0],
                                  input_size, input_size)

        # Enrichment inputs: |difference| inside each GT box against matched controls.
        nodule_diff, control_diff = [], []

        for box in boxes_target:
            value = patch_mean(difference, box)

            if value is None:
                continue

            nodule_diff.append(value)
            control_diff.extend(v for v in (patch_mean(difference, c)
                                            for c in control_boxes(box, target_size, rng))
                                if v is not None)

        crop = None

        if boxes_target:
            x, y, w, h = boxes_target[0]
            pad = max(8, int(max(w, h)))
            x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
            x1 = min(target_size, int(x + w) + pad)
            y1 = min(target_size, int(y + h) + pad)

            if x1 > x0 and y1 > y0:
                crop = (out_a[y0:y1, x0:x1], out_b[y0:y1, x0:x1], difference[y0:y1, x0:x1])

        records.append({
            "name": Path(targets["filename"][0]).stem,
            "input": images[0, 0].numpy(),
            "a": out_a, "b": out_b, "diff": difference,
            "boxes_target": boxes_target, "boxes_input": boxes_input,
            "nodule_diff": nodule_diff, "control_diff": control_diff, "crop": crop,
        })

        if len(records) >= arguments.n_films:
            break

    if not records:
        raise SystemExit("no nodule-bearing films found in the test split")

    tag = f"{label_a}_vs_{label_b}_i{input_size}_t{target_size}"
    per_film = arguments.out / f"resizer_comparison_{tag}.png"
    aggregate = arguments.out / f"resizer_aggregate_{tag}.png"

    figure_per_film(records, label_a, label_b, per_film)
    mean_abs = figure_aggregate(records, label_a, label_b, aggregate)

    nodule_values = [v for r in records for v in r["nodule_diff"]]
    control_values = [v for r in records for v in r["control_diff"]]

    print(f"\n  films            {len(records)}")
    print(f"  mean |B-A|       {mean_abs.mean():.6f}   max {mean_abs.max():.6f}")
    print(f"  intensity range  {label_a} [{records[0]['a'].min():.3f}, {records[0]['a'].max():.3f}]"
          f"   {label_b} [{records[0]['b'].min():.3f}, {records[0]['b'].max():.3f}]")

    if nodule_values and control_values:
        print(f"  nodule |B-A|     {np.mean(nodule_values):.6f}  (n={len(nodule_values)})")
        print(f"  control |B-A|    {np.mean(control_values):.6f}  (n={len(control_values)})")
        print(f"  enrichment       {np.mean(nodule_values)/np.mean(control_values):.3f}")

    print(f"\n  wrote {per_film}")
    print(f"  wrote {aggregate}\n")


if __name__ == "__main__":
    main()
