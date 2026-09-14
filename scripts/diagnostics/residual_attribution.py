"""
Does the loss actually tell the learned resizer to change pixels AT the nodules?

The standing argument for the Talebi resizer is algebraic: its output is
`bilinear(x) + residual(x)`, the residual starts near zero, and gradient descent only
moves it in directions that reduce the loss - so it should be at least as good as
bilinear. Whenever it is not, the interesting question is which link in that chain fails.

This script measures the two links nobody has looked at:

  1. GRADIENT ATTRIBUTION -- dL/d(LDM output), a spatial map with one value per output
     pixel. Because the model computes `output = bilinear + residual` and the bilinear
     branch carries no learnable parameters, dL/d(residual) IS dL/d(output). This map is
     literally the instruction the loss sends the resizer: "make this pixel bigger/smaller".
     If its energy is not concentrated on nodules, the resizer is never being asked to
     build nodule filters, and no amount of training will produce them.

  2. RESIDUAL ATTRIBUTION -- |ldm_output - bilinear| for a TRAINED checkpoint. Where the
     resizer actually ended up putting its energy.

Both are reported as an ENRICHMENT RATIO against a matched null: for every ground-truth
box, K control boxes of identical size are sampled at random non-overlapping positions in
the same image. Enrichment = (mean energy inside GT) / (mean energy inside controls).

  1.0  -> the resizer treats nodules exactly like any other patch of chest
  >1.0 -> energy concentrates on nodules
  <1.0 -> energy actively avoids nodules

The matched null is the point. An earlier version of this diagnostic compared "inside the
box" against "everything else", and everything else is a garbage bucket - ribs, mediastinum,
diaphragm, air outside the body - where residual magnitude is naturally large over
high-frequency bone and small over smooth lung. A ratio below 1 measured that way says
nothing about nodules. Same-size boxes in the same image remove that confound.

Usage:
    python scripts/diagnostics/residual_attribution.py --checkpoint <path/to/run_dir>
    python scripts/diagnostics/residual_attribution.py            # auto-picks recent runs
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

from src.data.splits import build_three_way_split
from src.evaluation.centernet_targets import make_centernet_targets
from src.evaluation.fcos_targets import make_fcos_targets
from src.losses.centernet_det import centernet_detection_loss
from src.losses.fcos import fcos_loss
from src.models.detector import NoduleDetector

N_CONTROL_BOXES = 8


def sample_control_boxes(gt_box, image_size, n_controls, rng, max_tries=40):
    """
    Boxes of identical size to `gt_box`, placed at random positions that do not overlap it.

    Matching the size is what makes the comparison fair: a larger control window would
    average over more background and dilute whatever it contains.

    Args:
        gt_box: (x1, y1, x2, y2) in pixels
        image_size: side of the square output
        n_controls: how many to sample

    Returns:
        list of (x1, y1, x2, y2) integer tuples, possibly shorter than n_controls
    """
    x1, y1, x2, y2 = gt_box
    width, height = int(round(x2 - x1)), int(round(y2 - y1))

    if width < 1 or height < 1 or width >= image_size or height >= image_size: return []

    controls = []

    for _ in range(max_tries):
        if len(controls) >= n_controls: break

        cx = rng.integers(0, image_size - width)
        cy = rng.integers(0, image_size - height)

        # Reject anything touching the ground-truth box, so the null is genuine background
        overlaps_x = cx < x2 and cx + width > x1
        overlaps_y = cy < y2 and cy + height > y1

        if overlaps_x and overlaps_y: continue

        controls.append((cx, cy, cx + width, cy + height))

    return controls


def mean_in_box(energy_map, box):
    """Mean of `energy_map` inside a pixel box, or None when the box is degenerate."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(energy_map.shape[1], x2), min(energy_map.shape[0], y2)

    if x2 <= x1 or y2 <= y1: return None

    return float(energy_map[y1:y2, x1:x2].mean())


def enrichment_for_image(energy_map, boxes, image_size, rng):
    """
    Per-nodule enrichment: mean energy inside the ground-truth box divided by the mean
    across matched control boxes in the same image.
    """
    ratios = []

    for box in boxes:
        inside = mean_in_box(energy_map, box)

        if inside is None: continue

        controls = sample_control_boxes(box, image_size, N_CONTROL_BOXES, rng)
        control_means = [m for m in (mean_in_box(energy_map, c) for c in controls) if m is not None]

        if not control_means: continue

        control_mean = float(np.mean(control_means))

        if control_mean <= 1e-12: continue

        ratios.append(inside / control_mean)

    return ratios


def measure_attribution(model, loader, device, input_size, target_size, head, max_batches=None):
    """
    Core measurement, callable on a live model. `analyse_checkpoint` wraps this for
    saved runs; `train.py --attribution` calls it directly at the end of a run.

    Returns the same dict of enrichment ratios and magnitudes.
    """
    rng = np.random.default_rng(0)
    gradient_ratios, residual_ratios = [], []
    gradient_magnitudes, residual_magnitudes = [], []

    captured = {}

    def hook(module, inputs, output):
        output.retain_grad()
        captured["ldm_output"] = output

    handle = model.ldm.register_forward_hook(hook)
    was_training = model.training
    model.eval()

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches: break

        images = images.to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)

        prediction = model(images)

        if head == "fcos":
            gt = make_fcos_targets(targets, images.shape[0], model.heatmap_size,
                                   input_size=input_size, device=device)
            loss, _ = fcos_loss(prediction, gt)
        else:
            gt = make_centernet_targets(targets, images.shape[0], model.heatmap_size,
                                        input_size=input_size, device=device)
            loss, _ = centernet_detection_loss(prediction, gt)

        loss.backward()

        ldm_output = captured["ldm_output"]
        gradient = ldm_output.grad

        if gradient is None: raise RuntimeError("no gradient captured at the LDM output")

        with torch.no_grad():
            bilinear = F.interpolate(images, size=(target_size,) * 2, mode="bilinear", align_corners=False)
            residual = (ldm_output.detach() - bilinear).abs()

        for i in range(images.shape[0]):
            n_boxes = int(targets["num_boxes"][i])

            if n_boxes == 0: continue

            scale = target_size / input_size
            raw = targets["boxes"][i, :n_boxes].float().numpy() * scale
            corners = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for cx, cy, w, h in raw]

            grad_map = gradient[i, 0].abs().float().cpu().numpy()
            res_map  = residual[i, 0].float().cpu().numpy()

            gradient_ratios += enrichment_for_image(grad_map, corners, target_size, rng)
            residual_ratios += enrichment_for_image(res_map,  corners, target_size, rng)
            gradient_magnitudes.append(float(grad_map.mean()))
            residual_magnitudes.append(float(res_map.mean()))

    handle.remove()
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    def summarise(values):
        if not values: return None, None
        return float(np.mean(values)), float(np.std(values) / np.sqrt(len(values)))

    grad_mean, grad_sem = summarise(gradient_ratios)
    res_mean, res_sem = summarise(residual_ratios)

    return {
        "n_nodules": len(gradient_ratios),
        "gradient_enrichment": grad_mean, "gradient_enrichment_sem": grad_sem,
        "residual_enrichment": res_mean, "residual_enrichment_sem": res_sem,
        "mean_gradient_magnitude": float(np.mean(gradient_magnitudes)) if gradient_magnitudes else None,
        "mean_residual_magnitude": float(np.mean(residual_magnitudes)) if residual_magnitudes else None,
    }


def analyse_checkpoint(run_dir, device, max_batches=None):
    """Run both attributions over the test split for one trained run."""
    meta = json.load(open(os.path.join(run_dir, "metrics.json")))
    variant   = meta["variant"]
    head      = meta["head"]
    input_size  = meta["input_size"]
    target_size = meta["target_size"]

    model = NoduleDetector(variant=variant, head=head, target_size=target_size,
                           backbone=meta.get("backbone") or "convnext_fpn").to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device))
    model.eval()

    import scripts.train as T
    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        seed=meta["seed"], test_seed=0, target_size=target_size, augment_train=False,
        datasets="node21", input_size=input_size, val_frac=0.20, group_by_image=True)
    loader = DataLoader(test_loaders["node21"], batch_size=4, shuffle=False, num_workers=0)

    stats = measure_attribution(model, loader, device, input_size, target_size, head, max_batches)

    return {"variant": variant, "head": head, "input_size": input_size,
            "target_size": target_size, "seed": meta["seed"], **stats}


def _demo():
    """Self-check on the enrichment machinery, using maps with known structure."""
    rng = np.random.default_rng(0)
    size = 256
    box = (100.0, 100.0, 140.0, 140.0)

    # A map that is uniformly 1.0 must give enrichment 1.0 - the null behaves
    flat = np.ones((size, size), dtype=np.float32)
    ratios = enrichment_for_image(flat, [box], size, rng)
    assert abs(np.mean(ratios) - 1.0) < 1e-6, f"flat map should give 1.0, got {np.mean(ratios)}"
    print(f"OK  uniform map -> enrichment {np.mean(ratios):.3f} (expected 1.000)")

    # A map with a hot patch exactly on the box must give a large ratio
    hot = np.ones((size, size), dtype=np.float32) * 0.1
    hot[100:140, 100:140] = 10.0
    ratios = enrichment_for_image(hot, [box], size, rng)
    assert np.mean(ratios) > 50, f"hot box should give a high ratio, got {np.mean(ratios)}"
    print(f"OK  energy on the nodule -> enrichment {np.mean(ratios):.1f} (expected >> 1)")

    # A map that is hot everywhere EXCEPT the box must give a ratio below 1
    cold = np.ones((size, size), dtype=np.float32) * 10.0
    cold[100:140, 100:140] = 0.1
    ratios = enrichment_for_image(cold, [box], size, rng)
    assert np.mean(ratios) < 0.05, f"cold box should give a low ratio, got {np.mean(ratios)}"
    print(f"OK  energy avoiding the nodule -> enrichment {np.mean(ratios):.3f} (expected << 1)")

    # Controls must match the GT box in size and must not overlap it
    controls = sample_control_boxes(box, size, 8, rng)
    assert len(controls) > 0
    for c in controls:
        assert (c[2] - c[0], c[3] - c[1]) == (40, 40), f"control size {c} != GT size"
        assert not (c[0] < box[2] and c[2] > box[0] and c[1] < box[3] and c[3] > box[1]), "control overlaps GT"
    print(f"OK  {len(controls)} control boxes: size-matched, non-overlapping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="a run directory")
    parser.add_argument("--run-group", type=str, default="v7_recipe_and_fcos")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _demo()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        run_dirs = [args.checkpoint]
    else:
        # Only variants with a residual are meaningful here; bilinear's residual is zero
        # by construction, but its GRADIENT map is still a useful reference.
        run_dirs = sorted(os.path.dirname(p) for p in
                          glob.glob(f"outputs/**/*_{args.run_group}/*/metrics.json", recursive=True))

    if not run_dirs:
        print("no runs found")
        return

    print(f"\n{'variant':<26}{'head':<15}{'res':>10}{'seed':>5}"
          f"{'GRAD enrich':>13}{'RESID enrich':>14}{'|grad|':>11}{'|resid|':>10}")
    print("-" * 104)

    for run_dir in run_dirs:
        try:
            r = analyse_checkpoint(run_dir, device, args.max_batches)
        except Exception as exc:
            print(f"  {os.path.basename(run_dir)[:40]:<40} FAILED: {exc}")
            continue

        grad = f"{r['gradient_enrichment']:.3f}+-{r['gradient_enrichment_sem']:.3f}" if r["gradient_enrichment"] else "  -  "
        res  = f"{r['residual_enrichment']:.3f}+-{r['residual_enrichment_sem']:.3f}" if r["residual_enrichment"] else "  -  "
        res_str = f"{r['input_size']}->{r['target_size']}"
        print(f"{r['variant']:<26}{r['head']:<15}{res_str:>10}{r['seed']:>6}"
              f"{grad:>18}{res:>16}{r['mean_gradient_magnitude']:>11.2e}{r['mean_residual_magnitude']:>10.2e}")

    print("\nenrichment 1.0 = treats nodules like any other same-size patch of chest")
    print("            >1  = concentrates there;  <1 = avoids them")


if __name__ == "__main__":
    main()
