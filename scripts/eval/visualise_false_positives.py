"""
Show what a false positive actually IS, and where the FP/img number comes from.

FP/img is the x-axis of every FROC curve in this project and the budget every headline
number is read at, but it is an aggregate - it never shows a single box. This draws the
boxes on real test films so the definition is visible rather than asserted:

    green   ground-truth nodule
    blue    TRUE POSITIVE   - a predicted box overlapping a GT nodule at IoU >= 0.25
    red     FALSE POSITIVE  - a predicted box overlapping NO ground-truth nodule
    (a green box with no blue on it is a FALSE NEGATIVE - a missed nodule)

THE ARITHMETIC. Every kept box is either a TP or an FP; there is no "true negative box",
which is why detection uses FROC rather than ROC. FP/img is simply

    FP / img  =  (total red boxes over the whole test split) / (number of images)

with the DENOMINATOR being all 517 test films - including the 374 that contain no nodule at
all. Those clean films cannot produce a TP, only FPs, so they are what the budget mostly
buys. A model is read "at 0.25 FP/img" when its score threshold is tuned so that total
comes to 0.25 x 517 ~ 129 red boxes across the split.

Panels are chosen to make the definition unambiguous: films where the nodule was found,
films where it was missed, and NODULE-FREE films where every box drawn is by construction a
false positive.

Run:
  python scripts/eval/visualise_false_positives.py --run <run_dir>
  python scripts/eval/visualise_false_positives.py --run <run_dir> --budget 0.25 --n-films 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.splits import build_three_way_split                            # noqa: E402
from src.evaluation.fcos_targets import decode_fcos                          # noqa: E402
from src.evaluation.centernet_targets import decode_detections               # noqa: E402
from src.evaluation.node21_detection_metrics import _match, _truth_boxes_for  # noqa: E402
from src.models.detector import NoduleDetector                               # noqa: E402
from scripts.eval.paired_nodule_test import threshold_for_budget             # noqa: E402
import scripts.train as T                                                    # noqa: E402

OUT_DIR = Path("outputs/figures")
IOU = 0.25

COLOUR_TRUTH = "#22C55E"
COLOUR_TP    = "#2563EB"
COLOUR_FP    = "#DC2626"


def to_corners(box):
    """Boxes are stored cx, cy, w, h; matplotlib wants a bottom-left corner plus extents."""
    cx, cy, width, height = [float(v) for v in box]

    return cx - width / 2, cy - height / 2, width, height


@torch.no_grad()
def cache_predictions(run_dir, loader, device):
    """Run inference once, keeping boxes, scores, truth and the image for every film."""
    metrics = json.load(open(run_dir / "metrics.json"))
    input_size = metrics["input_size"]

    model = NoduleDetector(variant=metrics["variant"], head=metrics["head"],
                           target_size=metrics["target_size"],
                           backbone=metrics.get("backbone", "convnext_fpn")).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    model.eval()

    cached = []

    for images, targets in loader:
        images_gpu = images.to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16,
                      enabled=device.type == "cuda"):
            output = model(images_gpu)

        if getattr(model, "is_fcos", False):
            boxes_batch, scores_batch = decode_fcos(
                output["classification"].float(), output["ltrb"].float(),
                output["centerness"].float(), input_size=input_size, top_k=20)
        else:
            boxes_batch, scores_batch = decode_detections(
                output["heatmap"].float(), output["size_wh"].float(),
                output["offset"].float(), input_size=input_size, top_k=20)

        boxes_batch, scores_batch = boxes_batch.cpu(), scores_batch.cpu()

        for i in range(images.shape[0]):
            truth, _ = _truth_boxes_for(targets, i, input_size)
            cached.append({
                "boxes":  boxes_batch[i],
                "scores": scores_batch[i],
                "truth":  truth,
                "image":  images[i, 0].numpy(),
            })

    del model
    torch.cuda.empty_cache()

    return cached, input_size, metrics


def classify(entry, threshold):
    """Split this film's kept predictions into true and false positives."""
    claimed, n_false = _match(entry["boxes"], entry["scores"], entry["truth"],
                              iou_threshold=IOU, score_threshold=threshold)

    kept = [(box, float(score)) for box, score in zip(entry["boxes"], entry["scores"])
            if float(score) >= threshold]

    # _match reports WHICH truths were claimed and HOW MANY predictions were false, but not
    # which ones. Re-derive per-box status the same way: a kept box is a TP when it clears
    # IoU against some truth box, otherwise it is an FP.
    true_positives, false_positives = [], []

    for box, score in kept:
        matched = False

        for truth_box in entry["truth"]:
            if box_iou_single(box, truth_box) >= IOU:
                matched = True
                break

        (true_positives if matched else false_positives).append((box, score))

    return true_positives, false_positives, claimed, n_false


def box_iou_single(a, b):
    """IoU of two cx, cy, w, h boxes."""
    ax0, ay0 = float(a[0]) - float(a[2]) / 2, float(a[1]) - float(a[3]) / 2
    ax1, ay1 = ax0 + float(a[2]), ay0 + float(a[3])
    bx0, by0 = float(b[0]) - float(b[2]) / 2, float(b[1]) - float(b[3]) / 2
    bx1, by1 = bx0 + float(b[2]), by0 + float(b[3])

    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    union = float(a[2]) * float(a[3]) + float(b[2]) * float(b[3]) - intersection

    return intersection / union if union > 0 else 0.0


def draw(axes, entry, threshold, title):
    axes.imshow(entry["image"], cmap="gray")
    true_positives, false_positives, _, _ = classify(entry, threshold)

    for truth_box in entry["truth"]:
        x, y, w, h = to_corners(truth_box)
        axes.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLOUR_TRUTH, lw=2.2))

    for box, score in true_positives:
        x, y, w, h = to_corners(box)
        axes.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLOUR_TP,
                                 lw=1.8, linestyle="--"))
        axes.text(x, y - 6, "TP {:.2f}".format(score), color=COLOUR_TP, fontsize=7.5)

    for box, score in false_positives:
        x, y, w, h = to_corners(box)
        axes.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLOUR_FP, lw=1.8))
        axes.text(x, y - 6, "FP {:.2f}".format(score), color=COLOUR_FP, fontsize=7.5)

    axes.set_title(title, fontsize=9)
    axes.set_xticks([])
    axes.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="run directory")
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-films", type=int, default=3, help="films per category")
    arguments = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = json.load(open(arguments.run / "metrics.json"))["input_size"]

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=input_size, seed=42, test_seed=0,
        val_frac=0.20, group_by_image=True)

    loader = DataLoader(test_loaders["node21"], batch_size=arguments.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    cached, input_size, metrics = cache_predictions(arguments.run, loader, device)
    threshold = threshold_for_budget(cached, arguments.budget)

    # The whole-split arithmetic, which is what FP/img actually means
    total_fp = total_tp = total_truth = 0
    clean_films = clean_film_fp = 0

    for entry in cached:
        true_positives, false_positives, claimed, _ = classify(entry, threshold)
        total_fp += len(false_positives)
        total_tp += len(claimed)
        total_truth += len(entry["truth"])

        if len(entry["truth"]) == 0:
            clean_films += 1
            clean_film_fp += len(false_positives)

    n_images = len(cached)

    print("\nRun      : {}".format(arguments.run.name))
    print("Threshold: {:.4f}  (fitted so the split lands at {} FP/img)".format(
        threshold, arguments.budget))
    print("\nWHERE FP/img COMES FROM")
    print("  images in test split          {}".format(n_images))
    print("    of which contain a nodule   {}".format(n_images - clean_films))
    print("    of which are nodule-free    {}".format(clean_films))
    print("  ground-truth nodules          {}".format(total_truth))
    print("\n  red boxes (false positives)   {}".format(total_fp))
    print("    on nodule-free films        {}  <- these films can ONLY produce FPs".format(
        clean_film_fp))
    print("    on nodule-bearing films     {}".format(total_fp - clean_film_fp))
    print("  blue boxes (true positives)   {}".format(total_tp))
    print("\n  FP / img = {} / {} = {:.3f}".format(total_fp, n_images, total_fp / n_images))
    print("  sensitivity = {} / {} = {:.3f}".format(total_tp, total_truth,
                                                    total_tp / total_truth))
    print("\n  NOTE there is no 'true negative box' - every kept box is TP or FP. That is")
    print("  why the x-axis is a COUNT PER IMAGE and not a rate in [0, 1], and why")
    print("  detection uses FROC rather than ROC.")

    # Pick films that make the definition unambiguous
    found = [e for e in cached if len(e["truth"]) and classify(e, threshold)[0]]
    missed = [e for e in cached if len(e["truth"]) and not classify(e, threshold)[0]]
    clean_with_fp = [e for e in cached if not len(e["truth"]) and classify(e, threshold)[1]]

    categories = [
        ("nodule FOUND (green + blue)", found),
        ("nodule MISSED (green, no blue)", missed),
        ("NODULE-FREE film: every box is an FP", clean_with_fp),
    ]

    columns = arguments.n_films
    figure, axes_grid = plt.subplots(len(categories), columns,
                                     figsize=(4.1 * columns, 4.3 * len(categories)))

    if len(categories) == 1: axes_grid = np.array([axes_grid])
    if columns == 1: axes_grid = axes_grid.reshape(len(categories), 1)

    for row, (label, pool) in enumerate(categories):
        for column in range(columns):
            axes = axes_grid[row][column]

            if column >= len(pool):
                axes.axis("off")
                continue

            entry = pool[column]
            true_positives, false_positives, _, _ = classify(entry, threshold)
            draw(axes, entry, threshold,
                 "{}\n{} TP, {} FP".format(label if column == 0 else "",
                                           len(true_positives), len(false_positives)))

    figure.suptitle(
        "What a false positive is. green = ground truth, blue = TP (IoU >= {}), "
        "red = FP.\nThreshold {:.3f} gives {} red boxes over {} films = {:.2f} FP/img".format(
            IOU, threshold, total_fp, n_images, total_fp / n_images), fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    path = OUT_DIR / "fig11_false_positives_{}.png".format(arguments.run.name[:60])
    figure.savefig(path, dpi=150)
    plt.close(figure)

    print("\nwrote {}".format(path))


if __name__ == "__main__":
    main()
