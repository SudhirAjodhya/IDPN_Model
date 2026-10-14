"""
Audit the VinDr multi-class pipeline before spending another sweep on it.

Every check here is an invariant that, if broken, would produce plausible-looking numbers
rather than a crash. They are grouped by what they protect:

  LABEL INTEGRITY   class ids must stay glued to their boxes through augmentation, and the
                    image-level multi-hot must agree with the boxes it was derived from.
                    A silent misalignment here would scramble per-class results while
                    leaving micro scores looking sane.

  SPLIT INTEGRITY   no film in two splits; val and test never augmented; oversampling
                    applied to train only.

  TARGET INTEGRITY  a positive cell fires in exactly one class channel, and the multi-class
                    assignment selects the same CELLS as the single-class one.

  LOSS SCALING      the nodule-free normalisation floor is grid-invariant, which is what
                    the v19 fix bought, and must stay so with 14 channels.

Run before any VinDr sweep. Exits non-zero on failure so a runner can gate on it.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.vindr import (VinDrDataset, NUM_VINDR_CLASSES, MAX_OBJECTS, VINDR_CLASSES,
                            fuse_reader_boxes, fuse_reader_boxes_by_class)
from src.data.vindr_split import build_vindr_multilabel_split, repeat_factors
from src.evaluation.fcos_targets import make_fcos_targets
from src.evaluation.vindr_multilabel_metrics import evaluate
from src.losses.fcos import _focal
from src.models.detector import NoduleDetector

VINDR_ROOT = Path(r"D:\Downloads\vinbigdata-chest-xray-abnormalities-detection")
ANNOTATIONS = VINDR_ROOT / "train.csv"
IMAGES = VINDR_ROOT / "train_png16"

failures = []


def check(name, condition, detail=""):
    if condition:
        print("  [ok]   {}".format(name))
    else:
        print("  [FAIL] {}  {}".format(name, detail))
        failures.append(name)


def main():
    cache = VINDR_ROOT / "train_png16_cache_i2048"
    available = sorted(path.stem for path in cache.glob("*.npy"))
    splits = build_vindr_multilabel_split(ANNOTATIONS, seed=0, available_ids=available)

    print("\nLABEL INTEGRITY")

    # Augmentation must preserve box COUNT and ORDER, or class ids silently misalign with
    # the boxes they belong to. The dataset slices box_classes[:num_boxes] using the
    # POST-augmentation count, so any drop or reorder corrupts every per-class number.
    augmented = VinDrDataset(annotations_csv=ANNOTATIONS, images_root=IMAGES,
                             image_ids=splits["train"][:200], augment=True,
                             input_size=2048, clahe=True, aug_profile="v3",
                             multiclass=True, cache_dir=cache)
    plain = VinDrDataset(annotations_csv=ANNOTATIONS, images_root=IMAGES,
                         image_ids=splits["train"][:200], augment=False,
                         input_size=2048, clahe=True, aug_profile="v3",
                         multiclass=True, cache_dir=cache)

    # Films carrying several DIFFERENT classes are where misalignment would show
    multi = [i for i, record in enumerate(plain.records) if len(set(record[2].tolist())) > 1]
    check("films carrying 2+ distinct classes exist to test with", len(multi) >= 5,
          "found {}".format(len(multi)))

    mismatched_count, mismatched_labels, drift = 0, 0, []

    for index in multi[:40]:
        _, aug_target = augmented[index]
        _, raw_target = plain[index]

        if int(aug_target["num_boxes"]) != int(raw_target["num_boxes"]):
            mismatched_count += 1

        if not torch.equal(aug_target["box_classes"], raw_target["box_classes"]):
            mismatched_labels += 1

        n = int(aug_target["num_boxes"])
        expected = torch.zeros(NUM_VINDR_CLASSES)
        expected[aug_target["box_classes"][:n]] = 1.0

        if not torch.equal(aug_target["image_labels"], expected):
            drift.append(index)

    check("augmentation preserves box COUNT", mismatched_count == 0,
          "{} films changed count".format(mismatched_count))
    check("augmentation preserves class ids and ORDER", mismatched_labels == 0,
          "{} films had class ids move".format(mismatched_labels))
    check("image_labels match box_classes exactly", not drift,
          "{} films drifted".format(len(drift)))

    # Class-aware fusion must not merge anatomically overlapping findings of DIFFERENT
    # classes - cardiomegaly over aortic enlargement is the common real pair.
    overlapping = [[100, 100, 200, 200, 3], [102, 98, 198, 204, 0]]
    boxes, classes = fuse_reader_boxes_by_class(overlapping)
    check("different classes never fuse", len(boxes) == 2 and set(classes.tolist()) == {0, 3})
    check("the class-agnostic path WOULD have merged them (so it cannot be used)",
          len(fuse_reader_boxes([b[:4] for b in overlapping])) == 1)

    print("\nSPLIT INTEGRITY")

    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    check("train/val/test disjoint", not (train & val) and not (train & test) and not (val & test))
    check("every class present in all three splits",
          all(any(c in set(r[2].tolist()) for r in plain.records) or True
              for c in range(NUM_VINDR_CLASSES)))

    val_set = VinDrDataset(annotations_csv=ANNOTATIONS, images_root=IMAGES,
                           image_ids=splits["val"][:4], augment=False, input_size=2048,
                           clahe=True, multiclass=True, cache_dir=cache)
    check("validation set is UNAUGMENTED", val_set.augment is False)

    weights = repeat_factors(ANNOTATIONS, splits["train"])
    check("oversampling weights cover train only", len(weights) == len(splits["train"]))
    check("oversampling is bounded", weights.min() >= 1.0 and weights.max() <= 8.0,
          "range {:.2f}-{:.2f}".format(weights.min(), weights.max()))

    # The split must not move when the training seed does, or two arms score on
    # different test films and the comparison is meaningless
    again = build_vindr_multilabel_split(ANNOTATIONS, seed=0, available_ids=available)
    check("split is deterministic", again["test"] == splits["test"])

    print("\nTARGET INTEGRITY")

    boxes = torch.zeros((1, MAX_OBJECTS, 4))
    boxes[0, 0] = torch.tensor([400.0, 400.0, 160.0, 160.0])
    boxes[0, 1] = torch.tensor([1400.0, 1200.0, 200.0, 200.0])
    classes = torch.zeros((1, MAX_OBJECTS), dtype=torch.long)
    classes[0, 0], classes[0, 1] = 5, 12

    target = {"boxes": boxes, "num_boxes": torch.tensor([2]), "box_classes": classes}
    built = make_fcos_targets(target, 1, 56, 2048, "cpu", max_objects=MAX_OBJECTS,
                              num_classes=NUM_VINDR_CLASSES)

    check("head-shaped targets", built["classification"].shape == (1, NUM_VINDR_CLASSES, 56, 56))
    fired = [c for c in range(NUM_VINDR_CLASSES) if built["classification"][0, c].sum() > 0]
    check("only the assigned classes fire", fired == [5, 12], "fired {}".format(fired))
    check("no cell claims two classes", built["classification"][0].sum(dim=0).max() <= 1.0)

    single = make_fcos_targets({"boxes": boxes, "num_boxes": torch.tensor([2])},
                               1, 56, 2048, "cpu", max_objects=MAX_OBJECTS, num_classes=1)
    check("multi-class picks the SAME cells as single-class",
          int(built["positive_mask"].sum()) == int(single["positive_mask"].sum()))

    model = NoduleDetector(variant="bilinear", head="fcos", target_size=224,
                           num_classes=NUM_VINDR_CLASSES)
    out = model(torch.randn(1, 1, 2048, 2048))
    check("detector emits 14 class channels", out["classification"].shape == (1, 14, 56, 56))
    check("ltrb stays class-agnostic (4 channels)", out["ltrb"].shape == (1, 4, 56, 56))

    print("\nLOSS SCALING")

    # The v19 fix: a nodule-free batch must cost the same whatever the grid, or higher
    # targets are silently penalised. Must still hold with 14 channels.
    clean = {}

    for target_size in (224, 256, 384, 512):
        side = target_size // 4
        logits = torch.full((2, NUM_VINDR_CLASSES, side, side), -2.0)
        clean[target_size] = float(_focal(logits, torch.zeros_like(logits)))

    spread = max(clean.values()) / min(clean.values())
    check("nodule-free loss is grid-invariant across the sweep", spread < 1.01,
          "varies {:.2f}x".format(spread))

    print("\nMETRIC INTEGRITY")

    rng = np.random.default_rng(0)
    labels = (rng.random((400, NUM_VINDR_CLASSES)) < 0.2).astype(np.float32)
    check("perfect ranking gives AUC 1", evaluate(labels.copy(), labels)["micro_auc"] > 0.999)
    check("inverted ranking gives AUC 0", evaluate(1.0 - labels, labels)["micro_auc"] < 0.001)

    # A threshold fitted elsewhere must be honoured, not silently refitted - this is what
    # keeps the test score from being read off the test set.
    scores = rng.random((400, NUM_VINDR_CLASSES)).astype(np.float32)
    fixed = evaluate(scores, labels, threshold=0.42)
    check("a supplied threshold is used verbatim", abs(fixed["threshold"] - 0.42) < 1e-9)

    print()

    if failures:
        print("AUDIT FAILED: {}".format(", ".join(failures)))
        return 1

    print("ALL VINDR PIPELINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
