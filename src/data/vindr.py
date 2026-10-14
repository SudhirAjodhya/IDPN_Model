"""
VinDr-CXR dataset, used as additional box supervision for the resizer.

The purpose of this dataset in this project is NOT pathology classification. It supplies
localised boxes so the residual receives far more spatial gradient than NODE21 alone can
provide: 18,231 fused boxes across 4,394 abnormal films against NODE21's 1,476, roughly
12.4x more. Every class is therefore collapsed to a single generic foreground class.

Two properties of the raw annotations are handled here rather than downstream:

  1. Multi-reader duplication. Three radiologists read essentially every film, so the
     same lesion appears up to three times as overlapping boxes. Left alone this
     over-weights agreed lesions and inflates the box count. Greedy IoU clustering
     fuses them, taking 36,096 raw boxes to 18,231.

  2. Films marked "No finding" carry no boxes and are kept as negatives, matching the
     way NODE21 retains its negative films.

Images are read from the PNG16 archive written by scripts/preprocess_vindr.py, which
preserves native resolution and raw pixel values losslessly. Native size is roughly
2128-3408 rows, so the decimation ratio to a target of 224 is about 12x against NODE21's
4.6x. That difference is deliberate: it is the harsher Nyquist case the learned resizer
is being asked to handle.
"""

import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import resize_preserving_detail
from src.data.node21 import AUGMENTATION_PROFILES, NODE21Dataset

# VinDr films carry far more boxes than NODE21's three. After fusion the distribution is
# median 3, p99 15, max 39. 16 covers the 99th percentile; the ~1% of films above it are
# truncated, which costs a handful of boxes on already box-dense films.
#
# This differs from NODE21Dataset.MAX_OBJECTS (4) on purpose, and that difference is
# LOAD-BEARING: a batch mixing the two datasets would try to stack a (4, 4) box tensor
# with a (16, 4) one and default_collate fails with "Trying to resize storage that is not
# resizable". Pooled training is therefore REQUIRED to use SingleSourceBatchSampler, which
# train.py enforces for every `--datasets node21_vindr` run, not only the mixed-resolution
# ones. Do not weaken that branch to a mixed-resolution check.
MAX_OBJECTS = 16

# Rows whose class marks the absence of any finding. The film is kept, the row is not.
NO_FINDING_CLASS = "No finding"

# The 14 abnormality classes, in a FIXED order. A class id must mean the same thing in
# every run, every checkpoint and every figure, so this list is sorted and frozen rather
# than derived from whatever order the CSV happens to be read in. Counts are films, from
# train.csv, and they are wildly imbalanced - Pneumothorax has 96 films against Aortic
# enlargement's 3,067, so any per-class number on the tail classes is indicative only.
VINDR_CLASSES = (
    "Aortic enlargement",      # 3067 films
    "Atelectasis",             #  186
    "Calcification",           #  452
    "Cardiomegaly",            # 2300
    "Consolidation",           #  353
    "ILD",                     #  386
    "Infiltration",            #  613
    "Lung Opacity",            # 1322
    "Nodule/Mass",             #  826
    "Other lesion",            # 1134
    "Pleural effusion",        # 1032
    "Pleural thickening",      # 1981
    "Pneumothorax",            #   96
    "Pulmonary fibrosis",      # 1617
)

CLASS_TO_INDEX = {name: index for index, name in enumerate(VINDR_CLASSES)}
NUM_VINDR_CLASSES = len(VINDR_CLASSES)


def load_dicom_pixels(dicom_path) -> tuple[np.ndarray, dict]:
    """
    Decode one VinDr DICOM film to uint16 with a consistent polarity.

    Lives here rather than in scripts/ so both the dataset's fallback path and
    scripts/preprocess_vindr.py use one implementation and cannot drift apart.

    Args:
        dicom_path: path to a single .dicom file

    Returns:
        (pixels, metadata) where pixels is (H, W) uint16 with higher values meaning
        denser tissue. No VOI LUT or windowing is applied, since both are lossy and
        the dataset min-max normalises per film anyway.
    """
    import pydicom

    dataset = pydicom.dcmread(str(dicom_path))
    pixels = dataset.pixel_array

    # Signed data would wrap when cast to uint16. The archive sampled as entirely
    # unsigned, but a stray signed film would otherwise corrupt silently.
    if pixels.min() < 0:
        pixels = pixels - pixels.min()

    photometric = str(dataset.PhotometricInterpretation)

    # MONOCHROME1 stores minimum value as white, the opposite polarity to MONOCHROME2
    # and to NODE21. Reflecting about the film's own range is bijective and does not
    # depend on BitsStored being accurate, which it is not always.
    inverted = photometric == "MONOCHROME1"

    if inverted:
        # Computed in int32 to avoid a uint16 overflow WARNING when max + min exceeds
        # 65535. The warning is benign - uint16 arithmetic is modular and the true result
        # max+min-x always lies within [min, max], so the wrap in the addition is exactly
        # cancelled by the wrap in the subtraction (verified identical over 20,000 random
        # cases). Kept only so the intent is explicit and the log stays clean; output is
        # unchanged, so the existing cache and PNG archive remain valid.
        reflection_point = int(pixels.max()) + int(pixels.min())
        pixels = (reflection_point - pixels.astype(np.int32)).astype(np.uint16)

    metadata = {
        "rows": int(dataset.Rows),
        "cols": int(dataset.Columns),
        "bits_stored": int(getattr(dataset, "BitsStored", 16)),
        "photometric": photometric,
        "inverted": int(inverted),
    }

    return pixels.astype(np.uint16), metadata


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    """
    Intersection over union of two boxes in (x_min, y_min, x_max, y_max) form.

    Args:
        first: array-like of 4 floats
        second: array-like of 4 floats

    Returns:
        IoU in [0, 1], or 0.0 for degenerate boxes.
    """
    intersect_x1, intersect_y1 = max(first[0], second[0]), max(first[1], second[1])
    intersect_x2, intersect_y2 = min(first[2], second[2]), min(first[3], second[3])

    intersect_width = max(0.0, intersect_x2 - intersect_x1)
    intersect_height = max(0.0, intersect_y2 - intersect_y1)
    intersection = intersect_width * intersect_height

    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection

    return intersection / union if union > 0 else 0.0


def fuse_reader_boxes(boxes: list, iou_threshold: float = 0.4) -> np.ndarray:
    """
    Collapse overlapping boxes from different radiologists into one box per lesion.

    Greedy largest-first clustering: each box either joins the first existing cluster it
    overlaps beyond the threshold, or seeds a new one. Each cluster is then averaged.

    CLASS-AGNOSTIC, and callers that care about class must not use it directly. Two
    different findings on one film routinely overlap - a cardiomegaly box and an aortic
    enlargement box sit on top of each other by anatomy - so fusing across classes would
    merge distinct lesions into one and silently lose a label. `fuse_reader_boxes_by_class`
    below is the class-aware entry point; this one is kept for the single-class pooled
    training path, whose behaviour must not move.

    Args:
        boxes: list of (x_min, y_min, x_max, y_max) in native pixel coordinates
        iou_threshold: overlap above which two boxes are treated as the same lesion

    Returns:
        (N, 4) float32 array of fused boxes, still in (x_min, y_min, x_max, y_max).
    """
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)

    # Largest first, so a big box anchors its cluster rather than being absorbed into a
    # small one that happened to be seen first.
    ordered = sorted(boxes, key=lambda box: -((box[2] - box[0]) * (box[3] - box[1])))
    clusters = []

    for box in ordered:
        for anchor, members in clusters:
            if box_iou(box, anchor) > iou_threshold:
                members.append(box)
                break

        else:
            clusters.append((box, [box]))

    return np.array([np.mean(members, axis=0) for _, members in clusters], dtype=np.float32)


def fuse_reader_boxes_by_class(boxes_with_class: list, iou_threshold: float = 0.4):
    """
    Fuse multi-reader boxes WITHIN each class, never across classes.

    Three radiologists marking one lesion agree on its class, so the duplication this
    removes is always within-class. Fusing across classes would collapse anatomically
    overlapping findings - cardiomegaly and aortic enlargement being the obvious pair -
    into a single box and lose one of the two labels.

    Args:
        boxes_with_class: list of (x_min, y_min, x_max, y_max, class_index)
        iou_threshold: overlap above which two boxes are the same lesion

    Returns:
        (boxes, classes) - (N, 4) float32 and (N,) int64, aligned row for row.
    """
    by_class = defaultdict(list)

    for x_min, y_min, x_max, y_max, class_index in boxes_with_class:
        by_class[int(class_index)].append([x_min, y_min, x_max, y_max])

    fused_boxes, fused_classes = [], []

    for class_index in sorted(by_class):
        for box in fuse_reader_boxes(by_class[class_index], iou_threshold):
            fused_boxes.append(box)
            fused_classes.append(class_index)

    if not fused_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(fused_boxes, dtype=np.float32), np.array(fused_classes, dtype=np.int64)


class VinDrDataset(Dataset):
    """
    VinDr-CXR films with every pathology box treated as one generic foreground class.

    Emits the same target dictionary shape as NODE21Dataset so both can feed the same
    heads, losses and target-assignment code unchanged.
    """

    # Augmentation is deliberately identical to NODE21's rather than a second
    # implementation that could drift from it. The method only reads self.input_size and
    # self.aug_profile, both of which this class provides.
    _augment = NODE21Dataset._augment

    def __init__(self, annotations_csv, images_root, image_ids=None, augment=False,
                 input_size=2048, clahe=True, aug_profile="v3", letterbox=False,
                 iou_threshold=0.4, include_negatives=True, negative_ratio=None,
                 subsample_seed=0, cache_dir=None, dicom_root=None, multiclass=False):
        """
        Args:
            annotations_csv: path to VinDr train.csv
            images_root: directory of PNG16 films from scripts/preprocess_vindr.py
            image_ids: optional iterable restricting the dataset to these ids (splits)
            augment: apply the shared geometric and photometric augmentation
            input_size: square size every film is resized to before the model sees it
            clahe: contrast normalisation, applied to train and eval alike
            aug_profile: key into the shared AUGMENTATION_PROFILES table
            letterbox: preserve aspect and pad, instead of squashing to square
            iou_threshold: overlap at which two readers' boxes are treated as one lesion
            multiclass: keep the 14 VinDr class labels instead of collapsing every box to
                one generic foreground class. Defaults False so every pooled
                NODE21+VinDr run behaves exactly as it did before this flag existed.
            include_negatives: keep "No finding" films as background-only samples
            negative_ratio: keep only this many negatives per positive film. None keeps
                every negative. 10,606 of VinDr's 15,000 films carry NO boxes, so they
                contribute nothing to the box supervision this dataset exists to provide
                while costing a full share of epoch time. 1.0 keeps all 18,231 boxes at
                41% fewer films.
            subsample_seed: fixes which negatives are kept, so the subset is reproducible
            cache_dir: where to store decoded films. None auto-names a sibling of
                images_root keyed on input_size and letterbox.
        """
        self.images_root = Path(images_root)
        self.dicom_root = Path(dicom_root) if dicom_root else None
        self.augment = augment
        self.input_size = input_size
        self.clahe = clahe
        self.aug_profile_name = aug_profile
        self.aug_profile = AUGMENTATION_PROFILES[aug_profile]
        self.letterbox = letterbox

        # Post-CLAHE, post-resize cache. Measured cost per film without it is ~370 ms,
        # of which PNG decode is 47%, CLAHE at native resolution 33% and normalisation
        # 14% - all repeated every epoch on every film. NODE21 has had an equivalent
        # cache all along and loads in 2.7 ms, so an uncached VinDr made the pooled
        # dataloader ~137x slower per film and left the GPU idle ~80% of the time.
        #
        # The key MUST include input_size and letterbox: the cached array is already
        # resized, so a 1024 run and a 2048 run need separate caches.
        if cache_dir is None:
            suffix = f"_i{input_size}" + ("_lb" if letterbox else "")
            self.cache_dir = self.images_root.with_name(self.images_root.name + "_cache" + suffix)
        else:
            self.cache_dir = Path(cache_dir)

        # Native dimensions come from the conversion manifest so a cache hit never has to
        # decode the PNG just to learn how to scale the boxes.
        self.native_dims = {}
        manifest_path = self.images_root.parent / f"{self.images_root.name}_manifest.csv"

        if manifest_path.exists():
            with open(manifest_path, newline="") as handle:
                for row in csv.DictReader(handle):
                    self.native_dims[row["image_id"]] = (int(row["rows"]), int(row["cols"]))

        self.multiclass = multiclass

        # Gather every annotated box per film, dropping the "No finding" marker rows but
        # remembering that the film existed so it can be kept as a negative.
        boxes_by_image = defaultdict(list)
        seen_image_ids = set()

        with open(annotations_csv, newline="") as handle:
            for row in csv.DictReader(handle):
                image_id = row["image_id"]
                seen_image_ids.add(image_id)

                if row["class_name"] == NO_FINDING_CLASS or row["x_min"] == "":
                    continue

                # The class index rides along as a 5th element. The single-class path
                # slices it off below, so its records are unchanged.
                boxes_by_image[image_id].append([float(row["x_min"]), float(row["y_min"]),
                                                 float(row["x_max"]), float(row["y_max"]),
                                                 CLASS_TO_INDEX[row["class_name"]]])

        wanted = set(image_ids) if image_ids is not None else seen_image_ids

        # Fuse once at construction. Doing it per __getitem__ would repeat the clustering
        # on every epoch for no benefit.
        self.records = []

        for image_id in sorted(wanted):
            if image_id not in seen_image_ids:
                continue

            raw = boxes_by_image.get(image_id, [])

            if self.multiclass:
                fused, classes = fuse_reader_boxes_by_class(raw, iou_threshold)
            else:
                # Drop the class column and fuse exactly as before - this branch feeds the
                # pooled NODE21+VinDr runs and must stay byte-identical to them.
                fused = fuse_reader_boxes([box[:4] for box in raw], iou_threshold)
                classes = np.zeros((len(fused),), dtype=np.int64)

            if len(fused) == 0 and not include_negatives:
                continue

            self.records.append((image_id, fused, classes))

        # Drop surplus negatives. Every box lives on a positive film, so this keeps 100%
        # of the supervision. Sorted-then-seeded so the kept subset is identical across
        # runs, arms and machines.
        if negative_ratio is not None:
            positives = [record for record in self.records if len(record[1])]
            negatives = [record for record in self.records if not len(record[1])]

            keep_count = min(len(negatives), int(round(len(positives) * negative_ratio)))
            generator = np.random.default_rng(subsample_seed)
            kept_indices = sorted(generator.choice(len(negatives), size=keep_count, replace=False))

            self.records = positives + [negatives[i] for i in kept_indices]
            self.records.sort(key=lambda record: record[0])

        # Refuse to cache when it would not fit. The cache is populated lazily by the
        # dataloader workers, so without this check a large input_size silently fills the
        # disk mid-run and takes the training down with it. At 1024 the cache is ~18 GB;
        # at 2048 it would be ~74 GB, which does not fit alongside the PNG archive.
        # Skipping it is cheap there anyway: a high-resolution run is compute-bound in the
        # resizer's conv stack, not dataloader-bound.
        projected_bytes = len(self.records) * input_size * input_size * 2
        free_bytes = shutil.disk_usage(self.images_root.drive or self.images_root).free

        if projected_bytes * 1.15 > free_bytes:
            print(f"VinDr cache DISABLED: needs {projected_bytes / 1e9:.0f} GB at "
                  f"input {input_size}, only {free_bytes / 1e9:.0f} GB free. "
                  f"Loading uncached at roughly 300 ms/film.")
            self.cache_dir = None

        positive_films = sum(1 for _, boxes, _ in self.records if len(boxes))
        total_boxes = sum(len(boxes) for _, boxes, _ in self.records)

        if self.cache_dir is None:
            cache_note = "[cache disabled]"
        else:
            cached = len(list(self.cache_dir.glob("*.npy"))) if self.cache_dir.exists() else 0
            cache_note = f"[cache {self.cache_dir.name}: {cached}/{len(self.records)} films]"

        print(f"VinDr loaded: {len(self.records)} films "
              f"({positive_films} with boxes, {total_boxes} fused boxes) from {self.images_root} "
              f"{cache_note}")

    def __len__(self):
        return len(self.records)

    def _load_native(self, image_id: str) -> np.ndarray:
        """
        Read one PNG16 film as float32 in [0, 1], optionally CLAHE-normalised.

        Min-max normalisation per film is what makes the varying stored bit depth
        (10, 12, 14 and 16-bit all appear in this archive) irrelevant downstream.

        Args:
            image_id: film identifier, without extension

        Returns:
            (H, W) float32 array in [0, 1] at native resolution.
        """
        image_path = self.images_root / f"{image_id}.png"
        # Existence-checked rather than letting cv2 fail: imread logs a warning per miss,
        # which would emit one line per film when the PNG archive is absent.
        raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED) if image_path.exists() else None

        # Fall back to the original DICOM when the PNG archive is absent. The per-input-size
        # cache supersedes that archive entirely - it is smaller and skips CLAHE as well as
        # decode - so the intermediate PNGs need not exist at all once a cache is warm.
        if raw is None and self.dicom_root is not None:
            dicom_path = self.dicom_root / f"{image_id}.dicom"

            if dicom_path.exists():
                raw, _ = load_dicom_pixels(dicom_path)

        if raw is None:
            raise FileNotFoundError(
                f"VinDr film not readable: {image_path}"
                + (f" (nor {self.dicom_root / (image_id + '.dicom')})" if self.dicom_root else "")
                + ". Warm the cache with scripts/warm_vindr_cache.py.")

        image = raw.astype(np.float32)

        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = image - image.min()

        if self.clahe:
            from src.data.augmentation import apply_clahe
            image = apply_clahe(image)

        return image

    def _load_input(self, image_id: str):
        """
        Return the model-ready (input_size, input_size) float32 [0,1] film.

        Reads the post-CLAHE, post-resize cache when present, and populates it otherwise.
        Cached as uint16, which is lossless for CLAHE output and halves the disk against
        float32, matching what node21.py does.

        Args:
            image_id: film identifier, without extension

        Returns:
            (image, (native_h, native_w)). Native dimensions come from the conversion
            manifest on a cache hit, so no PNG is decoded just to scale the boxes.
        """
        cache_path = self.cache_dir / f"{image_id}.npy" if self.cache_dir else None
        native = self.native_dims.get(image_id)

        if cache_path is not None and native is not None and cache_path.exists():
            try:
                return np.load(cache_path).astype(np.float32) / 65535.0, native

            except Exception:
                # A torn write from an interrupted run: fall through and rebuild it.
                pass

        image = self._load_native(image_id)
        native_h, native_w = image.shape

        # resize_preserving_detail anti-aliases on the way down. VinDr films are
        # 2128-3408 px natively, so reaching input_size decimates and a plain bilinear
        # kernel would alias in the dataloader as well as in the resizer under test.
        # Shared with NODE21 so the two datasets cannot drift apart on this.
        if self.letterbox:
            # One uniform scale for both axes, then centre-pad to square, so lesions keep
            # their true shape. VinDr is almost entirely non-square, so this matters more
            # here than on NODE21 where 39% of films are affected.
            scale = self.input_size / max(native_h, native_w)
            new_w, new_h = int(round(native_w * scale)), int(round(native_h * scale))
            resized = resize_preserving_detail(image, (new_w, new_h))

            image_input = np.zeros((self.input_size, self.input_size), dtype=np.float32)
            pad_x = (self.input_size - new_w) // 2
            pad_y = (self.input_size - new_h) // 2
            image_input[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        else:
            # Matches NODE21's default: independent per-axis scaling, which squashes
            # non-square films.
            image_input = resize_preserving_detail(image, (self.input_size, self.input_size))

        # Written atomically, so concurrent dataloader workers cannot read a partial file.
        if cache_path is not None and native is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_dir / f"{image_id}.tmp.{os.getpid()}.npy"
            np.save(temporary, (np.clip(image_input, 0.0, 1.0) * 65535).astype(np.uint16))
            temporary.replace(cache_path)

        return image_input, (native_h, native_w)

    def __getitem__(self, idx):
        image_id, boxes_native, box_classes = self.records[idx]

        image_input, (native_h, native_w) = self._load_input(image_id)

        # Box scaling mirrors whatever _load_input did to the pixels. Derived from the
        # native dimensions rather than from the image, so it is identical on a cache hit.
        if self.letterbox:
            scale = self.input_size / max(native_h, native_w)
            new_w, new_h = int(round(native_w * scale)), int(round(native_h * scale))
            pad_x = (self.input_size - new_w) // 2
            pad_y = (self.input_size - new_h) // 2
            scale_x = scale_y = scale

        else:
            scale_x = self.input_size / native_w
            scale_y = self.input_size / native_h
            pad_x = pad_y = 0

        # Convert (x_min, y_min, x_max, y_max) native boxes to (cx, cy, w, h) in
        # input_size pixel space, which is the convention every head in this repo reads.
        boxes_input = np.zeros((len(boxes_native), 4), dtype=np.float32)

        for i, (x_min, y_min, x_max, y_max) in enumerate(boxes_native):
            width, height = x_max - x_min, y_max - y_min
            boxes_input[i] = ((x_min + width / 2.0) * scale_x + pad_x,
                              (y_min + height / 2.0) * scale_y + pad_y,
                              width * scale_x, height * scale_y)

        if self.augment:
            image_input, boxes_input = self._augment(image_input, boxes_input)

        has_nodule = len(boxes_input) > 0
        image_tensor = torch.from_numpy(image_input).unsqueeze(0)

        # Pad to MAX_OBJECTS so default_collate can stack a fixed-shape tensor.
        # num_boxes carries the real count; padded rows are zero and must be masked.
        padded = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
        num_boxes = min(len(boxes_input), MAX_OBJECTS)
        padded[:num_boxes] = boxes_input[:num_boxes]

        # Class ids, padded the same way. Padded slots read 0, which is a REAL class id,
        # so num_boxes is the only thing that says which rows are meaningful - exactly as
        # it already is for the box coordinates themselves.
        padded_classes = np.zeros((MAX_OBJECTS,), dtype=np.int64)
        padded_classes[:num_boxes] = box_classes[:num_boxes]

        # Image-level multi-label vector, which is what the VinDr sweep actually scores
        # against. Derived here rather than downstream so augmentation that drops a box
        # cannot desynchronise it from the boxes.
        image_labels = np.zeros((NUM_VINDR_CLASSES,), dtype=np.float32)
        image_labels[box_classes[:num_boxes]] = 1.0

        first_cx, first_cy, first_w, first_h = (padded[0] if num_boxes else (0.0, 0.0, 0.0, 0.0))

        target = {
            "has_nodule": has_nodule,
            "center_2048": (float(first_cx), float(first_cy)),
            "radius_px": float(max(first_w, first_h) / 2.0),
            "boxes": torch.from_numpy(padded),          # (MAX_OBJECTS, 4) as cx, cy, w, h
            "box_classes": torch.from_numpy(padded_classes),   # (MAX_OBJECTS,) int64
            "image_labels": torch.from_numpy(image_labels),    # (14,) multi-hot
            "num_boxes": num_boxes,
            "native_width": int(native_w),
            "native_height": int(native_h),
            "filename": f"{image_id}.png",
        }

        return image_tensor, target


def _demo() -> None:
    """Self-check for the two pieces of logic here that can silently go wrong."""

    # Fusion collapses three readers marking one lesion, and keeps two distinct lesions
    # apart. This is the behaviour the 36,096 -> 18,231 reduction depends on.
    one_lesion_three_readers = [[100, 100, 200, 200], [102, 98, 198, 204], [99, 101, 201, 199]]
    fused = fuse_reader_boxes(one_lesion_three_readers)
    assert len(fused) == 1, f"three readers on one lesion should fuse to 1, got {len(fused)}"

    two_lesions = one_lesion_three_readers + [[800, 800, 900, 900]]
    assert len(fuse_reader_boxes(two_lesions)) == 2, "distinct lesions must not fuse"

    # Non-overlapping boxes must never fuse regardless of ordering.
    assert len(fuse_reader_boxes([[0, 0, 10, 10], [50, 50, 60, 60]])) == 2

    # CLASS-AWARE FUSION. The failure this guards against is anatomical: cardiomegaly and
    # aortic enlargement overlap heavily on a real film, and the class-agnostic path
    # merges them into one box, losing a label. Same geometry, two classes:
    overlapping_two_classes = [
        [100, 100, 200, 200, CLASS_TO_INDEX["Cardiomegaly"]],
        [102, 98, 198, 204, CLASS_TO_INDEX["Aortic enlargement"]],
    ]
    boxes, classes = fuse_reader_boxes_by_class(overlapping_two_classes)
    assert len(boxes) == 2, f"different classes must never fuse, got {len(boxes)} boxes"
    assert set(classes.tolist()) == {CLASS_TO_INDEX["Cardiomegaly"],
                                     CLASS_TO_INDEX["Aortic enlargement"]}
    assert len(fuse_reader_boxes([box[:4] for box in overlapping_two_classes])) == 1,         "the class-agnostic path is supposed to merge these - that is why it cannot be used"
    print("OK  two classes on overlapping boxes stay separate (class-agnostic gives 1)")

    # Within one class, three readers still collapse to one lesion
    three_readers_one_class = [box + [CLASS_TO_INDEX["Nodule/Mass"]]
                               for box in one_lesion_three_readers]
    boxes, classes = fuse_reader_boxes_by_class(three_readers_one_class)
    assert len(boxes) == 1 and classes.tolist() == [CLASS_TO_INDEX["Nodule/Mass"]]
    print("OK  three readers on one lesion still fuse to 1, class preserved")

    # The class vocabulary is frozen: an id must mean the same thing in every run
    assert len(VINDR_CLASSES) == 14 and len(CLASS_TO_INDEX) == 14
    assert VINDR_CLASSES[0] == "Aortic enlargement" and VINDR_CLASSES[-1] == "Pulmonary fibrosis"
    assert list(VINDR_CLASSES) == sorted(VINDR_CLASSES),         "VINDR_CLASSES must stay sorted or saved class ids stop matching new runs"
    print("OK  14 classes, frozen order")

    # IoU sanity: identical boxes are 1.0, disjoint are 0.0.
    assert abs(box_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    # A padded target must never claim more boxes than MAX_OBJECTS.
    assert min(40, MAX_OBJECTS) == MAX_OBJECTS

    print("vindr self-check passed")


if __name__ == "__main__":
    _demo()
