import cv2
import SimpleITK as sitk
import pandas as pd
import numpy as np

from .augmentation import resize_preserving_detail

# NODE21 has at most 3 nodules on any single film; 4 leaves headroom for padding.
MAX_OBJECTS = 4
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os

# Augmentation profiles. Selected per run so a sweep never silently carries an
# unvalidated change, and so results stay comparable to the baseline they are
# being measured against.
AUGMENTATION_PROFILES = {
    # Settings every result up to 2026-08-17 was produced under. Default.
    "v3": {"rotation_deg": (1.0, 5.0), "max_shift_frac": 0.0,  "noise_sigma": 0.01, "noise_prob": 0.3},

    # Stronger geometric and photometric jitter, 2026-08-18. Reduced overfitting
    # (val loss final 0.860 -> 0.781) but dropped NODE21 FROC AUC 0.803 -> 0.767 on
    # s42, confounded with positive oversampling in that same run. Opt-in until the
    # aug-only disambiguation run separates the two.
    "v4": {"rotation_deg": (1.0, 7.0), "max_shift_frac": 0.05, "noise_sigma": 0.03, "noise_prob": 0.5},
}


class NODE21Dataset(Dataset):
    def __init__(self, df, images_root, augment=False, input_size=2048, npy_cache_dir=None, clahe=True,
                 aug_profile="v3", group_by_image=False, letterbox=False):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.augment = augment
        self.input_size = input_size
        self.aug_profile_name = aug_profile
        self.aug_profile = AUGMENTATION_PROFILES[aug_profile]

        # Per-image mode for box detection. The annotation table has one row per NODULE,
        # so a 2-nodule film otherwise yields two samples in which the OTHER nodule sits
        # in the image but is labelled background. Grouping collapses each film to one
        # sample carrying every box it has.
        # letterbox=False squashes every film to a square, which is what all runs before
        # 2026-08-21 did. 39% of NODE21 films are non-square natively (up to 3920x3612),
        # so squashing distorts nodules into ellipses by a per-image factor. The NODE21
        # reference pipelines resize with aspect preserved and pad instead. Kept off by
        # default so earlier runs stay reproducible.
        self.letterbox = letterbox

        self.group_by_image = group_by_image
        self.records = None

        if group_by_image:
            self.records = []

            for image_name, rows in self.df.groupby("img_name", sort=False):
                positives = rows[rows["label"] == 1]

                if len(positives):
                    boxes_native = positives[["x", "y", "width", "height"]].to_numpy(dtype=np.float32)
                else:
                    boxes_native = np.zeros((0, 4), dtype=np.float32)

                self.records.append((image_name, boxes_native))
        self.clahe = clahe   # applied ALWAYS (train + val + test) - it's a normalization, not data aug

        # Auto-detect sibling '<images_root>_npy' cache written by scripts/cache_node21_npy.py.
        # Using it skips SimpleITK decode + normalize per __getitem__ (5-10x speedup, tiny RAM).
        if npy_cache_dir is None:
            candidate = self.images_root.with_name(self.images_root.name + "_npy")
            self.npy_cache_dir = candidate if candidate.exists() else None
        else:
            self.npy_cache_dir = Path(npy_cache_dir)

        positives = len(self.df[self.df['label'] == 1])
        unique_pos_images = self.df[self.df['label'] == 1]['img_name'].nunique()
        cache_note = f" [npy cache: {self.npy_cache_dir}]" if self.npy_cache_dir else " [no npy cache]"
        print(f"NODE21 loaded: {len(self.df)} examples ({positives} annotations across {unique_pos_images} unique images){cache_note}")

    def __len__(self):
        return len(self.records) if self.group_by_image else len(self.df)

    def _clahe_cache_dir(self):
        # Sibling '<images_root>_clahe_npy' cache: stores post-CLAHE float32 [0,1].
        # Populated lazily on first read per file, atomically to survive concurrent workers.
        return self.npy_cache_dir.with_name(
            self.npy_cache_dir.name.replace("_npy", "_clahe_npy")
        ) if self.npy_cache_dir is not None else None

    def _load_native(self, img_name, want_clahe=False):
        """Return native float32 [0,1] array. Two-tier cache: post-CLAHE, then raw."""
        stem = Path(img_name).stem
        # Tier 1: post-CLAHE cache - skips both raw load AND CLAHE compute.
        # Stored as uint16 (lossless for CLAHE output, halves disk vs float32).
        if want_clahe and (cdir := self._clahe_cache_dir()) is not None:
            cpath = cdir / f"{stem}.npy"
            if cpath.exists():
                return np.load(cpath).astype(np.float32) / 65535.0
        # Tier 2: raw npy cache - skips SimpleITK decode + normalize.
        if self.npy_cache_dir is not None:
            rpath = self.npy_cache_dir / f"{stem}.npy"
            if rpath.exists():
                img_np = np.load(rpath)
            else:
                img_np = None
        else:
            img_np = None
        # Fallback: original SimpleITK path.
        if img_np is None:
            img_sitk = sitk.ReadImage(str(self.images_root / img_name))
            img_np = sitk.GetArrayFromImage(img_sitk).squeeze().astype(np.float32)
            if img_np.max() > img_np.min():
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = img_np - img_np.min()
        # Apply CLAHE and populate cache atomically (concurrent workers safe).
        if want_clahe:
            from src.data.augmentation import apply_clahe
            img_np = apply_clahe(img_np)
            if (cdir := self._clahe_cache_dir()) is not None:
                cdir.mkdir(parents=True, exist_ok=True)
                cpath = cdir / f"{stem}.npy"
                tmp = cpath.parent / f"{cpath.name}.tmp.{os.getpid()}.npy"        # np.save appends .npy, so include it here so .replace() targets the right file
                np.save(tmp, (np.clip(img_np, 0.0, 1.0) * 65535).astype(np.uint16))
                tmp.replace(cpath)                                                # atomic rename on POSIX + Windows
        return img_np

    def __getitem__(self, idx):
        if self.group_by_image:
            image_name, boxes_native = self.records[idx]
        else:
            row = self.df.iloc[idx]
            image_name = row["img_name"]

            if not pd.isna(row["label"]) and row["label"] == 1:
                boxes_native = np.array([[row["x"], row["y"], row["width"], row["height"]]], dtype=np.float32)
            else:
                boxes_native = np.zeros((0, 4), dtype=np.float32)

        img_np = self._load_native(image_name, want_clahe=self.clahe)
        native_h, native_w = img_np.shape

        # Resize to input_size, carrying box coordinates through the same transform.
        #
        # CHANGED 2026-08-27: shrinking now uses INTER_AREA via resize_preserving_detail
        # instead of INTER_LINEAR. 63% of NODE21 films are larger than a 1024 input and
        # were previously decimated with no low-pass filter, which aliases. Every run
        # before this date used the aliasing path, so their numbers are NOT directly
        # comparable to runs after it. See __CURRENT_STATE.md for the re-run decision.
        if self.letterbox:
            # One uniform scale for both axes, then centre-pad to square. Nodules keep
            # their shape; the cost is padding bands on non-square films.
            scale = self.input_size / max(native_h, native_w)
            new_w, new_h = int(round(native_w * scale)), int(round(native_h * scale))
            resized = resize_preserving_detail(img_np, (new_w, new_h))

            pad_x = (self.input_size - new_w) // 2
            pad_y = (self.input_size - new_h) // 2

            img_input = np.zeros((self.input_size, self.input_size), dtype=np.float32)
            img_input[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            scale_x = scale_y = scale

        else:
            # Legacy: independent per-axis scaling, which squashes non-square films
            img_input = resize_preserving_detail(img_np, (self.input_size, self.input_size))
            scale_x = self.input_size / native_w
            scale_y = self.input_size / native_h
            pad_x = pad_y = 0

        # boxes_input: (N, 4) as (cx, cy, w, h) in input_size pixel space
        boxes_input = np.zeros((len(boxes_native), 4), dtype=np.float32)

        for i, (x, y, w, h) in enumerate(boxes_native):
            boxes_input[i] = ((x + w / 2.0) * scale_x + pad_x, (y + h / 2.0) * scale_y + pad_y,
                              w * scale_x, h * scale_y)

        if self.augment:
            img_input, boxes_input = self._augment(img_input, boxes_input)

        has_nodule = len(boxes_input) > 0
        img_tensor = torch.from_numpy(img_input).unsqueeze(0)

        # Pad to MAX_OBJECTS so default_collate can stack a fixed-shape tensor.
        # num_boxes carries the real count; padded rows are zero and must be masked.
        padded = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
        num_boxes = min(len(boxes_input), MAX_OBJECTS)
        padded[:num_boxes] = boxes_input[:num_boxes]

        # center_2048 / radius_px keep the pre-existing heatmap-only path working. They
        # describe the FIRST box only; the box-detection path must read "boxes" instead.
        first_cx, first_cy, first_w, first_h = (padded[0] if num_boxes else (0.0, 0.0, 0.0, 0.0))

        target = {
            "has_nodule": has_nodule,
            "center_2048": (float(first_cx), float(first_cy)),
            "radius_px": float(max(first_w, first_h) / 2.0),
            "boxes": torch.from_numpy(padded),          # (MAX_OBJECTS, 4) as cx, cy, w, h
            "num_boxes": num_boxes,
            "native_width": int(native_w),
            "native_height": int(native_h),
            "filename": image_name,
        }
        return img_tensor, target

    def _augment(self, img, boxes):
        """
        Geometric and photometric augmentation for a single training film.

        Rotation and translation are folded into one affine matrix so the image is
        resampled only once, and every box centre is carried through that same matrix.

        Box width and height are preserved rather than recomputed from rotated corners.
        A pulmonary nodule is approximately circular, so its axis-aligned box is
        rotation-invariant in size; taking the AABB of the rotated corners would instead
        inflate every box by (cos+sin) - about 8% at 5 degrees - biasing the wh
        regression target upward on every augmented sample.

        Args:
            img: (input_size, input_size) float32 [0, 1] image
            boxes: (N, 4) array of (cx, cy, w, h) in input_size pixel space, N may be 0

        Returns:
            (augmented image, augmented boxes)
        """
        boxes = boxes.copy()

        # Horizontal flip (p=0.5)
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()

            if len(boxes): boxes[:, 0] = self.input_size - 1 - boxes[:, 0]

        # Rotation, magnitude from the active profile with random sign. Kept inside the
        # range of real patient stance and positioning variation on a PA film.
        rotation_low, rotation_high = self.aug_profile["rotation_deg"]
        angle = float(np.random.uniform(rotation_low, rotation_high))

        if np.random.rand() > 0.5:
            angle = -angle

        center = (self.input_size / 2.0, self.input_size / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Where each centre lands after rotation alone, before translation is added
        if len(boxes):
            rotated_cx = rotation_matrix[0, 0] * boxes[:, 0] + rotation_matrix[0, 1] * boxes[:, 1] + rotation_matrix[0, 2]
            rotated_cy = rotation_matrix[1, 0] * boxes[:, 0] + rotation_matrix[1, 1] * boxes[:, 1] + rotation_matrix[1, 2]

        # Translation, up to +/-max_shift_frac of the frame per axis. When the profile
        # disables it the RNG is not drawn at all, so the "v3" profile reproduces the
        # original random stream exactly under a fixed seed.
        max_shift = self.aug_profile["max_shift_frac"] * self.input_size

        if max_shift > 0:
            shift_x = float(np.random.uniform(-max_shift, max_shift))
            shift_y = float(np.random.uniform(-max_shift, max_shift))

        else:
            shift_x, shift_y = 0.0, 0.0

        # Clamp the shift so no box is pushed out of frame. Without this a border nodule
        # can land outside the heatmap grid, producing an all-zero GT on a sample still
        # flagged has_nodule=True - silent label corruption. The clamp is taken over ALL
        # boxes, so a film keeps every nodule it started with.
        if len(boxes) and max_shift > 0:
            margin = 0.02 * self.input_size
            shift_x = float(np.clip(shift_x, (margin - rotated_cx).max(), (self.input_size - margin - rotated_cx).min()))
            shift_y = float(np.clip(shift_y, (margin - rotated_cy).max(), (self.input_size - margin - rotated_cy).min()))

        rotation_matrix[0, 2] += shift_x
        rotation_matrix[1, 2] += shift_y

        img = cv2.warpAffine(img, rotation_matrix, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        if len(boxes):
            boxes[:, 0] = rotated_cx + shift_x
            boxes[:, 1] = rotated_cy + shift_y

        # Additive Gaussian noise. Raised from sigma=0.01 in the v4 profile, which survived
        # neither CLAHE nor the bilinear downsample to target resolution.
        if np.random.rand() < self.aug_profile["noise_prob"]:
            img = img + np.random.randn(*img.shape).astype(np.float32) * self.aug_profile["noise_sigma"]
            img = np.clip(img, 0.0, 1.0)

        return img, boxes
