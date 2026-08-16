import cv2
import SimpleITK as sitk
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os

class NODE21Dataset(Dataset):
    def __init__(self, df, images_root, augment=False, input_size=2048, npy_cache_dir=None, clahe=True):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.augment = augment
        self.input_size = input_size
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
        return len(self.df)

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
        row = self.df.iloc[idx]
        img_np = self._load_native(row["img_name"], want_clahe=self.clahe)
        native_h, native_w = img_np.shape

        has_nodule = not pd.isna(row["label"]) and row["label"] == 1
        
        if has_nodule:
            x, y, w, h = row["x"], row["y"], row["width"], row["height"]
            cx_native = x + w / 2.0
            cy_native = y + h / 2.0
            size_native = max(w, h)
        else:
            cx_native, cy_native, size_native = 0.0, 0.0, 0.0

        # Resize to input_size
        img_input = cv2.resize(img_np, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        scale_x = self.input_size / native_w
        scale_y = self.input_size / native_h
        cx_input = cx_native * scale_x
        cy_input = cy_native * scale_y
        size_input = size_native * ((scale_x + scale_y) / 2.0)

        if self.augment:
            img_input, cx_input, cy_input = self._augment(img_input, cx_input, cy_input)

        img_tensor = torch.from_numpy(img_input).unsqueeze(0)

        target = {
            "has_nodule": has_nodule,
            "center_2048": (cx_input, cy_input),
            "radius_px": size_input / 2.0,  # "radius" in input space
            "filename": row["img_name"],
        }
        return img_tensor, target

    def _augment(self, img, cx, cy):
        #Horizontal flip (p=0.5)
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            cx = self.input_size - 1 - cx
            
        #Rotation, conservative (magnitude 1-5 deg, random sign)
        angle = float(np.random.uniform(1.0, 5.0))
        if np.random.rand() > 0.5:
            angle = -angle
            
        center = (self.input_size / 2.0, self.input_size / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        new_cx = rotation_matrix[0, 0] * cx + rotation_matrix[0, 1] * cy + rotation_matrix[0, 2]
        new_cy = rotation_matrix[1, 0] * cx + rotation_matrix[1, 1] * cy + rotation_matrix[1, 2]
        
        # Additive Gaussian noise on the normalised image (p=0.3)
        if np.random.rand() < 0.3:
            img = img + np.random.randn(*img.shape).astype(np.float32) * 0.01
            img = np.clip(img, 0.0, 1.0)
            
        return img, new_cx, new_cy
