import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

def _apply_rotation(image, cx, cy, angle_deg):
    """
    Data Augmentation: Rotation

    Method:
    Augments dataset by applying roation to image. Image is rotated about the center (width / 2, height / 2). We rotate all pixels and also need to update the cx and cy potisions, as nodule location has moved

    Args:
        image: chest X-ray image
        cx, cy: nodule center coordinates
        angle_deg: angle to be rotated
    """
    height, width = image.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0) # 2 x 3 affine transformation matrix (rotation), used to warp pixels of cxr
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, rotation_matrix[0, 0] * cx + rotation_matrix[0, 1] * cy + rotation_matrix[0, 2], rotation_matrix[1, 0] * cx + rotation_matrix[1, 1] * cy + rotation_matrix[1, 2]

class JSRTDataset(Dataset):
    """
    8-bit JSRT CXRs at 2048x2048 with nodule annotations
    """

    def __init__(self, df, image_directory, augment=False, input_size=2048, pixel_spacing_mm=0.175, clahe=True):
        self.df = df.reset_index(drop=True)
        self.image_directory = Path(image_directory)
        self.augment = augment # Flag for if we are doing data augmentation  - false by default
        self.input_size = input_size
        self.pixel_spacing_mm = pixel_spacing_mm
        self.clahe = clahe   # applied ALWAYS (train + val + test) — it's a normalization, not aug

    def __len__(self):
        return len(self.df)

    def _clahe_cache_dir(self):
        # Sibling '<image_directory>_clahe_npy'. Populated lazily, atomic writes for worker safety.
        return self.image_directory.with_name(self.image_directory.name + "_clahe_npy")

    def _load(self, image_name):
        """Return float32 [0,1] image, CLAHE'd if self.clahe (cached on first read)."""
        if self.clahe:
            cpath = self._clahe_cache_dir() / f"{Path(image_name).stem}.npy"
            if cpath.exists():
                return np.load(cpath).astype(np.float32) / 65535.0                # uint16 cache -> float32 [0,1]
        img = cv2.imread(str(self.image_directory / image_name), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Cannot load {self.image_directory / image_name}")
        img = img.astype(np.float32)
        if img.max() > 0:
            img = img / img.max()
        if self.clahe:
            from src.data.augmentation import apply_clahe
            import os
            img = apply_clahe(img)
            cdir = self._clahe_cache_dir()
            cdir.mkdir(parents=True, exist_ok=True)
            cpath = cdir / f"{Path(image_name).stem}.npy"
            tmp = cpath.parent / f"{cpath.name}.tmp.{os.getpid()}.npy"
            np.save(tmp, (np.clip(img, 0.0, 1.0) * 65535).astype(np.uint16))       # store uint16 (halves disk vs float32, lossless for CLAHE output)
            tmp.replace(cpath)
        return img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row["study_id"]
        xray_image = self._load(image_name)

        # Check if cxr has nodule, if not, then all metrics are initialized to 0
        has_nodule = row["state"] != "non-nodule" and not pd.isna(row["x"])
        cx, cy, radius_px, size_mm = 0.0, 0.0, 0.0, 0.0

        if has_nodule:
            cx = float(row["x"])
            cy = float(row["y"])
            size_mm = float(row["size"])
            radius_px = size_mm / self.pixel_spacing_mm / 2.0

        # If Augmentation is intended
        if self.augment:
            # Flip image with a 0.5 probability
            if np.random.rand() > 0.5:

                xray_image = np.flip(xray_image, axis=1).copy()

                if has_nodule:
                    cx = self.input_size - 1 - cx

            # all images are rotated with random magnitue (-5, 5)
            angle = float(np.random.uniform(1.0, 5.0)) * (1 if np.random.rand() > 0.5 else - 1)
            xray_image, cx, cy = _apply_rotation(xray_image, cx, cy, angle)

            # apply gaus noise with a 0.3 probability
            if np.random.rand() < 0.3:
                xray_image = np.clip(xray_image + np.random.randn(*xray_image.shape).astype(np.float32) * 0.01, 0.0, 1.0)

        return torch.from_numpy(xray_image).unsqueeze(0), {
            "has_nodule": has_nodule,
            "center_2048": (cx, cy),
            "radius_px": radius_px,
            "size_mm": size_mm,
            "filename": image_name,
        }
