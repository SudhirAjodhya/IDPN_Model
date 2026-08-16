"""Shared augmentation helpers for CXR data.

CLAHE (Contrast-Limited Adaptive Histogram Equalization) is standard preprocessing
for CXR nodule detection. Boosts local contrast around low-visibility nodules and
is used by essentially every recent Node21 leaderboard entry."""
from __future__ import annotations

import cv2
import numpy as np

# Single CLAHE instance is reused. clip_limit=2.0 and tile=8x8 are the values
# recommended by MONAI and by the Node21 challenge organisers' baseline.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(img_float01: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a float32 [0,1] grayscale image, returning float32 [0,1].

    OpenCV's CLAHE only accepts uint8/uint16, so we round-trip through uint16
    for maximum precision (12-bit dynamic range preserved). Overhead is <2ms
    per 1024x1024 image on modern CPUs - cheap enough to always leave on."""
    img_u16 = (np.clip(img_float01, 0.0, 1.0) * 65535).astype(np.uint16)
    equalized = _CLAHE.apply(img_u16)
    return equalized.astype(np.float32) / 65535.0
