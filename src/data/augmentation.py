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


def resize_preserving_detail(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Resize a film, using an anti-aliasing kernel whenever the image is being shrunk.

    cv2.INTER_LINEAR does not low-pass filter before decimating, so shrinking with it
    aliases: high-frequency content folds down into the low frequencies rather than
    being removed. 63% of NODE21 films are larger than a 1024 input and are therefore
    downscaled here, and VinDr films are larger still at 2128-3408 px natively.

    This matters more in this project than it usually would. The research question is
    whether a learned resizer can beat bilinear at exactly the decimation step where
    Nyquist is violated. Aliasing the film in the DATALOADER as well would put the same
    defect on both sides of that comparison and confound it, so the dataloader
    decimates cleanly and leaves the aliasing question to the resizer under test.

    INTER_AREA is used only when shrinking. It degrades on upscaling, where
    INTER_LINEAR remains correct.

    Args:
        image: (H, W) array
        size: destination (width, height), matching cv2's argument order

    Returns:
        The resized array.
    """
    destination_width, destination_height = size
    source_height, source_width = image.shape[:2]

    shrinking = destination_width * destination_height < source_width * source_height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR

    return cv2.resize(image, size, interpolation=interpolation)


def apply_clahe(img_float01: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a float32 [0,1] grayscale image, returning float32 [0,1].

    OpenCV's CLAHE only accepts uint8/uint16, so we round-trip through uint16
    for maximum precision (12-bit dynamic range preserved). Overhead is <2ms
    per 1024x1024 image on modern CPUs - cheap enough to always leave on."""
    img_u16 = (np.clip(img_float01, 0.0, 1.0) * 65535).astype(np.uint16)
    equalized = _CLAHE.apply(img_u16)
    return equalized.astype(np.float32) / 65535.0
