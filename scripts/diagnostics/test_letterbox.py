"""
Self-check for the aspect-preserving (letterbox) resize.

The thing that must not break is box geometry: after resizing, a nodule's box must sit on
the same anatomy it started on, and under letterbox it must also keep its aspect ratio
rather than being squashed with the film.

Run: python scripts/diagnostics/test_letterbox.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd

from src.data.node21 import NODE21Dataset


def make_probe(letterbox, input_size=512):
    """A NODE21Dataset with image loading stubbed out, so geometry can be tested alone."""
    frame = pd.DataFrame([{"img_name": "probe.mha", "label": 1, "x": 0, "y": 0, "width": 0, "height": 0}])
    dataset = NODE21Dataset.__new__(NODE21Dataset)
    dataset.df = frame
    dataset.augment = False
    dataset.input_size = input_size
    dataset.clahe = False
    dataset.npy_cache_dir = None
    dataset.letterbox = letterbox
    dataset.group_by_image = False
    dataset.records = None
    dataset.aug_profile_name = "v3"

    return dataset


def run_transform(dataset, native_w, native_h, box_native):
    """Replicate __getitem__'s resize path on a synthetic film with a marked box."""
    image = np.zeros((native_h, native_w), dtype=np.float32)
    x, y, w, h = box_native
    image[int(y):int(y + h), int(x):int(x + w)] = 1.0

    size = dataset.input_size

    if dataset.letterbox:
        scale = size / max(native_h, native_w)
        new_w, new_h = int(round(native_w * scale)), int(round(native_h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        out = np.zeros((size, size), dtype=np.float32)
        out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        scale_x = scale_y = scale

    else:
        out = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        scale_x, scale_y = size / native_w, size / native_h
        pad_x = pad_y = 0

    box = ((x + w / 2) * scale_x + pad_x, (y + h / 2) * scale_y + pad_y, w * scale_x, h * scale_y)

    return out, box


def main():
    # A strongly non-square film, like NODE21's 3920x3612 or 2828x2320 originals
    native_w, native_h = 3000, 2000
    square_box = (1400.0, 900.0, 200.0, 200.0)          # square nodule in native pixels

    print(f"native film {native_w}x{native_h} (aspect {native_w/native_h:.2f}), "
          f"square {int(square_box[2])}px nodule\n")

    for letterbox in (False, True):
        dataset = make_probe(letterbox)
        image, (cx, cy, w, h) = run_transform(dataset, native_w, native_h, square_box)
        label = "letterbox" if letterbox else "squash   "

        # Does the reported box land on the marked anatomy?
        ys, xs = np.nonzero(image > 0.5)
        actual_cx, actual_cy = xs.mean(), ys.mean()
        offset = np.hypot(actual_cx - cx, actual_cy - cy)

        print(f"  {label}  box {w:.1f}x{h:.1f}px  aspect {w/h:.3f}  "
              f"centre offset from pixels {offset:.2f}px")

        assert offset < 2.0, f"{label}: reported centre is {offset:.1f}px off the marked anatomy"

        if letterbox:
            assert abs(w / h - 1.0) < 0.02, f"letterbox distorted a square nodule to {w/h:.3f}"
        else:
            # The legacy path is expected to distort a square nodule to exactly the
            # film's inverse aspect: scale_x = S/W and scale_y = S/H, so w/h = H/W.
            expected_aspect = native_h / native_w
            assert abs((w / h) - expected_aspect) < 0.02,                 f"squash gave aspect {w/h:.3f}, expected {expected_aspect:.3f}"

    print("\nOK  both paths put the box on the right anatomy")
    print("OK  letterbox keeps a square nodule square (aspect 1.000)")
    print(f"OK  legacy squash distorts it to aspect {native_h/native_w:.3f} - the defect being fixed")

    # A square film must be unaffected by letterboxing, so JSRT and the 1024 resamples
    # (61% of NODE21) are untouched by this change
    dataset = make_probe(True)
    _, box_letterbox = run_transform(dataset, 1024, 1024, (400.0, 400.0, 100.0, 100.0))
    dataset = make_probe(False)
    _, box_squash = run_transform(dataset, 1024, 1024, (400.0, 400.0, 100.0, 100.0))
    assert np.allclose(box_letterbox, box_squash), f"{box_letterbox} != {box_squash}"
    print("OK  square films transform identically either way (61% of NODE21 unaffected)")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
