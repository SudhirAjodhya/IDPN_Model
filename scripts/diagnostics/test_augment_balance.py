"""
Self-checks for the augmentation geometry and the positive-oversampling sampler.

Run: python scripts/diagnostics/test_augment_balance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.node21 import NODE21Dataset


def make_probe_dataset(input_size=256):
    """A NODE21Dataset with augmentation on, whose image loading is stubbed out."""
    frame = pd.DataFrame([{"img_name": "probe.mha", "label": 1, "x": 0, "y": 0, "width": 20, "height": 20}])
    dataset = NODE21Dataset.__new__(NODE21Dataset)
    dataset.df = frame
    dataset.augment = True
    dataset.input_size = input_size
    dataset.clahe = False
    dataset.npy_cache_dir = None

    return dataset


def test_label_tracks_the_pixels():
    """
    The returned centre must land on the same anatomy the warp moved. Plant a bright
    marker at the nodule centre, augment, and confirm the marker is where the label says.
    """
    dataset = make_probe_dataset()
    size = dataset.input_size
    failures = 0

    for trial in range(200):
        np.random.seed(trial)

        cx, cy = float(np.random.uniform(40, size - 40)), float(np.random.uniform(40, size - 40))
        image = np.zeros((size, size), dtype=np.float32)
        cv2.circle(image, (int(round(cx)), int(round(cy))), 3, 1.0, -1)

        augmented, new_cx, new_cy = dataset._augment(image, cx, cy, has_nodule=True)

        # Brightest surviving pixel should sit on the reported centre
        peak_y, peak_x = np.unravel_index(int(np.argmax(augmented)), augmented.shape)
        offset = np.hypot(peak_x - new_cx, peak_y - new_cy)

        if offset > 4.0: failures += 1

    assert failures == 0, f"label drifted from the pixels on {failures}/200 trials"
    print("OK  label tracks pixels through flip + rotation + translation")


def test_nodule_never_leaves_the_frame():
    """Border nodules must stay inside the frame, or the GT heatmap silently goes empty."""
    dataset = make_probe_dataset()
    size = dataset.input_size
    image = np.zeros((size, size), dtype=np.float32)

    for trial in range(500):
        np.random.seed(trial)

        # Deliberately adversarial: nodules pinned right against each edge
        cx = float(np.random.choice([2.0, size - 3.0, size / 2.0]))
        cy = float(np.random.choice([2.0, size - 3.0, size / 2.0]))

        _, new_cx, new_cy = dataset._augment(image, cx, cy, has_nodule=True)

        assert 0 <= new_cx < size, f"cx {new_cx} left the frame from {cx}"
        assert 0 <= new_cy < size, f"cy {new_cy} left the frame from {cy}"

    print("OK  nodule centres stay in frame across 500 adversarial border draws")


def test_sampler_hits_the_requested_share():
    """WeightedRandomSampler weights must produce the requested positive fraction."""
    num_positive, num_negative = 1476, 3748
    positive_flags = [True] * num_positive + [False] * num_negative

    for target_fraction in (0.45, 0.5, 0.28):
        weight_positive = target_fraction / num_positive
        weight_negative = (1.0 - target_fraction) / num_negative
        sample_weights = [weight_positive if is_positive else weight_negative for is_positive in positive_flags]

        generator = torch.Generator().manual_seed(0)
        sampler = WeightedRandomSampler(sample_weights, num_samples=200_000, replacement=True, generator=generator)
        drawn = np.fromiter(iter(sampler), dtype=np.int64)
        observed = float((drawn < num_positive).mean())

        assert abs(observed - target_fraction) < 0.005, f"asked {target_fraction}, drew {observed:.4f}"
        print(f"OK  pos_frac={target_fraction:.2f} -> observed {observed:.4f}")


if __name__ == "__main__":
    test_label_tracks_the_pixels()
    test_nodule_never_leaves_the_frame()
    test_sampler_hits_the_requested_share()
    print("\nall checks passed")
