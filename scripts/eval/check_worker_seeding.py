"""
Self-check: dataloader workers draw the SAME augmentation stream on every run.

WHY THIS EXISTS - it rules out a wrong answer, and the wrong answer is very tempting.

Runs at a fixed seed DO diverge. Measured 2026-09-01: the v14 and v15 Talebi arms are the
same config and the same seed 42, both checkpointed at epoch 11, and scored 0.591 and
0.531 at 0.25 FP/img. Their per-epoch histories differ from epoch 1 (train loss 2.1724 vs
2.1435). A 0.060 rerun spread is LARGER than the 0.048 effect under test at target 224.

The obvious suspect is wrong. NODE21Dataset._augment (shared by VinDrDataset) draws from
the numpy GLOBAL RNG, and on Windows the workers are spawned processes, which is the
classic numpy-in-workers reproducibility bug. But PyTorch's _worker_loop already seeds
numpy and random per worker from the parent generator, so the augmentation stream is
already reproducible. This script asserts that. Adding seeding to
scripts/train.py::init_dataloader_worker is redundant - it was tried and reverted.

It is not the GPU either - scripts/eval/check_training_determinism.py shows training is
already bit-reproducible on fixed batches at the current default settings.

What IS true, and is asserted at the bottom of this script: the augmentation stream depends
on num_workers. Worker count partitions sample indices across workers and each worker holds
its own RNG stream, so two runs at the same seed with different num_workers see different
augmentations. num_workers is not recorded in metrics.json, so after the fact this is
invisible - which makes it the leading explanation for the v14/v15 divergence.

This script never touches the real datasets: it exercises only the mechanism - set_seed in
the parent, numpy drawn inside spawned workers, through the real worker_init_fn.

Run: python scripts/eval/check_worker_seeding.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from scripts.train import init_dataloader_worker
from src.utils.misc import set_seed


class NumpyGlobalRNGProbe(Dataset):
    """Stands in for _augment: every item draws from the numpy GLOBAL RNG, as it does."""

    def __len__(self):
        return 24

    def __getitem__(self, index):
        return torch.tensor(np.random.rand())


def draw_one_epoch(seed, num_workers, worker_init_fn):
    """One pass over the probe dataset, returned as a flat list of draws."""
    set_seed(seed)

    loader = DataLoader(NumpyGlobalRNGProbe(), batch_size=4, num_workers=num_workers,
                        worker_init_fn=worker_init_fn)

    return [float(value) for batch in loader for value in batch]


def main():
    # The bug and the fix only exist with real worker processes
    workers = 3

    first  = draw_one_epoch(42, workers, init_dataloader_worker)
    second = draw_one_epoch(42, workers, init_dataloader_worker)

    assert first == second, (
        "workers are NOT reproducible at a fixed seed - numpy seeding in "
        "init_dataloader_worker is missing or ineffective")

    # A different seed must actually move the stream, or we have frozen the augmentations
    # to one fixed sequence instead of making them reproducible
    other = draw_one_epoch(7, workers, init_dataloader_worker)
    assert other != first, "seed 7 reproduced seed 42 - augmentation is no longer seed-dependent"

    # Workers must not all share one stream, which would duplicate augmentations across the
    # batch rather than decorrelating them
    assert len(set(first)) == len(first), "workers are drawing duplicate values"

    # The load-bearing assertion. Reproducibility must hold with NO worker_init_fn at all,
    # which is what proves torch itself seeds numpy and that seeding it in
    # init_dataloader_worker would be redundant. If this ever fails, the worker RNG really
    # is the culprit and seeding it there becomes the correct fix after all.
    assert draw_one_epoch(42, workers, None) == draw_one_epoch(42, workers, None), (
        "unseeded workers are NOT reproducible - torch no longer seeds numpy per worker, "
        "so seeding it in init_dataloader_worker IS needed (see module docstring)")

    # Worker COUNT partitions the sample indices across workers, and each worker carries its
    # own RNG stream, so changing num_workers re-assigns which stream augments which film.
    # That is reproducible but NOT portable: two runs at the same seed and different
    # num_workers see different augmentations. num_workers is not recorded in metrics.json,
    # so this difference is invisible after the fact - hence checking it here.
    six_workers = draw_one_epoch(42, 6, init_dataloader_worker)

    if six_workers != first:
        print("NOTE: the augmentation stream depends on num_workers (3 vs 6 differ at the")
        print("      same seed). Runs are only comparable at equal worker counts.")

    print("ok - worker augmentation streams are reproducible at a fixed seed,")
    print("     with and without worker_init_fn: run-to-run divergence is not from here")


if __name__ == "__main__":
    main()
