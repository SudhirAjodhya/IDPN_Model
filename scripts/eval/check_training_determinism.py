"""
Locate the source of run-to-run divergence at a fixed seed, and show what removes it.

Two runs with the same config and the same --seed do not reproduce each other. Measured
2026-09-01: the v14 and v15 Talebi arms (same config, seed 42, both checkpointed at epoch
11) scored 0.591 and 0.531 at 0.25 FP/img, with histories diverging from epoch 1. That
0.060 spread is larger than the +0.048 effect under test, so every single-seed delta in
the project sits inside it.

scripts/eval/check_worker_seeding.py already rules out the dataloader. This script tests
what is left: the GPU. It runs the REAL detector and the REAL FCOS loss for a handful of
optimiser steps on fixed synthetic batches, twice per configuration, and compares the
resulting weights bit for bit. Synthetic input is deliberate - it holds the data stream
exactly constant so that anything that still differs is the GPU and nothing else.

Configurations, weakest to strongest:
    baseline      what train.py does today (src/utils/misc.py set_seed - no cudnn control)
    cudnn         + cudnn.deterministic, cudnn.benchmark off (src/core/seeding.py)
    full          + torch.use_deterministic_algorithms, which also covers the atomics in
                  scatter/index reductions that cudnn flags do not touch

Run: python scripts/eval/check_training_determinism.py
"""
import os
import sys
from pathlib import Path

# MUST precede the torch import: cuBLAS reads this once, at first CUDA context creation,
# and without it use_deterministic_algorithms raises on any matmul-backed op.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.fcos_targets import make_fcos_targets
from src.losses.fcos import fcos_loss
from src.models.detector import NoduleDetector
from src.utils.misc import set_seed

STEPS = 6
BATCH = 2
INPUT_SIZE = 1024
TARGET_SIZE = 224
MAX_OBJECTS = 4


def fixed_batches(device):
    """Identical synthetic batches on every call, so only the GPU can introduce variation."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    batches = []

    for _ in range(STEPS):
        images = torch.rand(BATCH, 1, INPUT_SIZE, INPUT_SIZE, generator=generator)

        # make_fcos_targets expects cx, cy, w, h in pixels at the loader's input size -
        # NOT corners - and reads num_boxes to know how much of the padding is real
        centres = torch.randint(300, 700, (BATCH, MAX_OBJECTS, 2), generator=generator).float()
        sizes = torch.randint(40, 120, (BATCH, MAX_OBJECTS, 2), generator=generator).float()

        boxes = torch.cat([centres, sizes], dim=-1)
        num_boxes = torch.ones(BATCH, dtype=torch.long)

        batches.append((images.to(device),
                        {"boxes": boxes.to(device), "num_boxes": num_boxes.to(device)}))

    return batches


def train_a_few_steps(device):
    """Seed, build, and take STEPS optimiser steps. Returns a flat copy of the weights."""
    set_seed(42)

    model = NoduleDetector(variant="node21_talebi_zeroinit",
                           target_size=TARGET_SIZE, head="fcos").to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4)
    heatmap_size = TARGET_SIZE // 4

    for images, targets in fixed_batches(device):
        optimiser.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = model(images)
            gt = make_fcos_targets(targets, images.shape[0], heatmap_size,
                                   input_size=images.shape[-1], device=device,
                                   max_objects=targets["boxes"].shape[1])
            loss, _ = fcos_loss(prediction, gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

    return torch.cat([p.detach().flatten().float().cpu() for p in model.parameters()])


def configure(mode):
    """Apply one determinism configuration. Returns False if the mode cannot be used."""
    if mode == "baseline":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    elif mode == "cudnn":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    elif mode == "full":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

    return True


def main():
    if not torch.cuda.is_available():
        print("no CUDA - this check only means anything on the GPU")
        return

    device = torch.device("cuda")
    results = {}

    for mode in ("baseline", "cudnn", "full"):
        configure(mode)

        try:
            first = train_a_few_steps(device)
            second = train_a_few_steps(device)

        except RuntimeError as error:
            # use_deterministic_algorithms raises on any op with no deterministic kernel.
            # That is the whole cost of this option, so report the op rather than hiding it.
            print("{:<10} UNAVAILABLE - {}".format(mode, str(error).split("\n")[0][:150]))
            results[mode] = None
            continue

        identical = torch.equal(first, second)
        largest_gap = (first - second).abs().max().item()
        results[mode] = identical

        print("{:<10} {:<16} max weight difference {:.3e}".format(
            mode, "IDENTICAL" if identical else "DIVERGED", largest_gap))

    print()

    if results.get("baseline") is True:
        print("Baseline already reproduced. The divergence seen between v14 and v15 is not")
        print("reproduced by this probe - it needs more steps, or its cause is elsewhere.")

    elif results.get("full") is True or results.get("cudnn") is True:
        winner = "cudnn" if results.get("cudnn") else "full"
        print("Fix confirmed: '{}' makes same-seed training bit-reproducible.".format(winner))
        print("Wire it into scripts/train.py to remove the rerun variance component.")

    else:
        print("Nothing tested here fixed it. Do NOT assume seeds are the only variance.")


if __name__ == "__main__":
    main()
