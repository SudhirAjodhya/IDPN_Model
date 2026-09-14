"""
Self-check for the anti-aliased and bicubic downsampler variants added 2026-09-04.

Two things must hold, and the first matters more than the second:

  1. BACKWARD COMPATIBILITY. `bilinear`, `node21_talebi_zeroinit` and
     `node21_talebi_zeroinit_attn` must behave EXACTLY as before the skip_mode /
     skip_antialias parameters were threaded through TalebiPaperResizer. Every run banked
     to date used them, and a silent change would make new runs incomparable to old ones
     without anything in metrics.json recording it.

  2. THE NEW ARMS ACTUALLY DIFFER. An anti-aliased skip must produce a visibly different
     image from an aliased one, or the comparison is measuring nothing.

Also reports the aliasing energy each classical downsampler leaves behind, measured on a
high-frequency chequerboard - the signal that decimation without a low-pass folds back into
the image. That number is the reason the anti-aliased baseline exists.

Run: python scripts/eval/check_antialias_variants.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.core.registry import DOWNSAMPLER_REGISTRY
import src.models.downsamplers                                        # noqa: F401  (registers)

INPUT_SIZE = 1024
TARGET_SIZE = 224


def build(variant, target_size=TARGET_SIZE):
    return DOWNSAMPLER_REGISTRY.get(variant)(target_size=target_size)


def chequerboard(size=INPUT_SIZE):
    """Highest-frequency pattern the input grid can carry - alternating single pixels."""
    row = torch.arange(size)
    board = ((row[None, :] + row[:, None]) % 2).float()

    return board[None, None]


def main():
    torch.manual_seed(0)
    image = torch.rand(1, 1, INPUT_SIZE, INPUT_SIZE)

    # ---- 1. Backward compatibility: defaults must reproduce the pre-change behaviour ----
    reference = F.interpolate(image, size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear",
                              align_corners=False)

    assert torch.equal(build("bilinear")(image), reference), \
        "bilinear moved - every banked run used this exact call"

    for variant in ("node21_talebi_zeroinit", "node21_talebi_zeroinit_attn"):
        module = build(variant)

        assert module.skip_mode == "bilinear" and module.skip_antialias is False, \
            "{} default skip changed - banked runs are no longer comparable".format(variant)

        # The skip is what must be unchanged; the residual is randomly initialised, so
        # compare the skip path directly rather than the module output.
        skip = F.interpolate(image, size=(TARGET_SIZE, TARGET_SIZE), mode=module.skip_mode,
                             align_corners=False, antialias=module.skip_antialias)

        assert torch.equal(skip, reference), "{} skip path moved".format(variant)

    print("ok - bilinear, node21_talebi_zeroinit and _attn are unchanged by the new parameters")

    # ---- 2. The new arms must actually differ ----
    antialiased = build("bilinear_antialias")(image)
    difference = (antialiased - reference).abs().mean().item()

    assert difference > 1e-4, \
        "bilinear_antialias is identical to bilinear - antialias flag had no effect"

    for variant, expected_mode, expected_aa in (
            ("node21_talebi_zeroinit_aa", "bilinear", True),
            ("node21_talebi_zeroinit_attn_aa", "bilinear", True)):
        module = build(variant)

        assert module.skip_mode == expected_mode and module.skip_antialias is expected_aa, \
            "{} is not configured with an anti-aliased skip".format(variant)

    # At init the residual is near zero, so the AA arm must start at anti-aliased bilinear
    with torch.no_grad():
        started_at = build("node21_talebi_zeroinit_aa")(image)

    drift = (started_at - antialiased).abs().mean().item()

    assert drift < 5e-2, \
        "zeroinit_aa does not start at its own baseline (drift {:.4f})".format(drift)

    print("ok - anti-aliased arms differ from plain bilinear (mean |diff| {:.5f})".format(difference))
    print("ok - node21_talebi_zeroinit_aa starts at anti-aliased bilinear (drift {:.5f})".format(drift))

    # ---- 3. Aliasing energy, the reason any of this exists ----
    board = chequerboard()

    print("\nDecimating a {}px chequerboard to {} ({:.1f}x). A perfect low-pass returns a".format(
        INPUT_SIZE, TARGET_SIZE, INPUT_SIZE / TARGET_SIZE))
    print("flat 0.5 field; residual std IS the aliasing folded back in.\n")
    print("  {:<26} {:>10}  {:>10}".format("variant", "std", "mean"))

    for variant in ("bilinear", "bilinear_antialias", "bicubic", "bicubic_antialias"):
        out = build(variant)(board)
        print("  {:<26} {:>10.5f}  {:>10.5f}".format(variant, out.std().item(), out.mean().item()))

    print("\nok")


if __name__ == "__main__":
    main()
