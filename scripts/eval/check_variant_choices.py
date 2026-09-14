"""
Every --variant the CLI accepts must actually build a model.

WHY. scripts/train.py used to restate the variant list by hand. It drifted twice, in
opposite directions, and neither failure was visible until a run died:

  - 2026-09-06: four newly registered resizers (bilinear_antialias, bicubic_antialias,
    node21_talebi_zeroinit_aa, node21_talebi_zeroinit_attn_aa) were missing from the list,
    so a 12-run sweep passed the smoke test and then died at run 1 on "invalid choice".
  - Long-standing: five names in the list - alpha, gamma, delta, saliency_talebi,
    inverse_bone_talebi - were NOT registry keys. argparse accepted them and the registry
    then raised KeyError. Dead options that looked supported.

VARIANTS is now derived from DOWNSAMPLER_REGISTRY plus an explicit alias list, so both
directions are closed by construction. This asserts that stays true, and that every choice
survives an actual NoduleDetector build rather than merely existing as a key.

Run: python scripts/eval/check_variant_choices.py
"""
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.core.registry import DOWNSAMPLER_REGISTRY
from src.models.detector import NoduleDetector
from scripts.train import VARIANTS, VARIANT_ALIASES

TARGET_SIZE = 224


def main():
    registered = set(DOWNSAMPLER_REGISTRY.list_registered())
    offered = set(VARIANTS)

    missing = registered - offered
    assert not missing, \
        "registered but not offered by --variant: {} (VARIANTS has drifted)".format(sorted(missing))

    unregistered = offered - registered - set(VARIANT_ALIASES)
    assert not unregistered, \
        "offered by --variant but not registered and not aliased: {}".format(sorted(unregistered))

    print("{} variants offered, {} registered + {} aliased".format(
        len(offered), len(registered), len(VARIANT_ALIASES)))

    # Existing as a key is not the same as being usable. Build each one and push a tensor
    # through it - that is what a run actually does, and it is where a bad signature shows.
    image = torch.randn(1, 1, 512, 512)
    failures, ignores_target = [], []

    for variant in sorted(offered):
        try:
            model = NoduleDetector(variant=variant, target_size=TARGET_SIZE, head="fcos")
            model.eval()

            with torch.no_grad():
                resized = model.ldm(image)

            if resized.shape[-1] != TARGET_SIZE:
                ignores_target.append((variant, resized.shape[-1]))

        except Exception as error:
            failures.append((variant, "{}: {}".format(type(error).__name__, str(error)[:80])))

    # These five predate the target_size plumbing and hardcode 128. Harmless while nothing
    # runs them, DANGEROUS if anything does: scripts/train.py writes args.target_size into
    # metrics.json unconditionally, so `--variant lip --target-size 224` would be filed as a
    # 224 run while actually training at 128, and would then be compared against real 224
    # runs. Listed rather than asserted away so the hazard stays visible.
    LEGACY_IGNORES_TARGET = {"blurpool", "learned_kernel", "lip", "residual", "weighted_pooling"}

    # The variants this project actually runs MUST honour target_size - the resolution sweep
    # is the whole experiment, and a variant that quietly pins its own output would produce
    # a flat line that looks like a finding.
    active = {v for v in offered if v.startswith(("bilinear", "bicubic", "node21_talebi"))}
    broken_active = [(v, size) for v, size in ignores_target if v in active]
    failed_active = [(v, why) for v, why in failures if v in active]

    assert not failed_active, "active variant(s) cannot run: {}".format(failed_active)
    assert not broken_active, \
        "active variant(s) ignore target_size: {} - the resolution sweep would be a lie".format(
            broken_active)

    unexpected = [(v, s) for v, s in ignores_target if v not in LEGACY_IGNORES_TARGET]
    assert not unexpected, "new variant(s) ignoring target_size: {}".format(unexpected)

    print("ok - every active variant builds and honours target_size={}".format(TARGET_SIZE))

    if ignores_target:
        print("\nlegacy variants that IGNORE --target-size (never pass these to a sweep;")
        print("metrics.json would record the requested size, not the one trained at):")

        for variant, size in sorted(ignores_target):
            print("  {:<24} always outputs {}".format(variant, size))

    if failures:
        print("\nlegacy variants that do not build at all:")

        for variant, reason in sorted(failures):
            print("  {:<24} {}".format(variant, reason))


if __name__ == "__main__":
    main()
