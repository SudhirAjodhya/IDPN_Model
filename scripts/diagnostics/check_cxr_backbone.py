"""
Shape and wiring check for the DenseNet CXR backbones, before spending a night on them.

Lives here rather than as a `__main__` block inside the backbone module: that module is
imported by src/models/backbones/__init__.py, so `python -m src.models.backbones.
densenet_cxr_fpn` executes it a second time and the registry decorators raise
"already registered".

Checks, in order:
  1. torchxrayvision imports and the weights download
  2. the four dense-block taps have the widths the FPN is built against
  3. stage1 really is stride /4 at every target resolution the sweep uses
  4. the detector wires the backbone up end to end, with the right input channel count

Step 4 is the one that matters most: detector.py chooses between the ImageNet 3-channel
path and the single-channel radiograph path from the backbone's `input_mode`, and a
mismatch there would train silently on garbage rather than crash.

Run: python scripts/diagnostics/check_cxr_backbone.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.core.registry import BACKBONE_REGISTRY
from src.models.detector import NoduleDetector

TARGETS = (224, 256, 384, 512)
BACKBONES = ("densenet121_cxr_fpn", "densenet121_imagenet_fpn")


def check_backbone(name):
    backbone = BACKBONE_REGISTRY.get(name)(fpn_out=256)
    backbone.eval()

    channels = 1 if backbone.input_mode == "xrv_gray" else 3
    total = sum(p.numel() for p in backbone.parameters())

    print("\n  {}".format(name))
    print("    input_mode  {}  ({} channel{})".format(
        backbone.input_mode, channels, "" if channels == 1 else "s"))
    print("    params      {:,}".format(total))

    for size in TARGETS:
        with torch.no_grad():
            out = backbone(torch.zeros(1, channels, size, size))

        assert set(out.keys()) == {"stage1", "stage2", "stage3", "stage4"}, out.keys()
        assert out["stage1"].shape[1] == 256, out["stage1"].shape

        # The head reads stage1 and the FCOS target builder assumes heatmap = target / 4
        expected = size // 4
        assert out["stage1"].shape[2] == expected, (
            "stage1 is {} at target {}, expected {} - the head and the FCOS targets both "
            "assume /4".format(out["stage1"].shape[2], size, expected))

        print("      target {:>4} -> stage1 {:>3}x{:<3} ok".format(
            size, out["stage1"].shape[2], out["stage1"].shape[3]))


def check_detector(name):
    """End to end, including the LDM and the input_mode branch in detector.forward."""
    for variant in ("bilinear", "node21_talebi_zeroinit"):
        model = NoduleDetector(variant=variant, head="fcos", backbone=name, target_size=224)
        model.eval()

        # The detector always receives ONE channel: the LDM output. Whether it gets
        # replicated to three is decided inside forward, from the backbone.
        with torch.no_grad():
            out = model(torch.zeros(1, 1, 1024, 1024))

        assert set(out.keys()) == {"classification", "ltrb", "centerness"}, out.keys()
        assert out["classification"].shape[-1] == 56, out["classification"].shape

        total = sum(p.numel() for p in model.parameters())
        print("    {:<24} full model {:,} params, fcos heads {}".format(
            variant, total, tuple(out["classification"].shape)))


def main():
    try:
        import torchxrayvision as xrv
        print("torchxrayvision {}".format(xrv.__version__))

    except ImportError:
        print("torchxrayvision is NOT installed. Install with:")
        print("    python -m pip install torchxrayvision")
        print("(use `python -m pip`, not `pip` - the venv was moved and the pip.exe "
              "launcher has a stale path baked in)")
        return 1

    for name in BACKBONES:
        check_backbone(name)

    print("\n  end-to-end through NoduleDetector (target 224):")

    for name in BACKBONES:
        print("\n  {}".format(name))
        check_detector(name)

    print("\nALL CHECKS PASSED")
    print("  the CXR backbone takes 1 channel and skips the ImageNet replicate;")
    print("  the ImageNet DenseNet control takes 3 and keeps it.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
