"""End-to-end smoke test for the Node21-only pipeline additions.

Exercises:
  1. downsampler registry has "node21_talebi"
  2. NoduleDetector builds with variant="node21_talebi" and forwards a fake batch
  3. TalebiPaperResizer produces target-resolution output
  4. src.data.nodule_sizes self-check (bin cutoffs + longest_diameter_mm)
  5. src.evaluation.size_stratified self-check (per-bin + macro + micro on fake preds)

No GPU, no filesystem, no real training data. Runs in seconds. Fails loud on any
regression. Called by run_node21_talebi_vs_bilinear.ps1 before the sweep."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def test_registry() -> None:
    from src.core.registry import DOWNSAMPLER_REGISTRY
    import src.models.downsamplers          # trigger registrations
    assert "node21_talebi" in DOWNSAMPLER_REGISTRY.list_registered(), (
        f"'node21_talebi' missing from registry. Have: {DOWNSAMPLER_REGISTRY.list_registered()}"
    )
    assert "bilinear" in DOWNSAMPLER_REGISTRY.list_registered()
    print("  [ok] registry has node21_talebi + bilinear")


def test_paper_resizer_forward() -> None:
    from src.models.downsamplers import TalebiPaperResizer
    resizer = TalebiPaperResizer(in_channels=1, target_size=224).eval()
    x = torch.randn(1, 1, 2048, 2048)
    with torch.no_grad():
        y = resizer(x)
    assert y.shape == (1, 1, 224, 224), f"unexpected output shape {y.shape}"
    assert torch.isfinite(y).all(), "TalebiPaperResizer produced non-finite output"
    print(f"  [ok] TalebiPaperResizer forward: {tuple(x.shape)} -> {tuple(y.shape)}")


def test_both_variants_agree_at_target_224() -> None:
    """Regression check: bilinear and node21_talebi must run at the same target_size
    (previously defaulted to 512 vs 224 → invalid comparison). Heatmap must match."""
    from src.models.detector import NoduleDetector
    x = torch.randn(1, 1, 2048, 2048)
    shapes = {}
    for variant in ("bilinear", "node21_talebi"):
        model = NoduleDetector(variant=variant, head="centernet", target_size=224).eval()
        with torch.no_grad():
            hm = model(x)
        assert torch.isfinite(hm).all(), f"{variant} produced non-finite heatmap"
        shapes[variant] = tuple(hm.shape)
        print(f"  [ok] NoduleDetector({variant}, target=224) heatmap {shapes[variant]}")
    assert shapes["bilinear"] == shapes["node21_talebi"], (
        f"variant heatmap mismatch (comparison would be invalid): {shapes}"
    )
    print(f"  [ok] both variants agree at target=224 (heatmap {shapes['bilinear']})")


def test_nodule_sizes_self_check() -> None:
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "src.data.nodule_sizes"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), (
        f"nodule_sizes self-check failed: rc={r.returncode} out={r.stdout!r} err={r.stderr!r}"
    )
    print("  [ok] src.data.nodule_sizes self-check")


def test_size_stratified_self_check() -> None:
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "src.evaluation.size_stratified"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), (
        f"size_stratified self-check failed: rc={r.returncode} out={r.stdout!r} err={r.stderr!r}"
    )
    print("  [ok] src.evaluation.size_stratified self-check")


def main() -> None:
    print("Node21 pipeline smoke test")
    print("-" * 40)
    test_registry()
    test_nodule_sizes_self_check()
    test_size_stratified_self_check()
    test_paper_resizer_forward()
    test_both_variants_agree_at_target_224()
    print("-" * 40)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
