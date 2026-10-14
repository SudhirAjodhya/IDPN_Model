"""
Self-check for the backbone freeze/unfreeze phase.

Verifies the three things that make a freeze real: weights do not move, BatchNorm
running statistics do not drift, and everything resumes training after unfreezing.

Run: python scripts/diagnostics/test_freeze.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
import torch

from src.models.detector import NoduleDetector


def snapshot(module):
    """Weights plus BatchNorm running stats, so a drifting-stats 'freeze' is caught."""
    weights = {name: p.detach().clone() for name, p in module.named_parameters()}
    buffers = {name: b.detach().clone() for name, b in module.named_buffers() if b.dtype.is_floating_point}

    return weights, buffers


def max_change(before, after):
    weights_before, buffers_before = before
    weights_after, buffers_after = after

    weight_delta = max((weights_after[k] - v).abs().max().item() for k, v in weights_before.items())
    buffer_delta = max([(buffers_after[k] - v).abs().max().item() for k, v in buffers_before.items()], default=0.0)

    return weight_delta, buffer_delta


def run_steps(model, backbone, n_steps, trainable):
    """Mimic what train_loop does per epoch, including the model.train() call."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)     # deliberately large

    for parameter in backbone.parameters():
        parameter.requires_grad = trainable

    backbone.train(trainable)

    for _ in range(n_steps):
        model.train()

        # This is the guard inside train_one_epoch: model.train() would otherwise
        # re-enable BatchNorm updates in a frozen backbone
        if not any(p.requires_grad for p in backbone.parameters()):
            backbone.eval()

        out = model(torch.randn(2, 1, 256, 256))
        loss = sum(v.float().pow(2).mean() for v in out.values()) if isinstance(out, dict) else out.float().pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    torch.manual_seed(0)
    model = NoduleDetector(variant="bilinear", head="centernet_det", target_size=224, backbone="resnet50_fpn")
    backbone = model.backbone_fpn.features

    n_bn = sum(1 for m in backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    print(f"backbone: {sum(p.numel() for p in backbone.parameters()):,} params, {n_bn} BatchNorm layers")

    # ---- frozen phase ----
    before = snapshot(backbone)
    head_before = snapshot(model.head)
    run_steps(model, backbone, n_steps=3, trainable=False)
    weight_delta, buffer_delta = max_change(before, snapshot(backbone))

    assert weight_delta == 0.0, f"frozen backbone weights moved by {weight_delta}"
    assert buffer_delta == 0.0, f"frozen backbone BN running stats drifted by {buffer_delta}"
    print(f"OK  frozen: weights moved {weight_delta:.2e}, BN stats moved {buffer_delta:.2e}")

    head_weight_delta, _ = max_change(head_before, snapshot(model.head))
    assert head_weight_delta > 0, "head should still train while the backbone is frozen"
    print(f"OK  head still trains while frozen (moved {head_weight_delta:.2e})")

    # ---- unfrozen phase ----
    before = snapshot(backbone)
    run_steps(model, backbone, n_steps=3, trainable=True)
    weight_delta, buffer_delta = max_change(before, snapshot(backbone))

    assert weight_delta > 0, "backbone weights should move after unfreezing"
    assert buffer_delta > 0, "BN running stats should update again after unfreezing"
    print(f"OK  unfrozen: weights moved {weight_delta:.2e}, BN stats moved {buffer_delta:.2e}")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
