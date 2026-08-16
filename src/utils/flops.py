"""
FLOPS + parameter counting for Talebi Table 4 reproduction.

Uses PyTorch's built-in torch.utils.flop_counter (torch >= 2.1), which counts
multiply-accumulate operations (MACs). Paper Table 4's "Total FLOPS (Billions)"
column reports the same convention (MACs, called FLOPS in vision literature).
We report both raw MACs and 2 * MACs (= true FLOPs) for downstream comparisons
that might use either convention.
"""
from __future__ import annotations
import logging
import torch
from torch.utils.flop_counter import FlopCounterMode

# We don't use triton kernels (pure Conv/BN/ReLU nets); silence the noisy
# warning the flop counter emits on every invocation when triton is absent.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


def count_flops_and_params(model, input_shape, device):
    """
    Return {"macs": float, "flops": float, "params": int} for `model` on a
    single dummy sample of `input_shape` (e.g. (3, 368, 368)) on `device`.

    macs   = multiply-accumulate ops (paper's "Total FLOPS" convention)
    flops  = 2 * macs (Nvidia / peak-throughput convention)
    params = total trainable parameter count
    """
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(dummy)
    macs = float(counter.get_total_flops())  # torch's "flops" == MACs
    params = sum(p.numel() for p in model.parameters())

    if was_training:
        model.train()

    return {
        "macs": macs,
        "flops": macs * 2.0,
        "params": params,
    }
