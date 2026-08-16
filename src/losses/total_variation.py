import torch

def total_variation_loss(x):
    """TV loss on a 2D map to enforce smoothness."""
    if x is None:
        return torch.tensor(0.0, device=x.device if isinstance(x, torch.Tensor) else "cpu")
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w
