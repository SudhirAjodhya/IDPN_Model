import torch
from src.core.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("focal")
def focal_loss_heatmap(pred, target, alpha=2.0, beta=4.0):
    """v4 focal loss — kept as baseline option."""
    pred   = torch.sigmoid(pred.float()).clamp(1e-6, 1 - 1e-6)
    target = target.float()
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - target, beta) * neg_mask
    return (pos_loss.sum() + neg_loss.sum()) / pos_mask.sum().clamp(min=1)

@LOSS_REGISTRY.register("focal_v5")
def focal_loss_heatmap_v5(pred, target, alpha=2.0, beta=4.0):
    """Relaxed pos_mask: ge(0.99) instead of eq(1) to guard against fp precision.
    Matches matrix_v1 exactly — do not change the threshold without evidence."""
    pred   = torch.sigmoid(pred.float()).clamp(1e-6, 1 - 1e-6)
    target = target.float()
    pos_mask = target.ge(0.99).float()
    neg_mask = target.lt(0.99).float()
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - target, beta) * neg_mask
    return (pos_loss.sum() + neg_loss.sum()) / pos_mask.sum().clamp(min=1)
