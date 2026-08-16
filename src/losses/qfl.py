import torch
import torch.nn.functional as F
from src.core.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("qfl")
def quality_focal_loss(pred, target, beta=2.0):
    """
    QFL (adapted from Li et al. 2020): continuous Gaussian target as quality supervision.
    No hard pos/neg split; pixels with target=0.5 (Gaussian tail) get moderate weighting.
    """
    pred   = torch.sigmoid(pred.float()).clamp(1e-6, 1 - 1e-6)
    target = target.float()
    scale  = (target - pred).abs().pow(beta)
    loss   = F.binary_cross_entropy(pred, target, reduction='none') * scale
    n_pos  = (target > 0.01).sum().clamp(min=1)
    return loss.sum() / n_pos
