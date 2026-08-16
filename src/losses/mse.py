import torch
import torch.nn.functional as F
from src.core.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register("mse")
def mse_heatmap_loss(pred, target):
    """MSE on sigmoid(pred) vs target. Weights all pixel deviations equally."""
    pred = torch.sigmoid(pred.float())
    return F.mse_loss(pred, target.float())
