import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import DOWNSAMPLER_REGISTRY

@DOWNSAMPLER_REGISTRY.register("bilinear")
class BilinearBaseline(nn.Module):
    """
    Standard bilinear downsampling baseline
    """
    def __init__(self, target_size=512):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
