import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import DOWNSAMPLER_REGISTRY

@DOWNSAMPLER_REGISTRY.register("lip")
class LIPResizer(nn.Module):
    """
    Local Importance-based Pooling (Gao et al. 2019, ICCV).
    Content-aware downsampling: 2048 -> 1024 -> 512 via two LIP stages
    Each stage learns per-pixel importance logits; output is a weighted avg_pool where weights = exp(amp * sigmoid(logit)).
    
    Zero-init on last conv so training starts from average pooling.
    ~290 trainable parameters.
    """

    def __init__(self, in_channels=1, amplification=12.0):
        super().__init__()
        self.amp = amplification

        self.logit1 = nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=1), nn.InstanceNorm2d(8, affine=True), nn.ReLU(inplace=True),nn.Conv2d(8, in_channels, 1),)
        nn.init.zeros_(self.logit1[-1].weight)
        nn.init.zeros_(self.logit1[-1].bias)

        self.logit2 = nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=1),nn.InstanceNorm2d(8, affine=True), nn.ReLU(inplace=True), nn.Conv2d(8, in_channels, 1),)
        nn.init.zeros_(self.logit2[-1].weight)
        nn.init.zeros_(self.logit2[-1].bias)

    def _lip2d(self, x, logit_module, kernel_size=3, stride=2, padding=1):
        logit = logit_module(x)
        logit = self.amp * torch.sigmoid(logit)
        weight = torch.exp(logit)
        
        return (F.avg_pool2d(x * weight, kernel_size, stride, padding) / F.avg_pool2d(weight, kernel_size, stride, padding))

    def forward(self, x_2048):
        x_1024 = self._lip2d(x_2048, self.logit1)   # (B, 1, 1024, 1024)
        x_512  = self._lip2d(x_1024, self.logit2)   # (B, 1, 512, 512)
        
        return x_512
