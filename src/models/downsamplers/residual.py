import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import DOWNSAMPLER_REGISTRY

@DOWNSAMPLER_REGISTRY.register("residual")
class SimpleResidualResizer(nn.Module):
    """
    Variant alpha: simple content-aware residual added on top of bilinear.
    Interpretation B (within-grid re-weighting).
    """
    
    def __init__(self, in_channels=1, target_size=512):
        super().__init__()
        self.target_size = target_size

        def dw_sep(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                          padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )

        self.stem = nn.Sequential(
            dw_sep(in_channels, 16, stride=2),  # 2048 -> 1024
            dw_sep(16, 16, stride=2),           # 1024 -> 512
            dw_sep(16, 16, stride=1),           # keep at 512
        )
        self.head = nn.Conv2d(16, in_channels, kernel_size=1)

        # Zero-init: start at bilinear
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_2048):
        base = F.interpolate(x_2048, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        residual = self.head(self.stem(x_2048))
        self._cache_residual(residual)

        return base + residual

    def get_last_residual(self):
        return getattr(self, "_last_residual", None)

    def _cache_residual(self, r):
        self._last_residual = r.detach()
