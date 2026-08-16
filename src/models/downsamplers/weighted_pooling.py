import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import DOWNSAMPLER_REGISTRY

@DOWNSAMPLER_REGISTRY.register("weighted_pooling")
class WeightedPoolingResizer(nn.Module):
    """
    Variant gamma: content-aware weighted pooling at the entry point.
    """
    def __init__(self, in_channels=1, saliency_channels=16, target_size=512):
        super().__init__()
        self.target_size = target_size

        def dw_sep(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )

        self.saliency_stem = nn.Sequential(
            dw_sep(in_channels, saliency_channels, stride=2),
            dw_sep(saliency_channels, saliency_channels, 2),
            dw_sep(saliency_channels, saliency_channels, 1),
        )
        self.saliency_head = nn.Conv2d(saliency_channels, 1, kernel_size=1)

        nn.init.normal_(self.saliency_head.weight, 0, 0.05)
        nn.init.zeros_(self.saliency_head.bias)

    def forward(self, x_2048):
        B, C, H, W = x_2048.shape
        # Assuming H=2048 and self.target_size=512 for block math
        pool_size = H // self.target_size
        
        saliency_512 = self.saliency_head(self.saliency_stem(x_2048))
        saliency_2048 = F.interpolate(saliency_512, size=(H, W), mode="bilinear", align_corners=False)

        x_blocks = x_2048.view(B, C, self.target_size, pool_size, self.target_size, pool_size)
        s_blocks = saliency_2048.view(B, 1, self.target_size, pool_size, self.target_size, pool_size)

        s_flat = s_blocks.reshape(B, 1, self.target_size, self.target_size, pool_size*pool_size)
        w_flat = F.softmax(s_flat, dim=-1)
        w_blocks = w_flat.reshape(B, 1, self.target_size, pool_size, self.target_size, pool_size)

        output = (x_blocks * w_blocks).sum(dim=(3, 5))

        self._last_weights = w_blocks.detach()
        self._last_saliency = saliency_512.detach()

        return output

    def get_last_weights(self):
        return getattr(self, "_last_weights", None)

    def get_last_saliency(self):
        return getattr(self, "_last_saliency", None)

class InNetworkWeightedPooling(nn.Module):
    """
    In-network variant of WeightedPooling for replacing downsample blocks.
    """
    def __init__(self, in_channels=96):
        super().__init__()
        self._forward_count = 0
        self.saliency = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(8, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.normal_(self.saliency[-1].weight, 0, 0.05)
        nn.init.zeros_(self.saliency[-1].bias)

    def forward(self, x):
        self._forward_count += 1
        B, C, H, W = x.shape
        H_out, W_out = H // 2, W // 2

        s = self.saliency(x)

        x_blocks = x.view(B, C, H_out, 2, W_out, 2)
        s_blocks = s.view(B, 1, H_out, 2, W_out, 2)

        s_flat = s_blocks.reshape(B, 1, H_out, W_out, 4)
        w_flat = F.softmax(s_flat, dim=-1)
        w_blocks = w_flat.reshape(B, 1, H_out, 2, W_out, 2)

        output = (x_blocks * w_blocks).sum(dim=(3, 5))

        self._last_weights = w_blocks.detach()
        return output

    def get_last_weights(self):
        return getattr(self, "_last_weights", None)

class InNetworkWeightedDownsample(nn.Module):
    def __init__(self, in_channels=96, out_channels=192):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels) # Equivalent to LayerNorm for NCHW
        self.weighted_pool = InNetworkWeightedPooling(in_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.weighted_pool(x)
        x = self.proj(x)
        return x
