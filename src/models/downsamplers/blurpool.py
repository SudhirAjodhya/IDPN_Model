import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import DOWNSAMPLER_REGISTRY

@DOWNSAMPLER_REGISTRY.register("blurpool")
class BlurPoolResizer(nn.Module):
    """
    Anti-aliased downsampling (Zhang 2019, ICML)
    Fixed Gaussian blur before stride-2 subsample
    two stages 2048 -> 1024 -> 512

    NO learnable parameters - tests whether aliasing (not content-awareness) is the bottleneck vs bilinear interpolation
    """

    def __init__(self, channels=1, kernel_size=5):
        super().__init__()

        sigma = (kernel_size - 1) / 4.0
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        
        g = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        
        self.register_buffer("blur_kernel", kernel_2d)
        self.pad = kernel_size // 2
        self.channels = channels

    def _blur_and_subsample(self, x):
        return F.conv2d(x, self.blur_kernel, stride=2, padding=self.pad, groups=self.channels)

    def forward(self, x_2048):
        x_1024 = self._blur_and_subsample(x_2048)   # (B, 1, 1024, 1024)
        x_512 = self._blur_and_subsample(x_1024)   # (B, 1, 512, 512)

        return x_512
