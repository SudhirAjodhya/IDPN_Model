import torch
import torch.nn as nn
from src.core.registry import DOWNSAMPLER_REGISTRY

@DOWNSAMPLER_REGISTRY.register("learned_kernel")
class LearnedKernelResizer(nn.Module):
    """
    learned global downsampling kernel

    Two-stage stride-2 depthwise convolution, initialized with a bilinear (tent) filter. 98 trainable parameters total (two 7 x 7 depthwise filters)
    Inspired by Talebi et al. 2021 ("Learning to Resize Images for CV Tasks")
    """

    def __init__(self, in_channels=1, kernel_size=7):
        super().__init__()
        
        k = kernel_size
        pad = k // 2
        
        self.stage1 = nn.Conv2d(in_channels, in_channels, k, stride=2, padding=pad, groups=in_channels, bias=False)
        self.stage2 = nn.Conv2d(in_channels, in_channels, k, stride=2, padding=pad, groups=in_channels, bias=False)
        self._init_bilinear()

    def _init_bilinear(self):

        for conv in [self.stage1, self.stage2]:
            k = conv.kernel_size[0]
            
            c = k // 2
            w = torch.zeros(1, 1, k, k)
            
            for i in range(k):
                for j in range(k):
                    w[0, 0, i, j] = (max(0, 1 - abs(i - c) / (c + 1)) * max(0, 1 - abs(j - c) / (c + 1)))
            
            w /= w.sum()
            conv.weight.data.copy_(w)

    def forward(self, x_2048):
        x = self.stage1(x_2048)   # (B, 1, 1024, 1024)
        x = self.stage2(x)        # (B, 1, 512, 512)
        
        return x
