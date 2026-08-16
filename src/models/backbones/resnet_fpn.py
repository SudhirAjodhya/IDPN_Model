"""ResNet50 + FPN backbones for the backbone-family sensitivity check.

Two variants: pretrained (ImageNet) and scratch. Both mirror the ConvNeXt
wiring: a `.features` extractor that returns stage1..stage4 taps, then an FPN
with fpn_out channels. Keeping `.features` at the same name lets train.py:140
param-group split (backbone vs head/LDM) work unchanged.

ResNet stage taps (torchvision resnet50):
  layer1 -> stride  4, 256  ch  -> stage1
  layer2 -> stride  8, 512  ch  -> stage2
  layer3 -> stride 16, 1024 ch  -> stage3
  layer4 -> stride 32, 2048 ch  -> stage4
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor

from src.core.registry import BACKBONE_REGISTRY


class _ResNet50FPN(nn.Module):
    def __init__(self, fpn_out=256, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.features = create_feature_extractor(backbone, return_nodes={
            "layer1": "stage1",
            "layer2": "stage2",
            "layer3": "stage3",
            "layer4": "stage4",
        })
        self.fpn = FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=fpn_out)

    def forward(self, x):
        return self.fpn(self.features(x))


@BACKBONE_REGISTRY.register("resnet50_fpn")
class ResNet50FPN(_ResNet50FPN):
    def __init__(self, fpn_out=256):
        super().__init__(fpn_out=fpn_out, pretrained=True)


@BACKBONE_REGISTRY.register("resnet50_fpn_scratch")
class ResNet50FPNScratch(_ResNet50FPN):
    def __init__(self, fpn_out=256):
        super().__init__(fpn_out=fpn_out, pretrained=False)
