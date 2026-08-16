import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
from src.core.registry import BACKBONE_REGISTRY
from src.models.downsamplers.weighted_pooling import InNetworkWeightedDownsample

@BACKBONE_REGISTRY.register("convnext_fpn")
class ConvNeXtFPNBackbone(nn.Module):
    def __init__(self, fpn_out=256, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)

        # Stage 1 - high resolution (128 x 128, 96 chnnels)
        self.features = create_feature_extractor(backbone, return_nodes={
            "features.1": "stage1",
            "features.3": "stage2",
            "features.5": "stage3",
            "features.7": "stage4",
        })

        self.fpn = FeaturePyramidNetwork(in_channels_list=[96, 192, 384, 768], out_channels=fpn_out)

    def forward(self, x):
        return self.fpn(self.features(x))


@BACKBONE_REGISTRY.register("convnext_fpn_scratch")
class ConvNeXtFPNBackboneScratch(ConvNeXtFPNBackbone):
    # Same wiring, no ImageNet init. Isolates the pretrained-backbone confound.
    def __init__(self, fpn_out=256):
        super().__init__(fpn_out=fpn_out, pretrained=False)


@BACKBONE_REGISTRY.register("convnext_in_network_ldm")
class ConvNeXtWithInNetworkLDM(nn.Module):
    """
    Manual decomposition of ConvNeXt-T with the LDM inserted between stage 1 and
    stage 2
    Does NOT use create_feature_extractor: the FX-traced graph calls leaf
    modules by qualified name (features.2.0, features.2.1), so patching '2' on the
    GraphModule attaches a module the traced forward never executes.
    """
    def __init__(self, fpn_out=256):
        super().__init__()
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        feats = backbone.features
        assert len(feats) == 8, (f"Expected 8 children in convnext_tiny().features, got {len(feats)} - " f"torchvision layout changed; the stage indices below are invalid")

        self.stem = feats[0]
        self.stage1 = feats[1]

        # SKIP feats[2] - replaced by LDM
        self.ldm = InNetworkWeightedDownsample(in_channels=96, out_channels=192)
        self.stage2 = feats[3]
        self.ds2 = feats[4]
        self.stage3 = feats[5]
        self.ds3 = feats[6]
        self.stage4 = feats[7]

        self.fpn = FeaturePyramidNetwork(in_channels_list=[96, 192, 384, 768], out_channels=fpn_out,)

    def forward(self, x):
        x0 = self.stem(x)        # (B, 96, 128, 128)
        s1 = self.stage1(x0)     # (B, 96, 128, 128)

        x1 = self.ldm(s1)        # (B, 192, 64, 64) -LDM replaces feats[2]
        s2 = self.stage2(x1)     # (B, 192, 64, 64)

        x2 = self.ds2(s2)        # (B, 384, 32, 32)
        s3 = self.stage3(x2)     # (B, 384, 32, 32)

        x3 = self.ds3(s3)        # (B, 768, 16, 16)
        s4 = self.stage4(x3)     # (B, 768, 16, 16)

        feat_dict = {
            "stage1": s1,   # 128x128, 96ch
            "stage2": s2,   # 64x64, 192ch
            "stage3": s3,   # 32x32, 384ch
            "stage4": s4,   # 16x16, 768ch
        }
        return self.fpn(feat_dict)

    def get_last_weights(self):
        return self.ldm.weighted_pool.get_last_weights()


class LIPDownsample(nn.Module):
    """
    Drop-in replacement for ConvNeXt's stage-transition downsampling layer
    """

    def __init__(self, in_channels, out_channels=None, amplification=12.0, logit_hidden_ratio=4):
        super().__init__()

        out_channels = out_channels or (2 * in_channels)
        self.amp = amplification
        self.norm = nn.LayerNorm(in_channels)

        hidden_ch = max(in_channels // logit_hidden_ratio, 8)

        self.logit = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_channels, kernel_size=1),
        )
        nn.init.zeros_(self.logit[-1].weight)
        nn.init.zeros_(self.logit[-1].bias)

        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _lip2d(self, x, kernel_size=3, stride=2, padding=1):
        logit = self.logit(x)
        logit = self.amp * torch.sigmoid(logit)
        weight = torch.exp(logit)

        return (F.avg_pool2d(x * weight, kernel_size, stride, padding) / F.avg_pool2d(weight,kernel_size, stride, padding))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.channel_proj(self._lip2d(x))


@BACKBONE_REGISTRY.register("convnext_lip")
class LIPConvNeXtBackbone(nn.Module):
    """
    ConvNeXt-T with LIP replacing strided downsampling at selected stage transitions
    """
    def __init__(self, lip_at=frozenset(["s1_s2"]), fpn_out=256):
        super().__init__()
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        self.stem   = base.features[0]
        self.stage1 = base.features[1]
        self.stage2 = base.features[3]
        self.stage3 = base.features[5]
        self.stage4 = base.features[7]

        self.down1 = (LIPDownsample(96,  192) if "s1_s2" in lip_at else base.features[2])
        self.down2 = (LIPDownsample(192, 384) if "s2_s3" in lip_at else base.features[4])
        self.down3 = (LIPDownsample(384, 768) if "s3_s4" in lip_at else base.features[6])

        self._copy_pretrained_channel_proj(base, lip_at)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[96, 192, 384, 768], out_channels=fpn_out)

    def _copy_pretrained_channel_proj(self, base, lip_at):
        transitions = [
            ("s1_s2", self.down1, base.features[2]),
            ("s2_s3", self.down2, base.features[4]),
            ("s3_s4", self.down3, base.features[6]),
        ]

        for name, new_module, old_module in transitions:
            if name not in lip_at: continue

            old_norm = old_module[0]
            old_conv = old_module[1]

            with torch.no_grad():
                new_module.norm.weight.copy_(old_norm.weight)
                new_module.norm.bias.copy_(old_norm.bias)
                spatial_avg = old_conv.weight.mean(dim=(2, 3), keepdim=True)
                new_module.channel_proj.weight.copy_(spatial_avg)
                new_module.channel_proj.bias.copy_(old_conv.bias)

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.stage1(x)
        x  = self.down1(c1)
        c2 = self.stage2(x)
        x  = self.down2(c2)
        c3 = self.stage3(x)
        x  = self.down3(c3)
        c4 = self.stage4(x)
        return self.fpn({"stage1": c1, "stage2": c2, "stage3": c3, "stage4": c4})
