import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import DOWNSAMPLER_REGISTRY

class _TalebiResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class TalebiPaperResizer(nn.Module):
    """
    Implementation of Talebi Resizer from 2021 paper for ImageNet classification to determine that this
    is a paper faithful implementation of the Resizer, before adapting for CXRs
    """

    def __init__(self, in_channels=1, n_filters=16, n_blocks=2, target_size=224):
        super().__init__()

        assert target_size in (224, 256, 384, 448, 512), \
            f"target_size {target_size} not tested"

        self.target_size = target_size


        # Entry block at original resoltuion
        # Figure 3 paper: conv2d (k7n16s1) + LeakyReLu -> conv2d (k1n16s1) + LeakyReLu + BatchNorm
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(n_filters),
        )

        # r * ResBlock at target resolution
        self.res_blocks = nn.Sequential(*[_TalebiResBlock(n_filters) for _ in range(n_blocks)])

        # proceding resBlocks, before inner skip
        self.post = nn.Sequential(nn.Conv2d(n_filters, n_filters, 3, padding=1), nn.BatchNorm2d(n_filters))

        # Final Projection
        self.output_conv = nn.Conv2d(n_filters, in_channels, 7, padding=3)


    def forward(self, x):

        # x: (B, C, H, W) - Tensor as original resolution of the image
        x_bilinear = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

        # Entry convolutions at original resolution
        f = self.input_conv(x)
        f = F.interpolate(f, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)

        f_skip = f

        # ResBlocks and post conv
        f = self.res_blocks(f)
        f = self.post(f)

        # inner skip
        f = f + f_skip

        # Final projection of resdiual
        residual = self.output_conv(f)

        return x_bilinear + residual




@DOWNSAMPLER_REGISTRY.register("talebi")
class TalebiResizer(nn.Module):
    """
    Learned image resizer from Talebi & Milanfar 2021 (ICCV).
    Composes: bilinear resize (geometric skip) + learned CNN residual (enhancement).
    Processes at 1024 intermediate resolution to fit GPU memory budget.
    Zero-init on output conv ensures training starts as pure bilinear.
    """

    def __init__(self, in_channels=1, n_filters=16, n_blocks=2, target_size=512):
        super().__init__()
        self.target_size = target_size
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 7, padding=3),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            _TalebiResBlock(n_filters) for _ in range(n_blocks)
        ])
        self.output_conv = nn.Conv2d(n_filters, in_channels, 7, padding=3)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x_2048):
        x_bilinear = F.interpolate(x_2048, size=(self.target_size, self.target_size),
                                    mode="bilinear", align_corners=False)
        x_1024 = F.interpolate(x_2048, size=(1024, 1024),
                                mode="bilinear", align_corners=False)
        feat = self.input_conv(x_1024)
        feat = self.res_blocks(feat)
        feat = F.interpolate(feat, size=(self.target_size, self.target_size),
                             mode="bilinear", align_corners=False)
        residual = self.output_conv(feat)
        return x_bilinear + residual


@DOWNSAMPLER_REGISTRY.register("talebi_saliency")
class SaliencyTalebiResizer(nn.Module):
    """
    Talebi resizer with saliency gating on the residual.
    output = bilinear(x) + sigmoid(saliency(bilinear(x))) * residual_cnn(x)
    Saliency is computed on the cheap low-res view; zero-init on saliency final
    conv gives sigmoid(0)=0.5 uniform gating at init. output_conv uses small
    kaiming init (v7 fix: zero-init killed upstream gradients in v6).
    """

    def __init__(self, in_channels=1, n_filters=16, n_blocks=2,
                 saliency_channels=8, target_size=512):
        super().__init__()
        self.target_size = target_size

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 7, padding=3),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            _TalebiResBlock(n_filters) for _ in range(n_blocks)
        ])
        self.output_conv = nn.Conv2d(n_filters, in_channels, 7, padding=3)
        nn.init.kaiming_normal_(self.output_conv.weight, a=0.2)
        self.output_conv.weight.data *= 0.01
        nn.init.zeros_(self.output_conv.bias)

        self.saliency = nn.Sequential(
            nn.Conv2d(in_channels, saliency_channels, 5, padding=2),
            nn.BatchNorm2d(saliency_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(saliency_channels, saliency_channels, 3, padding=1),
            nn.BatchNorm2d(saliency_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(saliency_channels, 1, 1),
        )
        nn.init.zeros_(self.saliency[-1].weight)
        nn.init.zeros_(self.saliency[-1].bias)

    def forward(self, x_2048):
        x_bilinear = F.interpolate(x_2048, size=(self.target_size, self.target_size),
                                    mode="bilinear", align_corners=False)

        saliency_logit = self.saliency(x_bilinear)       # (B, 1, target, target)
        saliency_gate  = torch.sigmoid(saliency_logit)   # [0, 1] gating

        x_1024 = F.interpolate(x_2048, size=(1024, 1024),
                                mode="bilinear", align_corners=False)
        feat = self.input_conv(x_1024)
        feat = self.res_blocks(feat)
        feat = F.interpolate(feat, size=(self.target_size, self.target_size),
                             mode="bilinear", align_corners=False)
        residual = self.output_conv(feat)

        return x_bilinear + saliency_gate * residual


@DOWNSAMPLER_REGISTRY.register("talebi_inverse_bone")
class InverseBoneTalebiResizer(nn.Module):
    """
    Talebi resizer with a fixed inverse-bone-energy gate on the residual.

    Sobel edge magnitude ≈ bone/mediastinal edges. Inverting it gives a
    soft-tissue prior that concentrates the learned enhancement in the lungs.

    output = bilinear(x) + soft_tissue_gate(x) * learned_residual(x)

    Zero new learnable parameters vs vanilla Talebi; gate is hand-crafted.
    output_conv uses small kaiming init so all layers get gradient from step 1.
    """

    def __init__(self, in_channels=1, n_filters=16, n_blocks=2, target_size=512):
        super().__init__()
        self.target_size = target_size

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 7, padding=3),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            _TalebiResBlock(n_filters) for _ in range(n_blocks)
        ])
        self.output_conv = nn.Conv2d(n_filters, in_channels, 7, padding=3)
        nn.init.kaiming_normal_(self.output_conv.weight, a=0.2)
        self.output_conv.weight.data *= 0.01
        nn.init.zeros_(self.output_conv.bias)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _compute_soft_tissue_gate(self, x_512):
        gx = F.conv2d(x_512, self.sobel_x, padding=1)
        gy = F.conv2d(x_512, self.sobel_y, padding=1)
        edge_energy = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        edge_min = edge_energy.amin(dim=(2, 3), keepdim=True)
        edge_max = edge_energy.amax(dim=(2, 3), keepdim=True)
        edge_norm = (edge_energy - edge_min) / (edge_max - edge_min + 1e-8)

        # Floor of 0.1 keeps 10% residual signal even over bone/ribs
        return 1.0 - 0.9 * edge_norm

    def forward(self, x_2048):
        x_bilinear = F.interpolate(x_2048, size=(self.target_size, self.target_size),
                                    mode="bilinear", align_corners=False)

        x_1024 = F.interpolate(x_2048, size=(1024, 1024),
                                mode="bilinear", align_corners=False)
        feat = self.input_conv(x_1024)
        feat = self.res_blocks(feat)
        feat = F.interpolate(feat, size=(self.target_size, self.target_size),
                             mode="bilinear", align_corners=False)
        residual = self.output_conv(feat)

        gate = self._compute_soft_tissue_gate(x_bilinear)
        return x_bilinear + gate * residual
