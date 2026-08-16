import torch.nn as nn
import torch.nn.functional as F

class CenterNetHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=1, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout2d(dropout),                       # ← regularization
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, fpn_features):
        return self.head(fpn_features["stage1"])


class CenterNetHeadHR(nn.Module):
    def __init__(self, in_ch=256, num_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, fpn_features):
        p2 = fpn_features["stage1"]
        p2_hr = F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(p2_hr)
