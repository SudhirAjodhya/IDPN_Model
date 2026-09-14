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


class CenterNetDetHead(nn.Module):
    """
    Complete CenterNet head: centre heatmap + box size + sub-pixel offset.

    The pre-existing CenterNetHead emits only the heatmap, which is one of the three
    branches in Zhou, Wang & Krahenbuhl (2019). Without size and offset the model cannot
    produce a box, so IoU and mAP are not computable against a dataset annotated with
    bounding boxes. This head restores the reference formulation.

    Each branch is a separate 3x3 -> 1x1 stack on the shared FPN feature, matching the
    reference implementation rather than splitting one trunk, so the regression branches
    cannot corrupt the classification features through a shared final layer.
    """

    def __init__(self, in_ch=256, num_classes=1, dropout=0.3, head_ch=64):
        super().__init__()

        def branch(out_ch, use_dropout):
            layers = [
                nn.Conv2d(in_ch, head_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, head_ch),
                nn.GELU(),
            ]

            if use_dropout: layers.append(nn.Dropout2d(dropout))

            layers.append(nn.Conv2d(head_ch, out_ch, kernel_size=1))

            return nn.Sequential(*layers)

        # Dropout only on the classification branch, as in the heatmap-only head.
        # Regression branches are supervised at a handful of pixels per image; dropping
        # activations there would add variance to an already sparse signal.
        self.heatmap_branch = branch(num_classes, use_dropout=True)
        self.size_branch    = branch(2, use_dropout=False)
        self.offset_branch  = branch(2, use_dropout=False)

        # Bias the heatmap toward "background" at init: sigmoid(-2.19) ~= 0.1, the focal
        # loss prior from RetinaNet/CenterNet that stops early training being swamped by
        # the negative term.
        nn.init.constant_(self.heatmap_branch[-1].bias, -2.19)

    def forward(self, fpn_features):
        feature = fpn_features["stage1"]

        return {
            "heatmap": self.heatmap_branch(feature),
            "size_wh": self.size_branch(feature),
            "offset":  self.offset_branch(feature),
        }
