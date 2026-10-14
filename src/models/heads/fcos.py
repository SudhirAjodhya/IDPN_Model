import torch
import torch.nn as nn


class FCOSHead(nn.Module):
    """
    Single-level FCOS head: classification + ltrb regression + centerness.

    Differs from CenterNetDetHead in what it supervises, not in capacity. CenterNet
    regresses a box at the object centre only; FCOS regresses one at every cell inside
    the box, so the resizer beneath receives gradient proportional to nodule area rather
    than from a single cell per object. Measured on NODE21 that is 6x more supervised
    cells at target=224, rising to 32x at target=512.

    ltrb is exponentiated to stay positive. With the raw output near zero at init, exp()
    gives distances of ~1 cell, close to the true half-width of a NODE21 nodule at these
    resolutions (boxes span 2.5-6 cells), so regression starts near-calibrated instead of
    having to travel from an arbitrary value.
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

        self.classification_branch = branch(num_classes, use_dropout=True)
        self.regression_branch     = branch(4, use_dropout=False)
        self.centerness_branch     = branch(1, use_dropout=False)

        # Focal prior: predict background almost everywhere at init
        nn.init.constant_(self.classification_branch[-1].bias, -4.59)     # sigmoid ~= 0.01
        nn.init.zeros_(self.regression_branch[-1].bias)                   # exp(0) = 1 cell

    def forward(self, fpn_features):
        feature = fpn_features["stage1"]

        return {
            "classification": self.classification_branch(feature),
            # clamp before exp so a large early logit cannot produce an infinite box
            "ltrb":           torch.exp(self.regression_branch(feature).clamp(-6.0, 6.0)),
            "centerness":     self.centerness_branch(feature),
        }
