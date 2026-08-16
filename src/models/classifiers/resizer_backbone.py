import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v2


BACKBONES = {
    "resnet50": resnet50,
    "mobilenet_v2": mobilenet_v2,
}


def build_backbone(name: str, num_classes: int) -> nn.Module:
    """Instantiate a torchvision backbone by name with random init."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}. options: {list(BACKBONES)}")
    return BACKBONES[name](weights=None, num_classes=num_classes)


class ResizerBackbone(nn.Module):
    """
    Optional learned resizer prepended to a torchvision classification backbone.
    Used for Talebi paper Table 4 reproduction on ImageNette.
    """

    def __init__(self, resizer: nn.Module | None, num_classes: int,
                 backbone: str = "resnet50"):
        super().__init__()
        self.resizer = resizer
        self.backbone_name = backbone
        self.backbone = build_backbone(backbone, num_classes)

    def forward(self, x):
        if self.resizer is not None:
            x = self.resizer(x)
        return self.backbone(x)
