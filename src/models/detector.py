import torch
import torch.nn as nn
from src.core.registry import DOWNSAMPLER_REGISTRY, BACKBONE_REGISTRY
from src.models.heads.centernet import CenterNetHead, CenterNetHeadHR

# Import to trigger registration
import src.models.downsamplers
import src.models.backbones

class NoduleDetector(nn.Module):
    HEADS = {
        "centernet": CenterNetHead,
        "centernet_hr": CenterNetHeadHR,
    }

    def __init__(self, variant="bilinear", head="centernet", backbone="convnext_fpn",
                 heatmap_size=128, target_size=None):
        super().__init__()
        self.variant = variant

        # Load from registries. Pass target_size iff the variant accepts it - keeps
        # legacy variants (that hardcode their own default) working unchanged.
        resizer_key = "talebi" if variant == "talebi_aux" else variant
        resizer_cls = DOWNSAMPLER_REGISTRY.get(resizer_key)
        try:
            self.ldm = resizer_cls(target_size=target_size) if target_size is not None else resizer_cls()
        except TypeError:
            # variant doesn't take target_size - fall back to its default
            self.ldm = resizer_cls()

        self.backbone_fpn = BACKBONE_REGISTRY.get(backbone)(fpn_out=256)

        if head not in self.HEADS:
            raise KeyError(f"Head {head} not found. Available: {list(self.HEADS.keys())}")
        self.head = self.HEADS[head](in_ch=256, num_classes=1)

        # heatmap_size is what the head *emits* - derived from target_size (stage1 is /4)
        # when target_size is given, else the legacy hardcoded 128.
        self.heatmap_size = target_size // 4 if target_size is not None else heatmap_size

    def forward(self, x_2048):
        x_512 = self.ldm(x_2048)
        x_3ch = x_512.expand(-1, 3, -1, -1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], device=x_512.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x_512.device).view(1, 3, 1, 1)
        x_3ch = (x_3ch - mean) / std
        hm = self.head(self.backbone_fpn(x_3ch))
        
        return torch.nan_to_num(hm.clamp(-16, 16), nan=0.0)

    def get_ldm_output(self, x_2048):
        with torch.no_grad():
            return self.ldm(x_2048)
