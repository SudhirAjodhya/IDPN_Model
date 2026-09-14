import torch
import torch.nn as nn
from src.core.registry import DOWNSAMPLER_REGISTRY, BACKBONE_REGISTRY
from src.models.heads.centernet import CenterNetHead, CenterNetHeadHR, CenterNetDetHead
from src.models.heads.fcos import FCOSHead

# Import to trigger registration
import src.models.downsamplers
import src.models.backbones

class NoduleDetector(nn.Module):
    HEADS = {
        "centernet": CenterNetHead,
        "centernet_hr": CenterNetHeadHR,
        "centernet_det": CenterNetDetHead,
        "fcos": FCOSHead,
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

        # centernet_det returns a dict of three branches; the older heads return a bare
        # heatmap tensor. Callers branch on this rather than sniffing the output type.
        self.head_name = head
        self.predicts_boxes = head in ("centernet_det", "fcos")
        self.is_fcos = head == "fcos"

        # heatmap_size is what the head *emits* - derived from target_size (stage1 is /4)
        # when target_size is given, else the legacy hardcoded 128.
        self.heatmap_size = target_size // 4 if target_size is not None else heatmap_size

    def forward(self, x_2048):
        x_512 = self.ldm(x_2048)

        # How the backbone wants its input, DECLARED BY THE BACKBONE rather than assumed
        # here. A CXR-pretrained encoder takes one channel in radiograph units; replicating
        # to three and applying ImageNet statistics to it would be actively wrong.
        # getattr with a default keeps every existing backbone on the original path, so
        # runs made before this branch existed still reproduce bit-identically.
        input_mode = getattr(self.backbone_fpn, "input_mode", "imagenet_rgb")

        if input_mode == "imagenet_rgb":
            x_in = x_512.expand(-1, 3, -1, -1).contiguous()
            mean = torch.tensor([0.485, 0.456, 0.406], device=x_512.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x_512.device).view(1, 3, 1, 1)
            x_in = (x_in - mean) / std

        elif input_mode == "xrv_gray":
            # TorchXRayVision's convention: xrv.datasets.normalize maps [0, maxval] onto
            # [-1024, 1024]. The LDM emits [0, 1], so this is exactly that map.
            x_in = (2.0 * x_512 - 1.0) * 1024.0

        else:
            raise ValueError("unknown backbone input_mode {!r}".format(input_mode))

        out = self.head(self.backbone_fpn(x_in))

        if self.is_fcos:
            # ltrb is already exponentiated and positive inside the head; clamping it
            # here would cap large boxes. Only the two logit branches are clamped.
            return {
                "classification": torch.nan_to_num(out["classification"].clamp(-16, 16), nan=0.0),
                "ltrb":           torch.nan_to_num(out["ltrb"], nan=1.0),
                "centerness":     torch.nan_to_num(out["centerness"].clamp(-16, 16), nan=0.0),
            }

        if self.predicts_boxes:
            # Only the heatmap is a logit needing clamping; size and offset are
            # unbounded regressions and clamping them would cap large boxes.
            return {
                "heatmap": torch.nan_to_num(out["heatmap"].clamp(-16, 16), nan=0.0),
                "size_wh": torch.nan_to_num(out["size_wh"], nan=0.0),
                "offset":  torch.nan_to_num(out["offset"], nan=0.0),
            }

        return torch.nan_to_num(out.clamp(-16, 16), nan=0.0)

    def get_ldm_output(self, x_2048):
        with torch.no_grad():
            return self.ldm(x_2048)
