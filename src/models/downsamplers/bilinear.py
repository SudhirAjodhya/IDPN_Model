import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import DOWNSAMPLER_REGISTRY

@DOWNSAMPLER_REGISTRY.register("bilinear")
class BilinearBaseline(nn.Module):
    """
    Standard bilinear downsampling baseline
    """
    def __init__(self, target_size=512):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)


class ClassicalBaseline(nn.Module):
    """
    Parameter-free classical downsampler with a selectable kernel and optional anti-aliasing.

    WHY ANTI-ALIASING NEEDS ITS OWN BASELINE. BilinearBaseline above calls F.interpolate
    with no low-pass filter, so decimating 1024 -> 224 samples the source grid roughly every
    4.6 pixels and folds everything above Nyquist back in as aliasing. That is precisely the
    failure the learned resizer is hypothesised to repair, which makes plain bilinear a weak
    comparator: beating it may show only that anti-aliasing helps, not that LEARNING helps.
    antialias=True applies the same adaptive low-pass PIL uses and stays parameter-free, so
    it separates "learned" from "properly filtered".

    Deliberately NOT folded into BilinearBaseline as a flag - that class is the comparator in
    every run banked to date and its behaviour must not move.

    Args:
        target_size: square output resolution
        mode: "bilinear" or "bicubic" - the interpolation kernel
        antialias: low-pass before sampling; F.interpolate ignores it when upscaling
    """

    def __init__(self, target_size=512, mode="bilinear", antialias=True):
        super().__init__()

        assert mode in ("bilinear", "bicubic"), \
            "F.interpolate only implements antialias for bilinear and bicubic"

        self.target_size = target_size
        self.mode = mode
        self.antialias = antialias

    def forward(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size), mode=self.mode,
                             align_corners=False, antialias=self.antialias)


@DOWNSAMPLER_REGISTRY.register("bilinear_antialias")
class BilinearAntialiasBaseline(ClassicalBaseline):
    """Bilinear WITH the low-pass filter plain `bilinear` omits. The fair classical comparator."""

    def __init__(self, target_size=512):
        super().__init__(target_size=target_size, mode="bilinear", antialias=True)


@DOWNSAMPLER_REGISTRY.register("bicubic")
class BicubicBaseline(ClassicalBaseline):
    """Bicubic without anti-aliasing - the naive drop-in reached for instead of bilinear."""

    def __init__(self, target_size=512):
        super().__init__(target_size=target_size, mode="bicubic", antialias=False)


@DOWNSAMPLER_REGISTRY.register("bicubic_antialias")
class BicubicAntialiasBaseline(ClassicalBaseline):
    """Bicubic WITH anti-aliasing - the strongest parameter-free downsampler in this set."""

    def __init__(self, target_size=512):
        super().__init__(target_size=target_size, mode="bicubic", antialias=True)
