"""
Learned downsamplers built from the 2026-08-21 attribution measurement.

WHAT THE MEASUREMENT SAID
    dL/d(residual) is 130-230x concentrated on nodules versus matched background, and
    statistically identical whether or not a learnable resizer is present. So the loss
    sends a strongly nodule-localised instruction. The trained Talebi residual answers
    with enrichment 0.62-0.85 - it deposits LESS energy on nodules than on matched
    control patches. The instruction is fine; the response is not.

    The reason is receptive field. The Talebi resizer is two 3x3 residual blocks at 16
    channels running after the interpolate, giving RF ~11 target cells against nodules of
    9.8 cells at target=224, 16.9 at 384 and 22.5 at 512 - a view 0.5x to 1.1x the size of
    the object it is supposed to distinguish from a vessel or a rib crossing. It cannot be
    selective, so it converges on a spatially uniform transform, and uniform transforms
    deposit energy on high-frequency structure (bone, edges) rather than low-contrast
    nodules. A misaimed residual is worse than none.

TWO RESPONSES, DELIBERATELY DIFFERENT IN KIND

    DilatedGatedResizer  - keeps Talebi's "uniform resample + learned residual" form and
        fixes the two things the measurement blames: receptive field (dilated blocks) and
        selectivity (a multiplicative spatial gate). Incremental, and each part can be
        switched off, so dilation and gating are separable ablations.

    WarpResizer - changes the form. Instead of adding a residual at fixed sampling
        locations, it learns WHERE to sample: a per-output-pixel offset field feeding
        grid_sample, so pixel budget can be spent non-uniformly. This is the shape the
        literature converged on for this problem - content-adaptive resamplers with
        per-pixel kernels and spatial offsets (Sun & Chen 2019), and geometry-guided
        resampling reporting large small-object gains over naive downsampling (Ghosh et
        al. 2023). Talebi & Milanfar's resizer is NOT of this family: it resamples
        uniformly and edits afterwards, which is precisely the freedom the attribution
        measurement shows it lacks.

Both keep the property that made zero-init worth having: at initialisation the output is
(numerically) the plain bilinear resize, so training starts from the baseline rather than
from noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import DOWNSAMPLER_REGISTRY


class _DilatedResBlock(nn.Module):
    """
    Talebi's residual block with a dilation factor.

    A 3x3 conv at dilation d has receptive field 2d+1, so a block contributes 4d cells
    rather than 4. Stacking dilations 1, 2, 4, 8 gives RF ~61 target cells against the
    ~11 of the undilated stack, without adding a single parameter.
    """

    def __init__(self, channels, dilation=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


@DOWNSAMPLER_REGISTRY.register("node21_ldm_dilated_gated")
class DilatedGatedResizer(nn.Module):
    """
    Talebi's structure with a wider view and a spatial gate on the residual.

    output = bilinear(x) + gate * output_conv(features)

    `gate` is a per-pixel sigmoid, which is the mechanism the attribution measurement says
    is missing: it lets the resizer apply the residual at some locations and not others.
    Note this is MULTIPLICATIVE. The earlier attention variant SUBTRACTED an attention
    branch and measured the worst enrichment of anything tested (0.644), so the subtractive
    formulation is refuted while the gating one is untested.

    Args:
        n_filters: channel width, 16 to match the paper
        n_blocks: number of residual blocks
        dilations: per-block dilation. None or all-1 reproduces Talebi's receptive field,
                   which makes "dilation off" a clean control.
        use_gate: False reduces this to a plain dilated Talebi
        target_size: output side length
    """

    def __init__(self, in_channels=1, n_filters=16, n_blocks=4, target_size=224,
                 dilations=(1, 2, 4, 8), use_gate=True):
        super().__init__()

        assert target_size in (224, 256, 384, 448, 512), f"target_size {target_size} not tested"

        if dilations is None: dilations = (1,) * n_blocks

        assert len(dilations) >= n_blocks, f"need {n_blocks} dilations, got {len(dilations)}"

        self.target_size = target_size
        self.use_gate = use_gate

        # Entry block at native resolution, unchanged from Talebi Figure 3
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(n_filters),
        )

        self.res_blocks = nn.Sequential(*[_DilatedResBlock(n_filters, dilations[i]) for i in range(n_blocks)])
        self.post = nn.Sequential(nn.Conv2d(n_filters, n_filters, 3, padding=1), nn.BatchNorm2d(n_filters))
        self.output_conv = nn.Conv2d(n_filters, in_channels, 7, padding=3)

        # Near-zero residual at init, so the model starts at bilinear (kaiming * 0.01,
        # not exact zero, which would starve the upstream layers of gradient for a step)
        nn.init.kaiming_normal_(self.output_conv.weight, a=0.2)
        self.output_conv.weight.data *= 0.01
        nn.init.zeros_(self.output_conv.bias)

        if use_gate:
            self.gate_conv = nn.Conv2d(n_filters, 1, 3, padding=1)
            # bias 0 -> sigmoid 0.5 at init: the gate starts neutral and learns which way
            # to move, rather than starting closed and having to be switched on
            nn.init.zeros_(self.gate_conv.weight)
            nn.init.zeros_(self.gate_conv.bias)

    def receptive_field_cells(self):
        """Receptive field of the post-interpolate stack, in target cells. For reporting."""
        rf = 1

        for block in self.res_blocks:
            dilation = block.block[0].dilation[0]
            rf += 2 * (2 * dilation)          # two dilated 3x3 convs per block

        return rf + 2                          # the post 3x3

    def forward(self, x):
        x_bilinear = F.interpolate(x, size=(self.target_size,) * 2, mode="bilinear", align_corners=False)

        features = self.input_conv(x)
        features = F.interpolate(features, size=(self.target_size,) * 2, mode="bilinear", align_corners=False)

        skip = features
        features = self.post(self.res_blocks(features))
        features = features + skip

        residual = self.output_conv(features)

        if self.use_gate:
            residual = residual * torch.sigmoid(self.gate_conv(features))

        return x_bilinear + residual


@DOWNSAMPLER_REGISTRY.register("node21_ldm_dilated")
class DilatedResizer(DilatedGatedResizer):
    """Dilation only - isolates receptive field from selectivity."""

    def __init__(self, in_channels=1, n_filters=16, n_blocks=4, target_size=224):
        super().__init__(in_channels, n_filters, n_blocks, target_size,
                         dilations=(1, 2, 4, 8), use_gate=False)


@DOWNSAMPLER_REGISTRY.register("node21_ldm_gated")
class GatedResizer(DilatedGatedResizer):
    """Gating only, at Talebi's original receptive field - isolates selectivity."""

    def __init__(self, in_channels=1, n_filters=16, n_blocks=2, target_size=224):
        super().__init__(in_channels, n_filters, n_blocks, target_size,
                         dilations=(1, 1), use_gate=True)


@DOWNSAMPLER_REGISTRY.register("node21_ldm_warp")
class WarpResizer(nn.Module):
    """
    Learns WHERE to sample rather than what to add afterwards.

    A small network predicts a per-output-pixel offset into the source image; the output
    is then taken by grid_sample at those shifted locations. Sampling is therefore
    non-uniform: the resizer can pull output pixels toward a nodule, spending more of a
    fixed pixel budget on diagnostically dense regions and less on empty lung field.

    This is a different response to the attribution result than DilatedGatedResizer.
    Talebi can only edit values at fixed sample points; warping changes the points. If
    the deficit really is "cannot be spatially selective", moving the sampling grid is a
    more direct remedy than gating an additively-applied residual.

    Offsets are predicted at a coarse control resolution and upsampled, which
    (a) makes the warp smooth, so anatomy is not torn, and (b) keeps the offset network
    cheap. `max_offset` bounds the displacement as a fraction of the output extent.

    At init the offset head is zero, so the sampling grid is exactly the identity and the
    output matches F.interpolate to numerical precision - the same "starts at bilinear"
    guarantee as the zero-init residual.
    """

    def __init__(self, in_channels=1, n_filters=16, target_size=224,
                 control_size=32, max_offset=0.05):
        super().__init__()

        assert target_size in (224, 256, 384, 448, 512), f"target_size {target_size} not tested"

        self.target_size = target_size
        self.control_size = control_size
        self.max_offset = max_offset

        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(n_filters, 2, 3, padding=1),
        )

        # Exact identity sampling at init
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def _identity_grid(self, batch_size, device, dtype):
        """
        Normalised sampling grid reproducing align_corners=False bilinear resize.

        Output pixel i samples input coordinate (i + 0.5) * S / T - 0.5, which in
        grid_sample's [-1, 1] convention is 2 * (i + 0.5) / T - 1.
        """
        coords = (torch.arange(self.target_size, device=device, dtype=dtype) + 0.5)
        coords = 2.0 * coords / self.target_size - 1.0

        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)              # (T, T, 2) as (x, y)

        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Predict offsets on a coarse control grid, then upsample for a smooth warp
        control = F.interpolate(x, size=(self.control_size,) * 2, mode="bilinear", align_corners=False)
        offsets = self.offset_net(control)
        offsets = F.interpolate(offsets, size=(self.target_size,) * 2, mode="bilinear", align_corners=False)

        # (B, 2, T, T) -> (B, T, T, 2), bounded so the warp cannot fold the image
        offsets = torch.tanh(offsets).permute(0, 2, 3, 1) * self.max_offset

        grid = self._identity_grid(batch_size, x.device, x.dtype) + offsets

        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)



@DOWNSAMPLER_REGISTRY.register("node21_ldm_warp_off15")
class WarpResizerOffset15(WarpResizer):
    """
    Warp with 3x the displacement bound. Tests whether the base variant's null is
    "deformation does not help here" or "the deformation was too tightly bounded to matter".

    Jin et al. (ICLR 2022) allow far more aggressive magnification than max_offset=0.05,
    which caps displacement at 2.5% of image width - 9.6 target cells at t384, against a
    median nodule of 19.1. The trained offset head already reaches 8.34 of that 9.6 bound,
    so the constraint plausibly binds.

    WARNING: scripts/diagnostics/warp_label_displacement.py measured label displacement at
    5-11% of nodule diameter for max_offset=0.05, which is why the linear target rule in
    fcos_targets.py stays valid there. Displacement scales with this bound, so re-run that
    diagnostic on any run of this variant BEFORE trusting its detection numbers.
    """

    def __init__(self, in_channels=1, n_filters=16, target_size=224,
                 control_size=32, max_offset=0.15):
        super().__init__(in_channels, n_filters, target_size, control_size, max_offset)


@DOWNSAMPLER_REGISTRY.register("node21_ldm_warp_off30")
class WarpResizerOffset30(WarpResizer):
    """
    Warp with 6x the displacement bound. The far end of the sweep, included so the
    constraint is unambiguously slack at one point in the range.

    At t384 this permits up to 57.6 target cells of displacement against a 19.1-cell median
    nodule, so the linear label rule is EXPECTED to break here. Treat detection numbers from
    this variant as uninterpretable until targets are built through the sampling grid; its
    value is in showing where the bound starts to matter, not in its sensitivity.
    """

    def __init__(self, in_channels=1, n_filters=16, target_size=224,
                 control_size=32, max_offset=0.30):
        super().__init__(in_channels, n_filters, target_size, control_size, max_offset)

def _demo():
    """Self-checks: receptive field, bilinear-at-init, and that the warp can actually warp."""
    torch.manual_seed(0)
    x = torch.randn(2, 1, 1024, 1024)

    print("receptive field of the post-interpolate stack, in target cells:")
    plain = DilatedGatedResizer(target_size=384, n_blocks=2, dilations=(1, 1), use_gate=False)
    dilated = DilatedGatedResizer(target_size=384, n_blocks=4, dilations=(1, 2, 4, 8))
    print(f"  Talebi-equivalent (2 blocks, no dilation): {plain.receptive_field_cells()} cells")
    print(f"  dilated (4 blocks, 1/2/4/8):               {dilated.receptive_field_cells()} cells")

    nodule_cells = 45.0 * 384 / 1024
    assert plain.receptive_field_cells() == 11, plain.receptive_field_cells()
    assert dilated.receptive_field_cells() > 3 * nodule_cells, "dilation should give >3x nodule context"
    print(f"  nodule at target 384 is {nodule_cells:.1f} cells -> "
          f"{plain.receptive_field_cells()/nodule_cells:.1f}x vs {dilated.receptive_field_cells()/nodule_cells:.1f}x context")

    # Every variant must start at (near) bilinear
    reference = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)

    for name, model in (("dilated+gated", dilated), ("dilated", DilatedResizer(target_size=384)),
                        ("gated", GatedResizer(target_size=384))):
        model.eval()
        with torch.no_grad(): deviation = (model(x) - reference).abs().max().item()
        assert deviation < 0.05, f"{name} deviates {deviation:.4f} from bilinear at init"
        print(f"OK  {name:14} starts within {deviation:.2e} of bilinear")

    warp = WarpResizer(target_size=384)
    warp.eval()
    with torch.no_grad(): deviation = (warp(x) - reference).abs().max().item()

    # On noise the identity grid agrees with F.interpolate to ~4e-4, which is float32
    # coordinate precision, not a sampling misalignment. The ramp below is the test that
    # tells those apart: a half-pixel shift would appear on a linear gradient as a
    # constant offset (a genuine one-pixel shift costs max ~4.2 on this data, four orders
    # of magnitude more), whereas precision error stays negligible on a smooth signal.
    assert deviation < 1e-3, f"warp identity grid deviates {deviation:.2e} from F.interpolate"

    ramp = torch.linspace(0, 1, 1024).view(1, 1, 1, -1).expand(1, 1, 1024, 1024).contiguous()
    ramp_reference = F.interpolate(ramp, size=(384, 384), mode="bilinear", align_corners=False)
    with torch.no_grad(): ramp_deviation = (warp(ramp) - ramp_reference).abs().max().item()
    assert ramp_deviation < 1e-5, f"warp is MISALIGNED, not just imprecise: {ramp_deviation:.2e} on a ramp"
    print(f"OK  {'warp':14} identity grid matches F.interpolate ({deviation:.1e} on noise, "
          f"{ramp_deviation:.1e} on a ramp -> precision, not a shift)")

    # ...and the warp must be capable of non-identity sampling once trained
    with torch.no_grad():
        nn.init.normal_(warp.offset_net[-1].bias, std=1.0)
        warped = warp(x)
    moved = (warped - reference).abs().mean().item()
    assert moved > 1e-3, "warp produced no displacement with a non-zero offset head"
    print(f"OK  {'warp':14} displaces sampling once offsets are non-zero (mean |delta| {moved:.3f})")

    # The offset-bound sweep variants differ ONLY in max_offset, and must keep the exact
    # identity-at-init guarantee, or their runs would not be comparable with the base warp.
    for name, expected in (("node21_ldm_warp", 0.05),
                           ("node21_ldm_warp_off15", 0.15),
                           ("node21_ldm_warp_off30", 0.30)):
        variant = DOWNSAMPLER_REGISTRY.get(name)(target_size=384)
        assert abs(variant.max_offset - expected) < 1e-9, (name, variant.max_offset)

        identity = variant._identity_grid(1, torch.device("cpu"), torch.float32)
        assert torch.allclose(variant(x)[0], reference[0], atol=1e-3), f"{name} not identity at init"
        # Bound in target cells, the unit the label-displacement diagnostic reports in
        bound_cells = expected * 384 / 2.0
        print(f"OK  {name:22} max_offset {expected:.2f} -> bound {bound_cells:5.1f} target cells, "
              f"identity at init")

    # The gate must be able to suppress the residual somewhere and not elsewhere
    with torch.no_grad():
        nn.init.normal_(dilated.gate_conv.weight, std=0.5)
        features = dilated.input_conv(x[:1])
        features = F.interpolate(features, size=(384, 384), mode="bilinear", align_corners=False)
        features = dilated.post(dilated.res_blocks(features)) + features
        gate = torch.sigmoid(dilated.gate_conv(features))
    assert gate.min() < 0.4 and gate.max() > 0.6, f"gate spans only {gate.min():.2f}-{gate.max():.2f}"
    print(f"OK  {'gate':14} spans {gate.min():.2f}-{gate.max():.2f}: it can be spatially selective")

    for name in ("node21_ldm_dilated_gated", "node21_ldm_dilated", "node21_ldm_gated", "node21_ldm_warp"):
        model = DOWNSAMPLER_REGISTRY._registry[name](target_size=224)
        assert model(torch.randn(1, 1, 512, 512)).shape == (1, 1, 224, 224)
        print(f"OK  registered {name:28} params {sum(p.numel() for p in model.parameters()):>7,}")


if __name__ == "__main__":
    _demo()
