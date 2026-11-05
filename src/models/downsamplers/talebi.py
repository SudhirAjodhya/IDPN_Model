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


@DOWNSAMPLER_REGISTRY.register("node21_talebi")
class TalebiPaperResizer(nn.Module):
    """
    Paper-faithful Talebi resizer (Talebi & Milanfar, ICCV 2021, Figure 3)
    Registered as "node21_talebi" for the Node21-only training track - chosen to
    disambiguate from the pre-existing "talebi" key (zero-init variant used in
    matrix_v1 on JSRT+NODE21 mixed pipeline).
    """

    def __init__(self, in_channels=1, n_filters=16, n_blocks=2, target_size=224,
                 skip_mode="bilinear", skip_antialias=False):
        super().__init__()

        # An allowlist of (224, 256, 384, 448, 512) used to sit here. It was a guard against
        # untested sizes, but nothing in this module depends on the target beyond the two
        # F.interpolate calls, so it was blocking a legitimate experiment rather than
        # catching a bug - the resolution sweep needs 768 and 1024 to find where the
        # returns from output resolution stop.
        #
        # What IS load-bearing is divisibility by the head stride: FCOS predicts on a
        # target/4 grid, so a target that is not a multiple of 4 silently changes the
        # rounding between box coordinates and grid cells.
        assert target_size >= 64 and target_size % 4 == 0, \
            f"target_size {target_size} must be >= 64 and divisible by the stride-4 head grid"

        assert skip_mode in ("bilinear", "bicubic"), \
            "F.interpolate only implements antialias for bilinear and bicubic"

        self.target_size = target_size

        # The GEOMETRIC SKIP only. The output is skip(x) + residual(x), so the skip is the
        # classical downsampler this module is measured against - matching it to the
        # baseline is what makes the difference between the two arms equal to the learned
        # residual and nothing else. Comparing an anti-aliased baseline against a resizer
        # whose skip still aliases would just reverse the unfairness.
        #
        # The FEATURE resize on line ~67 deliberately stays plain bilinear whatever this is
        # set to: it lives inside residual(x), and Talebi and Milanfar Figure 3 specify
        # bilinear there. Changing it would alter the published architecture rather than the
        # comparison. Defaults reproduce the paper exactly, so every banked run is unaffected.
        self.skip_mode = skip_mode
        self.skip_antialias = skip_antialias


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
        x_bilinear = F.interpolate(x, size=(self.target_size, self.target_size),
                                   mode=self.skip_mode, align_corners=False,
                                   antialias=self.skip_antialias)

        # Entry convolutions at original resolution. Plain bilinear on purpose - this resize
        # is inside the learned residual and is what Figure 3 specifies; only the geometric
        # skip above tracks the baseline.
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




@DOWNSAMPLER_REGISTRY.register("node21_talebi_zeroinit")
class TalebiPaperResizerZeroInit(TalebiPaperResizer):
    """
    Paper-faithful Talebi with near-zero init on output_conv (kaiming * 0.01, bias = 0). At step 1 residual ~= 0 so model output ~= bilinear -- Talebi
    starts strictly >= bilinear and can only improve. Near-zero rather than exactly zero: a pure-zero output layer
    passes no gradient back to the layers before it.
    """
    def __init__(self, in_channels=1, n_filters=16, n_blocks=2, target_size=224,
                 skip_mode="bilinear", skip_antialias=False):
        super().__init__(in_channels=in_channels, n_filters=n_filters, n_blocks=n_blocks,
                         target_size=target_size, skip_mode=skip_mode,
                         skip_antialias=skip_antialias)

        nn.init.kaiming_normal_(self.output_conv.weight, a=0.2)
        self.output_conv.weight.data *= 0.01
        nn.init.zeros_(self.output_conv.bias)


@DOWNSAMPLER_REGISTRY.register("node21_talebi_zeroinit_wide")
class TalebiPaperResizerZeroInitWide(TalebiPaperResizerZeroInit):
    """
    Capacity ablation: same architecture and same near-zero output_conv init as
    node21_talebi_zeroinit, with n_filters raised 16 -> 64.

    DEVIATION FROM THE PAPER. Talebi & Milanfar Figure 3 specifies k7n16s1, so 16 is the
    faithful width; 64 is ours and must be reported as an ablation, not as "Talebi".

    Motivation: under val_sens selection the zeroinit resizer degrades monotonically as
    input resolution rises at a fixed target (FROC AUC 0.883 at 224->224 down to 0.767 at
    1024->224, five points, all same direction). Either 16 channels cannot carry the
    information through a large downsample factor, or the limit is architectural. Widening
    is the cheapest test that separates those. Params in the resizer go ~13.6k -> ~210k,
    still negligible against the ~30.7M backbone.
    """

    def __init__(self, in_channels=1, n_filters=64, n_blocks=2, target_size=224):
        super().__init__(in_channels=in_channels, n_filters=n_filters, n_blocks=n_blocks, target_size=target_size)


# ==============================================================================
# Zero-init Talebi + Swin-style attention-subtraction branch
# ==============================================================================
# Matches node21_talebi_zeroinit architecture exactly (paper-faithful processing
# at target resolution, near-zero output_conv init). Only difference: adds an
# attention branch that operates at attn_res (default 128), captures globally-
# consistent content via WMSA+SWMSA, and gets SUBTRACTED from the conv branch
# before output_conv. Motivation: paper-faithful residual under-attends nodules
# (2026-08-15 diagnostic: nodule_ratio 0.699, ratio-of-ratios 0.553). Global-
# subtraction should suppress background structure, leaving nodule-scale detail.
#
# At init: attention branch has default MHA/LN/MLP init (not zeroed); output_conv
# is near-zero-kaiming so total residual ~= 0 regardless of what f_conv - f_attn
# looks like -> model output ~= bilinear at step 1. Same "at least bilinear at
# init" guarantee as node21_talebi_zeroinit.
# ==============================================================================

class _WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):  # (B*num_windows, ws*ws, C)
        out, _ = self.attn(x, x, x, need_weights=False)

        return out


class _SwinLikeBlock(nn.Module):
    """
    One Swin block = WMSA + MLP + SWMSA + MLP. GELU in MLPs. ponytail: no cyclic-shift attention mask; edge windows will see wrap-mixed content, which on centered CXRs at 128x128 is a minor artifact. 
    Add mask if attention viz shows edge issues
    """

    def __init__(self, dim, window_size=8, num_heads=2, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        h = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim); self.wmsa  = _WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim); self.mlp1  = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
        self.norm3 = nn.LayerNorm(dim); self.swmsa = _WindowAttention(dim, num_heads)
        self.norm4 = nn.LayerNorm(dim); self.mlp2  = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def _window_attn(self, x, attn_module, shift):
        B, H, W, C = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, f"H, W must be divisible by window_size={ws} (got {H}x{W})"

        if shift:
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

        # partition to non-overlapping windows: (B, H/ws, ws, W/ws, ws, C) -> (B*nw, ws*ws, C)
        x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ws * ws, C)
        out = attn_module(x)                                     # attention within each window
        out = out.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, H, W, C)

        if shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))

        return out

    def forward(self, x):                                        # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)                                # (B, H, W, C) for LN/attn
        x = x + self._window_attn(self.norm1(x), self.wmsa,  shift=0)
        x = x + self.mlp1(self.norm2(x))
        x = x + self._window_attn(self.norm3(x), self.swmsa, shift=self.window_size // 2)
        x = x + self.mlp2(self.norm4(x))

        return x.permute(0, 3, 1, 2).contiguous()                # (B, C, H, W)


@DOWNSAMPLER_REGISTRY.register("node21_talebi_zeroinit_attn")
class TalebiPaperResizerZeroInitAttn(TalebiPaperResizer):
    """Paper-faithful zero-init Talebi + attention-subtraction branch. See block comment above."""
    def __init__(self, in_channels=1, n_filters=16, n_blocks=2, target_size=224,
                 attn_res=128, attn_window=8, attn_heads=2,
                 skip_mode="bilinear", skip_antialias=False):
        super().__init__(in_channels=in_channels, n_filters=n_filters,
                         n_blocks=n_blocks, target_size=target_size,
                         skip_mode=skip_mode, skip_antialias=skip_antialias)
        # zero-init the residual (same as node21_talebi_zeroinit)
        nn.init.kaiming_normal_(self.output_conv.weight, a=0.2)
        self.output_conv.weight.data *= 0.01
        nn.init.zeros_(self.output_conv.bias)

        # attention branch (new)
        assert attn_res % attn_window == 0, f"attn_res={attn_res} must be divisible by attn_window={attn_window}"
        self.attn_res  = attn_res
        self.attention = _SwinLikeBlock(dim=n_filters, window_size=attn_window, num_heads=attn_heads)

    def forward(self, x):
        # Paper-faithful path up through inner-skip, identical to base zeroinit. The skip
        # tracks the baseline (see TalebiPaperResizer.__init__); the feature resizes below
        # stay plain bilinear because they live inside the learned residual.
        x_bilinear = F.interpolate(x, size=(self.target_size,) * 2, mode=self.skip_mode,
                                   align_corners=False, antialias=self.skip_antialias)
        f = self.input_conv(x)                                                # at native res
        f = F.interpolate(f, size=(self.target_size,) * 2, mode="bilinear", align_corners=False)
        f_skip = f
        f_conv = self.res_blocks(f)
        f_conv = self.post(f_conv)
        f_conv = f_conv + f_skip                                              # inner skip at target

        # Attention branch: down to attn_res, Swin, up to target. Captures globally-consistent content.
        f_attn = F.interpolate(f, size=(self.attn_res,) * 2, mode="bilinear", align_corners=False)
        f_attn = self.attention(f_attn)
        f_attn = F.interpolate(f_attn, size=(self.target_size,) * 2, mode="bilinear", align_corners=False)

        # Subtract global from local, then zero-init residual projection
        residual = self.output_conv(f_conv - f_attn)
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
