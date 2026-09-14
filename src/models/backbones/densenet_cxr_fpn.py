"""
DenseNet-121 + FPN with CHEST X-RAY pretraining, via TorchXRayVision.

WHY THIS EXISTS
  Every backbone in this project so far is ImageNet-pretrained, which leaves a standing
  objection to every result: natural-image filters being asked to read grayscale
  radiographs. This backbone removes that objection by starting from weights trained on
  real chest radiographs (CheXpert, NIH ChestX-ray14, PadChest, MIMIC-CXR, Openi, Kaggle,
  depending on the checkpoint).

  It also drops a step the ImageNet path forces. detector.py replicates the LDM's single
  channel to three and applies ImageNet mean/std; TorchXRayVision's conv0 has
  in_channels=1, so that replicate disappears - see `input_mode` below.

WHAT CHANGES, AND THE CONFOUND TO STATE
  This swaps architecture AND pretraining domain at once: DenseNet-121 (~7M trunk) against
  ConvNeXt-tiny (~28M), CXR against ImageNet. A difference from the ConvNeXt results is
  therefore not attributable to pretraining alone. `densenet121_imagenet_fpn` below is the
  control that separates them.

  Fewer parameters is arguably right here anyway: the ConvNeXt runs overfit from about
  epoch 12 on 3,591 training images.

WHY THE TRUNK IS SLICED BY HAND RATHER THAN TRACED
  torchvision's create_feature_extractor uses torch.fx symbolic tracing, and TorchXRayVision's
  DenseNet.forward calls utils.fix_resolution(), which does `len(x.shape)` - not traceable,
  so tracing raises.

  Bypassing their forward is what we want regardless: fix_resolution RESIZES THE INPUT to
  model.input_resolution (224). Routing through it would silently override --target-size and
  make every resolution in the sweep secretly 224, which is exactly the class of silent
  confound this project keeps finding. Slicing the trunk explicitly uses the pretrained
  weights and nothing else.

STAGE TAPS  (verified against the real module, not assumed)
  conv0 -> norm0 -> relu0 -> pool0   stride  4
  denseblock1                        stride  4,  256 ch  -> stage1
  transition1                        stride  8
  denseblock2                        stride  8,  512 ch  -> stage2
  transition2                        stride 16
  denseblock3                        stride 16, 1024 ch  -> stage3
  transition3                        stride 32
  denseblock4                        stride 32, 1024 ch  -> stage4

  norm5 and the classifier are dropped: they belong to the pathology-classification head,
  not to feature extraction.

  The trunk is exposed as `.features` so train.py's backbone-vs-head parameter split - and
  therefore --backbone-lr-factor - keeps working unchanged.

INSTALL
  python -m pip install torchxrayvision
  (use `python -m pip`, not `pip`: the venv was moved and the pip.exe launcher has a stale
  absolute path baked in)

  Weights download on first use to ~/.torchxrayvision/models_data/. If a download is
  interrupted the partial file is NOT resumed - torchxrayvision will try to load the
  truncated file and fail with "unexpected EOF". Delete it and re-run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork

from src.core.registry import BACKBONE_REGISTRY

# Channel widths of the four dense blocks in DenseNet-121. Asserted against the real model
# at construction rather than trusted, so a layout change fails loudly here instead of
# silently mis-wiring the FPN.
DENSENET121_STAGE_CHANNELS = [256, 512, 1024, 1024]


class _DenseNetTrunk(nn.Module):
    """
    The four dense blocks of a torchvision-layout DenseNet-121, returning stage taps.

    Holds the pretrained submodules directly, so `.parameters()` is exactly the backbone
    parameter set train.py needs for its learning-rate split.
    """

    def __init__(self, features):
        super().__init__()

        self.stem = nn.Sequential(features.conv0, features.norm0,
                                  features.relu0, features.pool0)
        self.block1, self.transition1 = features.denseblock1, features.transition1
        self.block2, self.transition2 = features.denseblock2, features.transition2
        self.block3, self.transition3 = features.denseblock3, features.transition3
        self.block4 = features.denseblock4

    def forward(self, x):
        x = self.stem(x)
        stage1 = self.block1(x)
        stage2 = self.block2(self.transition1(stage1))
        stage3 = self.block3(self.transition2(stage2))
        stage4 = self.block4(self.transition3(stage3))

        return {"stage1": stage1, "stage2": stage2, "stage3": stage3, "stage4": stage4}


class _DenseNetFPN(nn.Module):
    """A DenseNet-121 trunk with four stage taps and an FPN on top."""

    def __init__(self, features, fpn_out=256, expected_in_channels=1):
        super().__init__()

        assert features.conv0.in_channels == expected_in_channels, (
            "conv0 takes {} channels, expected {} - the wrong checkpoint family is loaded"
            .format(features.conv0.in_channels, expected_in_channels))

        self.features = _DenseNetTrunk(features)

        observed = self._observed_channels(expected_in_channels)
        assert observed == DENSENET121_STAGE_CHANNELS, (
            "unexpected DenseNet stage widths {} (expected {}). The model layout has "
            "changed; re-derive the taps before using this backbone."
            .format(observed, DENSENET121_STAGE_CHANNELS))

        self.fpn = FeaturePyramidNetwork(in_channels_list=DENSENET121_STAGE_CHANNELS,
                                         out_channels=fpn_out)

    def _observed_channels(self, channels):
        """Tap widths, from one small forward pass rather than from an assumption."""
        was_training = self.training
        self.eval()

        with torch.no_grad():
            taps = self.features(torch.zeros(1, channels, 224, 224))

        if was_training: self.train()

        return [taps["stage{}".format(i)].shape[1] for i in (1, 2, 3, 4)]

    def forward(self, x):
        return self.fpn(self.features(x))


def _load_xrv_features(weights):
    """The pretrained trunk from a TorchXRayVision DenseNet, without its forward."""
    try:
        import torchxrayvision as xrv

    except ImportError as exc:
        raise ImportError(
            "torchxrayvision is required for the CXR-pretrained backbone. Install with:  "
            "python -m pip install torchxrayvision"
        ) from exc

    return xrv.models.DenseNet(weights=weights).features


@BACKBONE_REGISTRY.register("densenet121_cxr_fpn")
class DenseNetCXRFPN(_DenseNetFPN):
    """Trained on the union of the TorchXRayVision corpora. The general-purpose choice."""

    input_mode = "xrv_gray"          # read by NoduleDetector.forward

    def __init__(self, fpn_out=256):
        super().__init__(_load_xrv_features("densenet121-res224-all"),
                         fpn_out=fpn_out, expected_in_channels=1)


@BACKBONE_REGISTRY.register("densenet121_cxr_nih_fpn")
class DenseNetCXRNIHFPN(_DenseNetFPN):
    """
    NIH ChestX-ray14 only. A narrower pretraining corpus, kept as a sensitivity check: if
    this differs materially from the `-all` weights then the result depends on which CXR
    corpus was used, and that belongs in the write-up.
    """

    input_mode = "xrv_gray"

    def __init__(self, fpn_out=256):
        super().__init__(_load_xrv_features("densenet121-res224-nih"),
                         fpn_out=fpn_out, expected_in_channels=1)


@BACKBONE_REGISTRY.register("densenet121_imagenet_fpn")
class DenseNetImageNetFPN(_DenseNetFPN):
    """
    DenseNet-121 with IMAGENET weights and three-channel input.

    The control that separates the two variables the CXR backbone changes together.
    Against this, densenet121_cxr_fpn differs ONLY in pretraining domain - same
    architecture, same parameter count - so a difference between them is attributable to
    the domain. Without this arm, DenseNet-vs-ConvNeXt and CXR-vs-ImageNet stay confounded.
    """

    input_mode = "imagenet_rgb"

    def __init__(self, fpn_out=256):
        from torchvision.models import densenet121, DenseNet121_Weights

        super().__init__(densenet121(weights=DenseNet121_Weights.DEFAULT).features,
                         fpn_out=fpn_out, expected_in_channels=3)

# No `if __name__ == "__main__"` block here on purpose. This module is imported by
# src/models/backbones/__init__.py, so running it with `python -m` would execute it a
# SECOND time and the registry decorators would raise "already registered". The shape
# check lives in scripts/diagnostics/check_cxr_backbone.py instead.
