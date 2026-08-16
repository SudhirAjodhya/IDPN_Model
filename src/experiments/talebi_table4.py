"""
Talebi & Milanfar (ICCV 2021) Table 4 configuration table.

Ten rows per backbone (per paper Table 4, ResNet-50 and MobileNet-v2 columns):
  - 5 Bilinear rows: no resizer, backbone consumes native full resolution
    at input_res in {224, 256, 320, 368, 448}, batch scales down as input grows.
  - 5 Proposed rows: TalebiPaperResizer(input_res -> 224), backbone always
    consumes 224, backbone init loaded from the bilinear_224 checkpoint of
    the SAME backbone + SAME seed (Table 3 protocol: baselines are used to
    initialize the classification CNNs; the resizer weights are random).

Batch sizes match paper Table 4 exactly. Backbones = ResNet-50 + MobileNet-v2
(the two we can afford in the 12h budget).

The config table is deterministic; ordering matters — bilinear_224 must run
first per (backbone, seed) so its checkpoint is available before any
proposed_* run for that (backbone, seed).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TableRow:
    label: str              # e.g. "bilinear_224", "proposed_368"
    resizer: Optional[str]  # None or "talebi_paper"
    input_res: int          # what the *pipeline* feeds the model (before resizer)
    target_res: int         # what the backbone consumes (bilinear rows: == input)
    batch_size: int         # paper Table 4 batch column
    init_from: Optional[str]  # label of prior config whose backbone init we load


# Paper Table 4 rows, output resolution 224 for all proposed configs.
# Batch sizes are copied from Table 4 verbatim.
BILINEAR_CONFIGS = [
    TableRow("bilinear_224", None, 224, 224, 128, None),
    TableRow("bilinear_256", None, 256, 256,  96, None),
    TableRow("bilinear_320", None, 320, 320,  64, None),
    TableRow("bilinear_368", None, 368, 368,  48, None),
    TableRow("bilinear_448", None, 448, 448,  32, None),
]

PROPOSED_CONFIGS = [
    TableRow("proposed_224", "talebi_paper", 224, 224, 128, "bilinear_224"),
    TableRow("proposed_256", "talebi_paper", 256, 224,  96, "bilinear_224"),
    TableRow("proposed_320", "talebi_paper", 320, 224,  64, "bilinear_224"),
    TableRow("proposed_368", "talebi_paper", 368, 224,  48, "bilinear_224"),
    TableRow("proposed_448", "talebi_paper", 448, 224,  32, "bilinear_224"),
]

ALL_CONFIGS = BILINEAR_CONFIGS + PROPOSED_CONFIGS

CONFIGS_BY_LABEL = {c.label: c for c in ALL_CONFIGS}

# Backbones we plan to sweep. Additional backbones may be added if you have
# torchvision support and compute; the table structure does not care.
DEFAULT_BACKBONES = ["resnet50", "mobilenet_v2"]


def get_config(label: str) -> TableRow:
    if label not in CONFIGS_BY_LABEL:
        raise KeyError(
            f"unknown config {label!r}. options: {sorted(CONFIGS_BY_LABEL)}"
        )
    return CONFIGS_BY_LABEL[label]


def iter_run_order(backbones=None, seeds=(0, 1)):
    """
    Iterate (backbone, config, seed) tuples in dependency-safe order:
    all bilinear rows first per (backbone, seed), then all proposed rows.
    Ensures every proposed_* run can find its `init_from` checkpoint.
    """
    backbones = backbones or DEFAULT_BACKBONES
    for backbone in backbones:
        for seed in seeds:
            for cfg in BILINEAR_CONFIGS:
                yield backbone, cfg, seed
            for cfg in PROPOSED_CONFIGS:
                yield backbone, cfg, seed
