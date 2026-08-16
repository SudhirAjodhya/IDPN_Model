"""Size-stratified sensitivity for Node21.

Post-hoc analysis: takes a trained detector + the Node21 test loader + the bin
lookup produced by scripts/stratify_nodules.py, emits per-bin sensitivity plus
macro (unweighted mean across bins) and micro (pooled across all bins) summaries.

Reuses the peak-extraction and centre-in-radius hit logic from
`detection_metrics.compute_detection_metrics_v4` so numbers are commensurable
with the pipeline's headline sensitivity."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable

import torch
from torch.cuda.amp import autocast

from src.evaluation.detection_metrics import _extract_peaks, _nms_radius_for
from src.data.nodule_sizes import BINS, bin_assignment


def _load_bin_lookup(stratification_csv: str) -> Dict[str, str]:
    """Read stratify_nodules.py output → {img_name: bin}.
    Assumes one nodule per image on the Node21 test split — for multi-nodule
    images we bin by the first row's diameter (rare on Node21; if it matters,
    key on (img_name, x, y) instead)."""
    import pandas as pd
    df = pd.read_csv(stratification_csv)
    return dict(zip(df["img_name"], df["bin"]))


@torch.no_grad()
def compute_size_stratified_sensitivity(
    model,
    loader,
    bin_lookup: Dict[str, str],
    device,
    score_thresh: float = 0.3,
    dist_thresh_mm: float = 15.0,
    pixel_spacing_mm: float = 0.175,
    input_size: int = 2048,
) -> dict:
    """Returns {'per_bin': {'S': {...}, 'M': {...}, 'L': {...}},
                'macro': float, 'micro': float, 'unbinned': int}."""
    model.eval()
    heatmap_size = model.heatmap_size
    nms_radius = _nms_radius_for(heatmap_size)
    dist_thresh_px = dist_thresh_mm / pixel_spacing_mm
    scale = input_size / heatmap_size

    counters = {b: {"tp": 0, "fn": 0, "n_nod": 0} for b in BINS}
    unbinned = 0

    for images, targets in loader:
        images = images.to(device)
        with autocast(enabled=device.type == "cuda"):
            preds = torch.sigmoid(model(images)).cpu()

        for i in range(preds.shape[0]):
            if not bool(targets["has_nodule"][i]):
                continue                                  # negatives don't contribute to sensitivity
            filename = targets["filename"][i] if "filename" in targets else None
            bin_label = bin_lookup.get(filename)
            if bin_label is None:
                unbinned += 1
                continue

            gt_cx = targets["center_2048"][0][i].item()
            gt_cy = targets["center_2048"][1][i].item()

            peak_coords, peak_scores = _extract_peaks(preds[i, 0], score_thresh, nms_radius)
            detected = False
            if len(peak_scores) > 0:
                order = torch.argsort(peak_scores, descending=True)
                for k in order:
                    py = peak_coords[k, 0].item() * scale
                    px = peak_coords[k, 1].item() * scale
                    if math.sqrt((px - gt_cx) ** 2 + (py - gt_cy) ** 2) <= dist_thresh_px:
                        detected = True
                        break

            counters[bin_label]["n_nod"] += 1
            counters[bin_label]["tp" if detected else "fn"] += 1

    per_bin = {}
    for b in BINS:
        c = counters[b]
        if c["n_nod"] == 0:
            per_bin[b] = {"sensitivity": None, "n_nodules": 0, "tp": 0, "fn": 0}
        else:
            per_bin[b] = {
                "sensitivity": c["tp"] / c["n_nod"],
                "n_nodules": c["n_nod"], "tp": c["tp"], "fn": c["fn"],
            }

    populated = [per_bin[b]["sensitivity"] for b in BINS if per_bin[b]["n_nodules"] > 0]
    macro = sum(populated) / len(populated) if populated else None

    tp_total = sum(counters[b]["tp"] for b in BINS)
    n_total  = sum(counters[b]["n_nod"] for b in BINS)
    micro    = tp_total / n_total if n_total > 0 else None

    return {"per_bin": per_bin, "macro": macro, "micro": micro,
            "unbinned": unbinned, "n_total": n_total}


def _demo() -> None:
    """Assert-based self-check: fake model + fake loader + fake bin lookup.
    Two nodules, one detected → 50% micro. Two bins populated (S detected, L missed) → 50% macro."""
    class _FakeModel:
        heatmap_size = 128
        def eval(self): pass
        def __call__(self, x):
            # Emit peak at pixel (64, 64) — hits centre_2048 == (1024, 1024) after scale=16.
            B = x.shape[0]
            hm = torch.full((B, 1, 128, 128), -5.0)
            hm[:, 0, 64, 64] = 5.0
            return hm

    from types import SimpleNamespace
    device = torch.device("cpu")
    # Two samples: sample 0 has nodule at centre (S bin, detected);
    # sample 1 has nodule off-centre (L bin, missed).
    images  = torch.zeros(2, 1, 2048, 2048)
    targets = {
        "has_nodule":   torch.tensor([True, True]),
        "center_2048": (torch.tensor([1024.0, 200.0]), torch.tensor([1024.0, 200.0])),
        "filename":     ["s.mha", "l.mha"],
    }
    loader = [(images, targets)]
    bin_lookup = {"s.mha": "S", "l.mha": "L"}

    result = compute_size_stratified_sensitivity(_FakeModel(), loader, bin_lookup, device)
    assert result["per_bin"]["S"]["sensitivity"] == 1.0, result
    assert result["per_bin"]["L"]["sensitivity"] == 0.0, result
    assert result["per_bin"]["M"]["n_nodules"] == 0, result
    assert result["macro"] == 0.5, result
    assert result["micro"] == 0.5, result
    print("ok")


if __name__ == "__main__":
    _demo()
