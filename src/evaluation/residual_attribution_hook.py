"""
Thin adapter so training can call the residual-attribution measurement.

The measurement itself lives in scripts/diagnostics/residual_attribution.py, which is a
CLI tool. This module makes it importable from src/ without moving it, so a run can
record its own mechanism readout in metrics.json.
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "diagnostics"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def run_attribution(model, loader, device, input_size, target_size, head, max_batches=None):
    """Gradient and residual enrichment at nodules, against matched control boxes."""
    from residual_attribution import measure_attribution

    return measure_attribution(model, loader, device, input_size, target_size, head, max_batches)
