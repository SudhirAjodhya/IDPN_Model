"""
Results aggregation for the Talebi reproduction experiment.

Parses per-phase per-seed JSON result files and generates a Markdown
summary table with accuracy deltas (Talebi - Bilinear) across seeds.
"""

import json
import statistics
from pathlib import Path

from src.training.classification_trainer import RESULT_DIR


def summarize(result_dir: Path | None = None) -> None:
    """
    Aggregate all phase*_seed*.json files into SUMMARY.md.

    Args:
        result_dir: Directory containing result JSONs. Defaults to RESULT_DIR.
    """
    result_dir = result_dir or RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = {"A": [], "B": []}

    for f in sorted(result_dir.glob("phase*_seed*.json")):
        d = json.loads(f.read_text())

        if d["phase"] in rows and not d.get("smoke", False):
            rows[d["phase"]].append(d)

    lines = [
        "# Talebi Reproduce v1 - Results", "",
        "Paper headline (ResNet-50, ImageNet-1k):",
        "  bilinear-224 top-1 err 24.7%  ->  proposed 368->224 top-1 err 23.0%",
        "  **paper Delta = +1.7 pp top-1**",
        "",
        "Our reproduction reports top-1 *accuracy* (not error). Delta = Talebi_acc - Bilinear_acc.",
        "Success: Delta positive, > seed range, same order of magnitude as paper.",
        "",
        "| Config | Seed | Val top-1 (%) | Val top-5 (%) |",
        "|---|---|---|---|",
    ]

    for d in rows["A"]:
        lines.append(f"| Bilinear 224 (Phase A) | {d['seed']} | "
                     f"{d['final_val_top1']:.2f} | {d['final_val_top5']:.2f} |")
    for d in rows["B"]:
        lines.append(f"| Talebi 368->224 (Phase B) | {d['seed']} | "
                     f"{d['final_val_top1']:.2f} | {d['final_val_top5']:.2f} |")

    by_A = {d["seed"]: d for d in rows["A"]}
    by_B = {d["seed"]: d for d in rows["B"]}
    seeds = sorted(set(by_A) & set(by_B))

    if seeds:
        d1 = [by_B[s]["final_val_top1"] - by_A[s]["final_val_top1"] for s in seeds]
        d5 = [by_B[s]["final_val_top5"] - by_A[s]["final_val_top5"] for s in seeds]

        lines += ["",
            f"**Delta top-1 (Talebi - Bilinear): mean = {statistics.mean(d1):+.2f} pp, "
            f"range = [{min(d1):+.2f}, {max(d1):+.2f}] over seeds {seeds}**",
            f"**Delta top-5:                     mean = {statistics.mean(d5):+.2f} pp, "
            f"range = [{min(d5):+.2f}, {max(d5):+.2f}]**",
        ]

    out = result_dir / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print(out.read_text())
    print(f"\nwrote {out}")
