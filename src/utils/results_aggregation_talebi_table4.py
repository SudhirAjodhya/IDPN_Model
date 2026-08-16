"""
Aggregate per-run JSONs from the Talebi Table 4 sweep into a paper-format
markdown table + delta summary, and call the Table 4 plot generators.

Reads:  outputs/talebi_table4/{backbone}/results/{config}_seed{seed}.json
Writes:
  outputs/talebi_table4/{backbone}/TABLE4.md         # per-backbone paper table
  outputs/talebi_table4/{backbone}/plots/*.png       # aggregate plots
  outputs/talebi_table4/TABLE4_all.md                # cross-backbone summary
"""
from __future__ import annotations
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.experiments.talebi_table4 import (
    ALL_CONFIGS, DEFAULT_BACKBONES, BILINEAR_CONFIGS, PROPOSED_CONFIGS,
)
from src.training.classification_trainer import (
    OUTPUT_ROOT, backbone_dir, plots_dir,
)
from src.visualization.table4_plots import save_all_table4_plots


def _load_runs(backbone: str) -> list[dict]:
    d = backbone_dir(backbone) / "results"
    if not d.exists():
        return []
    runs = []
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if not r.get("smoke", False):
            runs.append(r)
    return runs


def _mean_range(xs):
    if not xs:
        return None, None, None
    return statistics.mean(xs), min(xs), max(xs)


def _paper_table_md(backbone: str, runs: list[dict]) -> str:
    """
    Build a paper-Table-4 style markdown table for one backbone.

    Rows are ordered like paper: all Proposed rows first (by input res),
    dashed separator, then all Bilinear rows.
    """
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_config[r["config"]["label"]].append(r)

    def row_lines(configs):
        lines = []
        for cfg in configs:
            rs = by_config.get(cfg.label, [])
            if not rs:
                lines.append(
                    f"| {backbone} | {cfg.resizer or 'bilinear'} "
                    f"| {cfg.input_res}×{cfg.input_res} | {cfg.target_res}×{cfg.target_res} "
                    f"| {cfg.batch_size} | — | — | — | (no runs) |"
                )
                continue
            gmacs = rs[0]["macs"] / 1e9
            f_top1 = [100.0 - r["final_val_top1"] for r in rs]
            f_top5 = [100.0 - r["final_val_top5"] for r in rs]
            b_top1 = [100.0 - r["best_val_top1"] for r in rs]
            fm, flo, fhi = _mean_range(f_top1)
            f5m, _, _    = _mean_range(f_top5)
            bm, _, _     = _mean_range(b_top1)
            n = len(rs)
            lines.append(
                f"| {backbone} | {cfg.resizer or 'bilinear'} "
                f"| {cfg.input_res}×{cfg.input_res} | {cfg.target_res}×{cfg.target_res} "
                f"| {cfg.batch_size} "
                f"| {fm:.2f} [{flo:.2f}, {fhi:.2f}] "
                f"| {f5m:.2f} "
                f"| {bm:.2f} "
                f"| {gmacs:.2f} "
                f"| n={n} |"
            )
        return lines

    hdr = [
        f"# Talebi Table 4 — {backbone}",
        "",
        "Paper §4 recipe reproduced on ImageNette (10-class ImageNet subset).",
        "Errors are `100 - accuracy` to align with paper Table 4 direction (↓ better).",
        "'Final' = last-epoch metrics (paper-comparable). 'Best' = min val_loss checkpoint.",
        "",
        "| Model | Resizer | Input res | Output res | Batch "
        "| Top-1 err (final, mean [min, max]) "
        "| Top-5 err (final, mean) "
        "| Top-1 err (best-val) "
        "| GMACs "
        "| Seeds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines = hdr + row_lines(PROPOSED_CONFIGS) + ["|" + " --- |" * 10] + row_lines(BILINEAR_CONFIGS)

    # Delta section: Proposed_R vs Bilinear_224, per R
    lines += ["", "## Δ Top-1 Error vs Bilinear-224 (mean of seeds)", ""]
    lines += ["| Config | Δ top-1 err (pp, negative = improvement over bilinear-224) |",
              "|---|---|"]
    bil_224 = by_config.get("bilinear_224", [])
    if bil_224:
        b224_mean = statistics.mean([100.0 - r["final_val_top1"] for r in bil_224])
        for cfg in PROPOSED_CONFIGS:
            rs = by_config.get(cfg.label, [])
            if not rs:
                continue
            p_mean = statistics.mean([100.0 - r["final_val_top1"] for r in rs])
            delta = p_mean - b224_mean  # negative = better (lower err)
            lines.append(f"| {cfg.label} vs bilinear_224 | {delta:+.2f} |")
    else:
        lines.append("| (bilinear_224 runs missing — cannot compute Δ) |  |")

    return "\n".join(lines) + "\n"


def _cross_backbone_md(backbones: list[str]) -> str:
    """One-page summary across all backbones: headline Δ at the paper's best row."""
    lines = ["# Talebi Table 4 — All Backbones", "",
             "Headline: bilinear-224 vs proposed_368→224 (paper's best row for R50).", ""]
    lines += ["| Backbone | Bilinear-224 top-1 err (%) | Proposed 368→224 top-1 err (%) "
              "| Δ top-1 err (pp) | Paper Δ (pp) |",
              "|---|---|---|---|---|"]
    paper_deltas = {"resnet50": -1.7, "mobilenet_v2": -1.1,
                    "densenet121": -2.9, "inception_v2": -2.6}
    for bb in backbones:
        runs = _load_runs(bb)
        if not runs:
            lines.append(f"| {bb} | — | — | — | {paper_deltas.get(bb, '?')} |")
            continue
        by_cfg = defaultdict(list)
        for r in runs:
            by_cfg[r["config"]["label"]].append(r)
        b224 = [100.0 - r["final_val_top1"] for r in by_cfg.get("bilinear_224", [])]
        p368 = [100.0 - r["final_val_top1"] for r in by_cfg.get("proposed_368", [])]
        if not b224 or not p368:
            lines.append(f"| {bb} | {statistics.mean(b224) if b224 else '—'} "
                         f"| {statistics.mean(p368) if p368 else '—'} "
                         f"| — | {paper_deltas.get(bb, '?')} |")
            continue
        bm, pm = statistics.mean(b224), statistics.mean(p368)
        lines.append(f"| {bb} | {bm:.2f} | {pm:.2f} | {pm - bm:+.2f} "
                     f"| {paper_deltas.get(bb, '?')} |")
    return "\n".join(lines) + "\n"


def summarize(backbones: list[str] | None = None) -> None:
    backbones = backbones or DEFAULT_BACKBONES
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for bb in backbones:
        runs = _load_runs(bb)
        if not runs:
            print(f"[{bb}] no completed runs — skipping")
            continue
        md = _paper_table_md(bb, runs)
        out = backbone_dir(bb) / "TABLE4.md"
        out.write_text(md, encoding="utf-8")
        print(f"[{bb}] wrote {out}  ({len(runs)} runs)")

        save_all_table4_plots(runs, plots_dir(bb), backbone=bb)
        print(f"[{bb}] wrote aggregate plots -> {plots_dir(bb)}")

    all_md = _cross_backbone_md(backbones)
    (OUTPUT_ROOT / "TABLE4_all.md").write_text(all_md, encoding="utf-8")
    print(f"wrote cross-backbone summary -> {OUTPUT_ROOT / 'TABLE4_all.md'}")


if __name__ == "__main__":
    summarize()
