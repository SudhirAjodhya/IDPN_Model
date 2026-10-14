"""Walk every v2 checkpoint dir, pull metrics.json + size_stratified.json (if present)
+ froc_data.json (if present), aggregate into one master JSON at
outputs/results/aggregated_v2.json.

Purpose: single-file source for report writing. Every table + plot in the report
should be reconstructable from this file without re-running any model inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent.parent
V8   = ROOT / "outputs" / "v8"
OUT  = ROOT / "outputs" / "results" / "aggregated_v2.json"


def infer_backbone(dirname: str) -> str:
    for bb in ("convnext_fpn_scratch", "resnet50_fpn_scratch", "resnet50_fpn"):
        if bb in dirname:                                                          # substring; also matches when _iXXX_tYYY suffix follows
            return bb
    return "convnext_fpn"

import re
_RES_SUFFIX = re.compile(r"_i(\d+)_t(\d+)(?:_|$)")
# Original v3 sweep runs overwrote each other because dir naming had no _iXXX_tYYY suffix.
# Only the LAST run in each arm survived per date. Map the surviving dirs to their
# ACTUAL (input, target) using the launcher ordering + heatmap_size.
_V3_SURVIVING = {
    # (date_group, variant) -> (input_size, target_size)
    ("2026-08-16_v3_res_sweep", "bilinear"):              (512,  512),
    ("2026-08-17_v3_res_sweep", "bilinear"):              (1024, 1024),
    ("2026-08-16_v3_res_sweep", "node21_talebi_zeroinit"): (512,  224),
    ("2026-08-17_v3_res_sweep", "node21_talebi_zeroinit"): (1024, 224),
}

def infer_input_target(dirname: str, date_group: str, variant: str, target_from_heatmap: int) -> tuple[int, int]:
    """Recover (input_size, target_size) from dir name suffix _iXXX_tYYY.
    New runs have the suffix; older runs (pre-2026-08-17 fix) lack it.
    v3-sweep overwrites are special-cased; other missing-suffix runs default to
    input=2048 (v2 recipe canonical) with target derived from heatmap_size."""
    m = _RES_SUFFIX.search(dirname)
    if m:
        return int(m.group(1)), int(m.group(2))
    if (date_group, variant) in _V3_SURVIVING:
        return _V3_SURVIVING[(date_group, variant)]
    return 2048, target_from_heatmap


def main():
    entries = []
    for group_dir in sorted(V8.glob("*_v?_*")):                                 # v2, v3, ...
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_json = run_dir / "metrics.json"
            if not metrics_json.exists():
                continue                                                        # failed / crashed run

            m = json.loads(metrics_json.read_text())
            target_from_heatmap = m["heatmap_size"] * 4
            input_size, target_size = infer_input_target(run_dir.name, group_dir.name, m["variant"], target_from_heatmap)
            # If new-style metrics.json (has input_size), prefer that as ground truth
            input_size  = m.get("input_size",  input_size)
            target_size = m.get("target_size", target_size)
            entry = {
                "run_group":     m.get("run_group"),
                "date_group":    group_dir.name,
                "variant":       m["variant"],
                "backbone":      m.get("backbone", infer_backbone(run_dir.name)),
                "seed":          m["seed"],
                "head":          m["head"],
                "loss":          m["loss"],
                "input_size":    input_size,
                "target_size":   target_size,
                "epochs_trained": m["epochs_trained"],
                "best_val_sensitivity":      m["best_val_sensitivity"],
                "best_val_sensitivity_epoch": m["best_val_sensitivity_epoch"],
                "peak_sensitivity":          m.get("peak_sensitivity"),
                "test": m["test_results"],                                      # combined / jsrt / node21 / jsrt_by_size
                "checkpoint_path": str((run_dir / "best_model.pth").relative_to(ROOT)),
            }

            # val -> test drop = test_combined_sens - best_val_sens
            entry["val_to_test_drop"] = entry["test"]["combined"]["sensitivity"] - entry["best_val_sensitivity"]

            # size-stratified NODE21 (if we ran run_size_stratified_all)
            strat_json = run_dir / "size_stratified.json"
            if strat_json.exists():
                s = json.loads(strat_json.read_text())
                entry["size_stratified_node21"] = {
                    "per_bin": s["per_bin"],
                    "macro":   s["macro"],
                    "micro":   s["micro"],
                    "unbinned": s["unbinned"],
                }

            # FROC data (if --froc was passed during training)
            froc_json = run_dir / "froc_data.json"
            if froc_json.exists():
                f = json.loads(froc_json.read_text())
                # froc_data has keys thresholds, all, jsrt, node21. Each is a list of
                # {threshold, sensitivity, fp_per_image} at descending threshold.
                # Compute FROC-AUC via trapezoid on operating points below 4 FP/img (Node21 convention).
                entry["froc"] = {}
                for split_key in ("all", "jsrt", "node21"):
                    if split_key not in f: continue
                    pts = sorted((p["fp_per_image"], p["sensitivity"]) for p in f[split_key])
                    auc = 0.0; last_fp = 0.0; last_sens = 0.0
                    for fp, sens in pts:
                        if fp > 4.0: break
                        auc += (fp - last_fp) * (sens + last_sens) / 2
                        last_fp, last_sens = fp, sens
                    entry["froc"][split_key] = {
                        "auc_0_4fp": auc / 4.0,                                 # normalised
                        "curve":     [{"fp": p[0], "sens": p[1]} for p in pts if p[0] <= 4.0],
                    }

            entries.append(entry)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": date.today().isoformat(),
        "n_runs":    len(entries),
        "runs":      entries,
    }, indent=2))
    print(f"Aggregated {len(entries)} v2 runs -> {OUT.relative_to(ROOT)}")

    # Quick summary table by (variant, backbone) across seeds
    from collections import defaultdict
    grouped = defaultdict(list)
    for e in entries:
        grouped[(e["variant"], e["backbone"])].append(e)

    print(f"\n{'variant':38s} {'backbone':24s} {'n':>2s}   {'sens':>6s}   {'macro':>6s}   {'FROC_AUC':>8s}")
    print("-" * 100)
    for (variant, backbone), runs in sorted(grouped.items()):
        sens = [r["test"]["combined"]["sensitivity"] for r in runs]
        macros = [r["size_stratified_node21"]["macro"] for r in runs if "size_stratified_node21" in r]
        aucs = [r["froc"]["all"]["auc_0_4fp"] for r in runs if r.get("froc", {}).get("all")]
        sens_str  = f"{sum(sens)/len(sens):.3f}"
        macro_str = f"{sum(macros)/len(macros):.3f}" if macros else "---"
        auc_str   = f"{sum(aucs)/len(aucs):.3f}"     if aucs   else "---"
        print(f"{variant:38s} {backbone:24s} {len(runs):>2d}   {sens_str:>6s}   {macro_str:>6s}   {auc_str:>8s}")


if __name__ == "__main__":
    main()
