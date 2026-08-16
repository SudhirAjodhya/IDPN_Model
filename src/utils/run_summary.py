import json
from pathlib import Path

def generate_run_summary(run_group_dir: Path):
    run_group_dir = Path(run_group_dir)
    rows = []
    for metrics_path in sorted(run_group_dir.glob("*/metrics.json")):
        with open(metrics_path) as f:
            m = json.load(f)
        combined = m.get("test_results", {}).get("combined", {})
        jsrt     = m.get("test_results", {}).get("jsrt",     {})
        node21   = m.get("test_results", {}).get("node21",   {})
        rows.append({
            "variant":               m.get("variant", "?"),
            "seed":                  m.get("seed", "?"),
            "loss":                  m.get("loss", "?"),
            "head":                  m.get("head", "?"),
            "best_epoch":            m.get("best_val_sensitivity_epoch", "?"),
            "best_val_sensitivity":  m.get("best_val_sensitivity", float("nan")),
            "comb_sens":             combined.get("sensitivity", float("nan")),
            "jsrt_sens":             jsrt.get("sensitivity",     float("nan")),
            "node21_sens":           node21.get("sensitivity",   float("nan")),
            "fp_per_image":          combined.get("fp_per_image",float("nan")),
        })

    out_path = run_group_dir / "run_summary.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=4)

    hdr = (f"{'Variant':<22} | {'Seed':>4} | {'Loss':<9} | {'Head':<13} | "
           f"{'BestEp':>6} | {'ValSens':>7} | {'CombSens':>8} | "
           f"{'JSRTSens':>8} | {'N21Sens':>7} | {'FP/img':>6}")
    sep = "-" * len(hdr)
    print("=" * len(hdr))
    print(f"Run group: {run_group_dir.name}")
    print("=" * len(hdr))
    print(hdr)
    print(sep)
    for r in rows:
        print(f"{r['variant']:<22} | {str(r['seed']):>4} | {r['loss']:<9} | "
              f"{r['head']:<13} | {str(r['best_epoch']):>6} | "
              f"{r['best_val_sensitivity']:>7.3f} | {r['comb_sens']:>8.3f} | "
              f"{r['jsrt_sens']:>8.3f} | {r['node21_sens']:>7.3f} | {r['fp_per_image']:>6.2f}")
    print("=" * len(hdr))
    print(f"Summary saved: {out_path}")
    return rows
