"""
Read the v9 warp resolution sweep.

Everything is reported at MATCHED FALSE-POSITIVE BUDGETS rather than at a fixed score
threshold. FCOS scores sqrt(cls * centerness), so a fixed threshold sits far right on the
curve; in v8 that made a flat-to-worse variant look like the winner and produced alarming
~0.9 FP/img figures from perfectly ordinary curves.

Residual enrichment is reported alongside, because in v8 it moved 0.765 -> 1.435 while
detection moved 0.000 - it is the more sensitive instrument, and the two can disagree.

Runs are gathered from ALL fcos run groups, since the bilinear controls at 1024->384 and
1024->512 were produced in v7/v8 under identical recipe defaults.

Run: python scripts/eval/analyse_v9.py
"""

import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

BUDGETS = (0.25, 0.5, 1.0)
BINOMIAL_SE = 0.041
HEAD = "fcos"


def load():
    """(variant, input, target) -> list of per-seed records, fcos and squash only."""
    runs = {}

    for metrics_path in glob.glob('outputs/**/metrics.json', recursive=True):
        d = json.load(open(metrics_path))

        if d.get('head') != HEAD or not d.get('box_metrics') or d.get('letterbox'): continue

        froc_path = os.path.join(os.path.dirname(metrics_path), 'froc_data.json')

        if not os.path.exists(froc_path): continue

        points = json.load(open(froc_path)).get('node21') or []

        if not points: continue

        fp = np.array([p['fp_per_image'] for p in points])
        se = np.array([p['sensitivity'] for p in points])
        order = np.argsort(fp)
        fp, se = fp[order], se[order]

        record = {
            'seed': d['seed'],
            'fixed': [float(np.interp(b, fp, se, left=0.0, right=se[-1])) for b in BUDGETS],
            'froc': d['box_metrics']['iou0.25']['froc_auc'],
            'enrichment': (d.get('attribution') or {}).get('residual_enrichment'),
        }

        key = (d['variant'], d['input_size'], d['target_size'])
        runs.setdefault(key, [])

        # A config can appear in more than one run group; keep one record per seed
        if record['seed'] not in [r['seed'] for r in runs[key]]:
            runs[key].append(record)

    return runs


def summarise(records):
    fixed = [st.mean(r['fixed'][i] for r in records) for i in range(len(BUDGETS))]
    froc = st.mean(r['froc'] for r in records)
    enrichments = [r['enrichment'] for r in records if r['enrichment'] is not None]

    return fixed, froc, (st.mean(enrichments) if enrichments else None), sorted(r['seed'] for r in records)


def table(title, rows, runs, baseline_key=None):
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(f"\n  {'variant':<20}{'in->tgt':>11}{'n':>3}" +
          "".join(f"@{b}FP".rjust(9) for b in BUDGETS) + f"{'FROC':>8}{'enrich':>9}  seeds")

    baseline = {}

    for label, key in rows:
        if key not in runs:
            print(f"  {label:<20}{f'{key[1]}->{key[2]}':>11}  -   (not run)")
            continue

        fixed, froc, enrichment, seeds = summarise(runs[key])
        enrich_text = f"{enrichment:9.3f}" if enrichment is not None else "      n/a"
        print(f"  {label:<20}{f'{key[1]}->{key[2]}':>11}{len(seeds):>3}" +
              "".join(f"{x:9.3f}" for x in fixed) + f"{froc:8.3f}{enrich_text}  {seeds}")

        if baseline_key and key[0] == baseline_key:
            baseline[(key[1], key[2])] = fixed

    return baseline



def report_iso_compute(runs):
    """
    The comparison the learned-resizer premise actually makes.

    The backbone dominates cost and runs at TARGET resolution, so warp X->224 and
    bilinear 1024->224 cost the same to run, while bilinear 1024->384 costs roughly 3x
    more. The claim worth testing is not "warp beats bilinear at matched target" but
    "warp at a cheap target reaches bilinear at an expensive one" - i.e. the resizer
    preserves enough information to buy back compute.
    """
    print()
    print("" + "=" * 88)
    print("ISO-COMPUTE -- does warp at target 224 reach bilinear at a LARGER target?")
    print("  Backbone cost scales with target, so every warp row below is ~as cheap as")
    print("  bilinear 1024->224. Rows above it in the bilinear column cost more to run.")
    print("=" * 88)

    ladder = []
    for t in (224, 256, 384, 512):
        key = ("bilinear", 1024, t)
        if key in runs:
            fixed, froc, _, seeds = summarise(runs[key])
            ladder.append((t, fixed[0], froc, len(seeds)))

    if not ladder:
        print()
        print("no bilinear reference runs yet")
        return

    print()
    print(f"bilinear reference ladder (sensitivity @0.25 FP):")
    for t, sens, froc, n in ladder:
        print(f"    1024->{t:<5} {sens:.3f}   FROC {froc:.3f}   (n={n})")

    print()
    print(f"warp at target 224, and the bilinear target it matches:")
    any_warp = False

    for i in (224, 256, 384, 512, 1024):
        key = ("node21_ldm_warp", i, 224)

        if key not in runs: continue

        any_warp = True
        fixed, froc, enrichment, seeds = summarise(runs[key])
        sens = fixed[0]

        # Highest bilinear target this warp run matches or beats
        matched = [t for t, b_sens, _, _ in ladder if sens >= b_sens - BINOMIAL_SE]
        best = max(matched) if matched else None
        verdict = f"matches bilinear 1024->{best}" if best else "below bilinear at every target"
        saving = ""

        if best and best > 224:
            # Backbone compute scales roughly with target^2
            saving = f"  [~{(best / 224) ** 2:.1f}x cheaper backbone]"

        enrich_text = f"enrich {enrichment:.3f}" if enrichment is not None else "enrich n/a"
        print(f"    warp {i:>4}->224  {sens:.3f}   {enrich_text:<14} {verdict}{saving}")

    if not any_warp:
        print("    (no warp X->224 runs yet)")

    print()
    print("Reading: a warp row matching a bilinear row ABOVE 224 means the resizer")
    print("  preserved enough information to run the backbone smaller for the same")
    print("  detection performance. That is the learned-resizer premise, and it does not")
    print("  require beating bilinear at matched target.")


def main():
    runs = load()
    print(f"\nloaded {sum(len(v) for v in runs.values())} fcos runs across {len(runs)} configs")

    # ---- Arms B and R1: matched input, varying target ----
    rows = []
    for t in (224, 256, 384, 512):
        rows.append(("bilinear", ("bilinear", 1024, t)))
        rows.append(("  ldm_warp", ("node21_ldm_warp", 1024, t)))

    print()
    table("ARMS B vs R1 -- input fixed at 1024, target varies. Matched pairs.",
          rows, runs)

    print("\n  delta (warp - bilinear) at each target:")
    for t in (224, 256, 384, 512):
        b_key, w_key = ("bilinear", 1024, t), ("node21_ldm_warp", 1024, t)

        if b_key not in runs or w_key not in runs:
            print(f"    1024->{t:<5} incomplete")
            continue

        b_fixed, _, _, _ = summarise(runs[b_key])
        w_fixed, _, _, _ = summarise(runs[w_key])
        deltas = [w - b for w, b in zip(w_fixed, b_fixed)]
        verdict = "inside noise" if all(abs(d) < BINOMIAL_SE for d in deltas) else "OUTSIDE noise at some budget"
        print(f"    1024->{t:<5}" + "".join(f"{d:+9.3f}" for d in deltas) + f"   {verdict}")

    # ---- Arm R2: target fixed at 224, input varies ----
    rows = [("ldm_warp", ("node21_ldm_warp", i, 224)) for i in (224, 256, 384, 512, 1024)]
    print()
    table("ARM R2 -- target pinned at 224, input varies. Does more input help the resizer?",
          rows, runs)

    present = [(i, runs[("node21_ldm_warp", i, 224)]) for i in (224, 256, 384, 512, 1024)
               if ("node21_ldm_warp", i, 224) in runs]

    if len(present) >= 3:
        values = [summarise(r)[0][0] for _, r in present]      # sensitivity @0.25 FP
        rising = all(a <= b for a, b in zip(values, values[1:]))
        falling = all(a >= b for a, b in zip(values, values[1:]))
        spread = max(values) - min(values)

        print(f"\n  @0.25 FP across inputs {[i for i, _ in present]}: "
              + " ".join(f"{v:.3f}" for v in values))
        print(f"  spread {spread:.3f}", end="  ")

        if spread < BINOMIAL_SE:
            print("=> flat within noise: extra input resolution neither helps nor hurts the resizer")
        elif rising:
            print("=> monotone RISING: the resizer converts extra input into detection")
        elif falling:
            print("=> monotone FALLING: extra input hurts, as the pre-v6 metric wrongly suggested")
        else:
            print("=> non-monotone; treat as noise unless the spread is large")

    report_iso_compute(runs)

    print(f"\nbinomial SE at n=147 test nodules is +/-{BINOMIAL_SE:.3f}; FCOS seed spread measured 0.020 in v8")
    print("v8 reference at 1024->384: bilinear 0.307/0.614/0.798 FROC 0.790; warp 0.309/0.614/0.801 FROC 0.788 enrich 1.435\n")


if __name__ == "__main__":
    main()
