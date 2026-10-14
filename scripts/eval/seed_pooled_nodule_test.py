"""
Seed-averaged paired comparison of two variants at a matched false-positive budget.

WHY THIS EXISTS. Running one seed per cell and reading the per-cell McNemar leaves two
different uncertainties tangled together:

  seed variance      shrinks as more seeds are run
  test-set sampling  does NOT shrink - all seeds are scored on the same 147 nodules

Reporting six per-cell p-values hides that. Six cells all showing a positive delta looks
like six pieces of evidence, and a sign test over them would read p = 0.031, but the cells
share one test split, so their errors are correlated and that number is not valid.

What this script does instead: for every nodule, average the hit indicator over seeds
within a variant, then compare the two variants nodule-by-nodule on those averages. Seed
noise is averaged down where it belongs - inside the per-nodule estimate - while n stays at
147 nodules, so the surviving uncertainty is honestly the test split's.

Because the seed-averaged per-nodule values are fractions rather than 0/1, McNemar no
longer applies. The paired test on the 147 per-nodule differences is the Wilcoxon signed
rank (distribution-free, no normality assumption on a bounded fraction), with a paired
bootstrap over nodules for the confidence interval.

Every arm's threshold is fitted to the budget independently, exactly as
scripts/eval/paired_nodule_test.py does - never a shared score threshold (trap 4.1).

Run:
  python scripts/eval/seed_pooled_nodule_test.py \
      --a-glob "outputs/**/bilinear_fcos_focal_v5_s*_i1024_t224_vindr_vin2048_vnr1" \
      --b-glob "outputs/**/node21_talebi_zeroinit_fcos_focal_v5_s*_i1024_t224_vindr_vin2048_vnr1"
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.splits import build_three_way_split                      # noqa: E402
from scripts.eval.paired_nodule_test import nodule_hits                # noqa: E402
import scripts.train as T                                              # noqa: E402

BOOTSTRAP_DRAWS = 10000


def wilcoxon_signed_rank(differences):
    """
    Two-sided Wilcoxon signed-rank p-value, normal approximation with tie correction.

    Zeros are dropped (standard Wilcoxon handling) and tied absolute differences share
    their mean rank, which matters here because seed-averaged fractions collide often -
    with 3 seeds a nodule difference can only take a few distinct values.

    Args:
        differences: per-nodule (B - A) values

    Returns:
        (p_value, n_used) - n_used excludes exact zeros
    """
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    n = len(nonzero)

    if n == 0: return 1.0, 0

    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    # Average the ranks inside each group of equal magnitude
    magnitudes = np.abs(nonzero)

    for magnitude in np.unique(magnitudes):
        tied = magnitudes == magnitude
        if tied.sum() > 1: ranks[tied] = ranks[tied].mean()

    positive_sum = ranks[nonzero > 0].sum()
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24

    # Tie correction, which without it overstates significance
    for magnitude in np.unique(magnitudes):
        tied = int((magnitudes == magnitude).sum())
        if tied > 1: variance -= (tied ** 3 - tied) / 48

    if variance <= 0: return 1.0, n

    # Continuity correction
    z = (abs(positive_sum - mean) - 0.5) / variance ** 0.5
    p_value = 2 * (1 - 0.5 * (1 + torch.erf(torch.tensor(z / 2 ** 0.5)).item()))

    return min(1.0, max(0.0, p_value)), n


def bootstrap_interval(differences, draws=BOOTSTRAP_DRAWS, seed=0):
    """Percentile CI for the mean difference, resampling NODULES (the paired unit)."""
    generator = np.random.default_rng(seed)
    n = len(differences)
    means = np.empty(draws)

    for draw in range(draws):
        means[draw] = differences[generator.integers(0, n, n)].mean()

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect(run_dirs, loader, device, budget, label):
    """Per-nodule hit vectors for every run in one arm, stacked (n_seeds, n_nodules)."""
    stack = []

    for run_dir in run_dirs:
        hits, threshold, achieved = nodule_hits(run_dir, loader, device, budget)
        seed = json.load(open(run_dir / "metrics.json"))["seed"]

        print("  {} seed {:<5} sens {:.3f}   threshold {:.4f} -> {:.3f} FP/img".format(
            label, seed, hits.mean(), threshold, achieved))

        stack.append(hits.astype(float))

    return np.vstack(stack)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-glob", required=True, help="glob for baseline run dirs")
    parser.add_argument("--b-glob", required=True, help="glob for learned-resizer run dirs")
    parser.add_argument("--run-group", default=None,
                        help="REQUIRED IN PRACTICE. Substring the run_group must contain. "
                             "Run directory LEAF NAMES are not unique across run groups - "
                             "v14 arm N and v15 t224 have byte-identical leaf names under "
                             "different date parents - so a bare glob silently pools runs "
                             "trained under different recipes and selection rules.")
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="keep at the training value; trap 13.1 makes bf16 evaluation "
                             "batch-size dependent")
    arguments = parser.parse_args()

    def matching(pattern):
        found = []

        for path in glob.glob(pattern, recursive=True):
            run_dir = Path(path)
            metrics_path = run_dir / "metrics.json"

            if not metrics_path.exists(): continue

            metrics = json.load(open(metrics_path))

            if arguments.run_group and arguments.run_group not in str(metrics.get("run_group", "")):
                continue

            found.append(run_dir)

        return sorted(found)

    runs_a = matching(arguments.a_glob)
    runs_b = matching(arguments.b_glob)

    if not runs_a or not runs_b:
        print("no runs matched (a: {}, b: {})".format(len(runs_a), len(runs_b)))
        return

    # THE GUARD THAT MATTERS. A duplicate seed inside one arm means the glob crossed run
    # groups, and the arm is now a mixture of recipes weighted by however many runs each
    # contributed. This actually happened: without --run-group, the t224 globs pulled in
    # v14 arm N alongside v15, giving two "seed 42" runs per arm and a spurious p = 0.0047.
    # Equal seed COUNTS do not catch it - both arms were contaminated symmetrically.
    for label, runs in (("baseline", runs_a), ("learned", runs_b)):
        seeds = [json.load(open(r / "metrics.json"))["seed"] for r in runs]
        duplicates = {s for s in seeds if seeds.count(s) > 1}

        if duplicates:
            print("REFUSING: {} arm has seed(s) {} more than once - the glob is crossing "
                  "run groups. Pass --run-group.".format(label, sorted(duplicates)))

            for run_dir, seed in zip(runs, seeds):
                print("    seed {:<5} {}".format(seed, run_dir))

            return

    # Selection rule must match across every run: a checkpoint chosen on val_sens at a
    # fixed 0.3 threshold is not comparable to one chosen at the matched FP budget, and
    # mixing them is what produced v14's phantom +0.150.
    rules = {json.load(open(r / "metrics.json")).get("select_on") for r in runs_a + runs_b}

    if len(rules) != 1:
        print("REFUSING: runs span selection rules {}. Not comparable.".format(sorted(map(str, rules))))
        return

    # Unequal seed counts would weight the two arms differently, which is exactly the kind
    # of silent asymmetry this project has been bitten by before
    if len(runs_a) != len(runs_b):
        print("REFUSING: {} baseline runs vs {} learned runs. Equal seed counts are "
              "required or the arms are not comparable.".format(len(runs_a), len(runs_b)))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = json.load(open(runs_a[0] / "metrics.json"))["input_size"]

    targets = {json.load(open(r / "metrics.json"))["target_size"] for r in runs_a + runs_b}

    if len(targets) != 1:
        print("REFUSING: runs span target sizes {}. Pool seeds, never targets - a target "
              "change retrains the model and is a different quantity.".format(sorted(targets)))
        return

    _, _, test_loaders, _ = build_three_way_split(
        jsrt_csv_path=T.JSRT_CSV_PATH, jsrt_image_dir=T.JSRT_IMAGE_DIR,
        node21_csv_path=T.NODE21_CSV_PATH, node21_image_dir=T.NODE21_IMAGE_DIR,
        datasets="node21", input_size=input_size, seed=42, test_seed=0,
        val_frac=0.20, group_by_image=True)

    loader = DataLoader(test_loaders["node21"], batch_size=arguments.batch_size,
                        shuffle=False, num_workers=0, pin_memory=True)

    print("\nPer-run sensitivities at {} FP/img (target {})".format(
        arguments.budget, sorted(targets)[0]))

    hits_a = collect(runs_a, loader, device, arguments.budget, "A")
    hits_b = collect(runs_b, loader, device, arguments.budget, "B")

    assert hits_a.shape == hits_b.shape, "hit matrices must align nodule-for-nodule"

    # Average over seeds WITHIN a variant, keeping the nodule axis intact
    mean_a = hits_a.mean(axis=0)
    mean_b = hits_b.mean(axis=0)
    differences = mean_b - mean_a

    n_seeds, n_nodules = hits_a.shape
    delta = differences.mean()

    p_value, n_used = wilcoxon_signed_rank(differences)
    low, high = bootstrap_interval(differences)

    print("\n{} seeds per arm, {} nodules".format(n_seeds, n_nodules))
    print("  seed-averaged sensitivity   A {:.3f}   B {:.3f}".format(
        mean_a.mean(), mean_b.mean()))
    print("  difference                  {:+.4f}".format(delta))
    print("  95% bootstrap CI over nodules  [{:+.4f}, {:+.4f}]".format(low, high))
    print("  Wilcoxon signed rank  p = {:.4f}  ({} nodules differ, {} tied)".format(
        p_value, n_used, n_nodules - n_used))

    # Per-seed spread, to show what replication actually bought
    per_seed = hits_b.mean(axis=1) - hits_a.mean(axis=1)
    print("\n  per-seed deltas  {}".format(
        "  ".join("{:+.3f}".format(d) for d in per_seed)))
    print("  seed spread      {:.4f} (max - min)".format(per_seed.max() - per_seed.min()))

    print("\n  verdict: {} at 0.05".format(
        "SIGNIFICANT" if p_value < 0.05 else "not significant"))
    print("  The CI is over NODULES, the one test split. It does not cover the risk that")
    print("  this split is unrepresentative - only a second test set would.")


if __name__ == "__main__":
    main()
