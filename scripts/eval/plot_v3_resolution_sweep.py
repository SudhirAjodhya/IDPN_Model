"""Talebi Table 4 analogue: combined-test sens vs input resolution.

Two lines:
  - bilinear at input=target=X
  - Talebi zeroinit at input=X, target=224
Both from datasets=both recovery runs (v3_res_sweep + surviving 512/1024).

If native-1024 sweep data (datasets=node21) is present, also draws a third line
of Talebi at input=1024, target=X (varying target on the same native input).

Reads outputs/results/aggregated_v2.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
IN   = ROOT / "outputs" / "results" / "aggregated_v2.json"
OUT  = ROOT / "outputs" / "results" / "plots"


def main():
    data = json.loads(IN.read_text())
    runs = data["runs"]

    # Arm A: bilinear at input=target=X, datasets=both (v3 recovery + surviving)
    # Arm B: Talebi zeroinit at input=X, target=224, datasets=both (v3 recovery + surviving)
    bilin_XX  = {}                                                                # input -> (sens, jsrt_sens, node21_sens)
    talebi_X_224 = {}
    # Native-1024 arm: Talebi zeroinit at input=1024, target=X, datasets=node21
    talebi_1024_Y = {}
    bilin_1024_Y  = {}                                                            # for paired native reference

    for r in runs:
        if r["backbone"] != "convnext_fpn": continue
        v           = r["variant"]
        input_size  = r["input_size"]                                              # aggregate now derives from dir name for old runs
        target_size = r["target_size"]
        test        = r["test"]

        # Categorise
        if v == "bilinear" and input_size == target_size and r["seed"] == 42:
            bilin_XX[input_size] = test
        elif v == "node21_talebi_zeroinit" and target_size == 224 and r["seed"] == 42:
            if input_size == 1024:
                talebi_1024_Y[input_size] = test                                   # will be duplicated by target=224 entry
            talebi_X_224[input_size] = test
        elif v == "node21_talebi_zeroinit" and input_size == 1024 and r["seed"] == 42:
            talebi_1024_Y[target_size] = test
        elif v == "bilinear" and input_size == 1024 and r["seed"] == 42:
            bilin_1024_Y[target_size] = test

    # Combined-test sens on both/JSRT+NODE21 or NODE21 for node21-only runs
    def sens(test_dict, prefer="combined"):
        return test_dict.get(prefer, test_dict.get("node21", {})).get("sensitivity", np.nan)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ============================================================
    # Panel 1 — v3 resolution sweep: sens vs input res
    # ============================================================
    xs = sorted(set(bilin_XX) | set(talebi_X_224))
    ys_bilin  = [sens(bilin_XX[x],  "combined") if x in bilin_XX  else np.nan for x in xs]
    ys_talebi = [sens(talebi_X_224[x], "combined") if x in talebi_X_224 else np.nan for x in xs]
    a1.plot(xs, ys_bilin,  marker="o", color="#4C72B0", linewidth=2, label="Bilinear (input=target=X)")
    a1.plot(xs, ys_talebi, marker="s", color="#55A868", linewidth=2, label="Talebi zero-init (input=X, target=224)")
    a1.set_xscale("log", base=2)
    a1.set_xticks(xs); a1.set_xticklabels([str(x) for x in xs])
    a1.set_xlabel("Input resolution (px)")
    a1.set_ylabel("Combined-test sensitivity")
    a1.set_title("Original v3 sweep — datasets=both, s42")
    a1.grid(alpha=0.3)
    a1.legend(loc="lower right", fontsize=9)
    for x, y in zip(xs, ys_bilin):
        if not np.isnan(y):  a1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#4C72B0")
    for x, y in zip(xs, ys_talebi):
        if not np.isnan(y):  a1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#55A868")

    # ============================================================
    # Panel 2 — native-1024 sweep: sens vs target res (paired)
    # ============================================================
    ys_tgt = sorted(set(talebi_1024_Y) | set(bilin_1024_Y))
    ys_bil_nat = [sens(bilin_1024_Y[t],  "node21") if t in bilin_1024_Y  else np.nan for t in ys_tgt]
    ys_tal_nat = [sens(talebi_1024_Y[t], "node21") if t in talebi_1024_Y else np.nan for t in ys_tgt]
    if ys_tgt:
        a2.plot(ys_tgt, ys_bil_nat, marker="o", color="#4C72B0", linewidth=2, label="Bilinear (input=1024→T)")
        a2.plot(ys_tgt, ys_tal_nat, marker="s", color="#55A868", linewidth=2, label="Talebi zero-init (input=1024→T)")
        a2.set_xticks(ys_tgt); a2.set_xticklabels([str(t) for t in ys_tgt])
    a2.set_xlabel("Target resolution T (px), input fixed at 1024")
    a2.set_ylabel("NODE21 test sensitivity")
    a2.set_title("Native-1024 sweep — datasets=node21, s42")
    a2.grid(alpha=0.3)
    a2.legend(loc="lower right", fontsize=9)
    for t, y in zip(ys_tgt, ys_bil_nat):
        if not np.isnan(y):  a2.annotate(f"{y:.3f}", (t, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#4C72B0")
    for t, y in zip(ys_tgt, ys_tal_nat):
        if not np.isnan(y):  a2.annotate(f"{y:.3f}", (t, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#55A868")

    fig.suptitle("v3 resolution sweeps — Talebi Table 4 analogue on CXR nodule detection", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"resolution_sweep.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote resolution_sweep.{{png,pdf}} -> {OUT.relative_to(ROOT)}/")

    # Diagnostic dump
    print("\nData used:")
    print(f"  Arm A bilinear input=target inputs: {sorted(bilin_XX)}")
    print(f"  Arm B Talebi input=X target=224 inputs: {sorted(talebi_X_224)}")
    print(f"  Native-1024 bilinear targets: {sorted(bilin_1024_Y)}")
    print(f"  Native-1024 Talebi targets: {sorted(talebi_1024_Y)}")


if __name__ == "__main__":
    main()
