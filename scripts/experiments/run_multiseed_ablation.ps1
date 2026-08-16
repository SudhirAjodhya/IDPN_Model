# run_multiseed_ablation.ps1
# Extends 2026-08-10 ablation_v1 to n=3 on the augmented, fixed pipeline.
#
# Three variants x three seeds on datasets=both, input=2048, target=224,
# batch=2, patience=15, --augment. Enables the pairwise Talebi-vs-bilinear
# comparison matrix_v1 achieved but this time on the corrected src/ pipeline.
#
# Reuses existing s42 outputs from ablation_v1 for bilinear and node21_talebi
# (both already run on the augmented pipeline). Runs the missing 7 configs:
#
#   bilinear         both  s7    s123
#   talebi (0-init)  both  s42   s7    s123     <- variant not yet tested on augmented pipeline
#   node21_talebi    both  s7    s123
#
# Also runs bilinear s42 and node21_talebi s42 into ablation_v2 for coherent
# n=3 data under one run_group (~5h each x 2 = 10h extra; comment them out
# if compute is tight, cross-run_group comparisons work fine).
#
# Order chosen so the most-informative runs finish first:
#   1. talebi (0-init) s42  <- fills the "does the zero-init variant still win
#                              under the fixed pipeline?" gap
#   2-4. one seed per variant   <- gives you n=1 across all three ASAP
#   5-9. remaining seeds        <- fills n=3
#
# ~5h per run at input=2048 batch=2 on RTX 5070 Ti. Full 9 runs ~45h; the
# priority 7 (skip the two reruns) is ~35h. Splits across Fri overnight +
# Sat day + Sat overnight comfortably.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "ablation_v2"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

# Ordered so partial completions still land the most informative results first.
# Comment out any row to skip. The two "rerun" rows are marked; skip them if
# reusing ablation_v1's s42 outputs is fine.
$RUNS = @(
    @{ variant = "talebi";        seed = 42  },   # 1. missing: zero-init on augmented pipeline
    @{ variant = "bilinear";      seed = 7   },   # 2. bilinear s7
    @{ variant = "node21_talebi"; seed = 7   },   # 3. paper-faithful s7
    @{ variant = "talebi";        seed = 7   },   # 4. zero-init s7
    @{ variant = "bilinear";      seed = 123 },   # 5. bilinear s123
    @{ variant = "node21_talebi"; seed = 123 },   # 6. paper-faithful s123
    @{ variant = "talebi";        seed = 123 }    # 7. zero-init s123
    # Optional reruns (skip if reusing ablation_v1 s42) - uncomment and put a
    # comma at the end of the preceding row if enabling either:
    # @{ variant = "bilinear";      seed = 42  }
    # @{ variant = "node21_talebi"; seed = 42  }
)

function Run($label, $args_) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    "=== $label ===" | Add-Content $LOG
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -u scripts/train.py @args_ 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LOG -Append
    $ErrorActionPreference = $prev
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
}

Write-Host "Running smoke test..." -ForegroundColor Yellow
& python scripts/diagnostics/smoke_test_node21_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed. Aborting sweep." }

$RUN_START = Get-Date
foreach ($cfg in $RUNS) {
    $label = "$($cfg.variant)  datasets=both  s$($cfg.seed)"
    Run $label @(
        "--variant",     $cfg.variant,
        "--datasets",    "both",
        "--seed",        "$($cfg.seed)",
        "--target-size", "224",
        "--input-size",  "2048",
        "--batch-size",  "2",
        "--epochs",      "50",
        "--patience",    "15",
        "--run-group",   $RUN_GROUP,
        "--augment",
        "--froc"
    )
    $elapsed = (Get-Date) - $RUN_START
    Write-Host "[timing] elapsed since sweep start: $($elapsed.ToString('hh\:mm'))" -ForegroundColor DarkGray
}

Write-Host "Multi-seed ablation complete." -ForegroundColor Green
Write-Host "Analyse with: compare NODE21 test sens across (variant, seed) triples for n=3 paired-seed Talebi-vs-bilinear delta." -ForegroundColor Green


