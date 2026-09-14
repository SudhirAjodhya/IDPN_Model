# run_v3_resolution_sweep.ps1
# Resolution sweep for the Talebi Table 4 analogue plot (input res vs sensitivity).
#
# Two arms, s42 only (1 seed for the sweep -- cost/benefit choice, not statistical):
#
#   ARM A (bilinear reference): input=target=X, backbone runs at X features
#     - As X grows, backbone gets more info AND more compute. Reference for "how does
#       plain training benefit from higher res?"
#
#   ARM B (zero-init Talebi, aggressive downsample): input=X, target=224, backbone at 224
#     - Talebi's LDM does the input->224 downsample. Backbone compute FIXED at 224
#       regardless of input res. Tests: "how much info can the learned resizer preserve
#       from a higher-res input into a small feature map?"
#
# 10 runs total (5 res values x 2 arms). All use v2 recipe (batch=8 lr=2e-4 no-cosine
# val_frac=0.20 augment CLAHE-cache).
#
# Cost estimate (rough, backbone-dominated):
#   Bilinear: 224 30min, 256 30min, 384 1h, 512 2h, 1024 4-6h  =>  ~9-11h
#   Talebi:  all 224 backbone, but LDM at input res              =>  30-45min each x 5 = ~3-4h
#   TOTAL: 12-15h. Two nights or one long day.
#
# Ordering: smallest first (fastest, most likely to run to completion).

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$RUN_GROUP = "v3_res_sweep"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

$SEED = 42

# Micro-batch shrinks with input res to fit 16 GB VRAM; grad_accum keeps effective
# batch = 8 constant so LR stays 2e-4 across all runs (no confound with LR scaling).
# Bilinear arm feels this most (backbone at 1024^2 features is expensive); Talebi arm
# keeps the same schedule for symmetry even though its backbone always runs at 224^2.
$SCHEDULE = @(
    @{ res = 224;  batch = 8; accum = 1 },
    @{ res = 256;  batch = 8; accum = 1 },
    @{ res = 384;  batch = 8; accum = 1 },
    @{ res = 512;  batch = 4; accum = 2 },
    @{ res = 1024; batch = 2; accum = 4 }
)

# Arm A: bilinear at input=target=X
# Arm B: zero-init Talebi at input=X, target=224
$RUNS = @()
foreach ($s in $SCHEDULE) {
    $RUNS += @{ variant = "bilinear";               input = $s.res; target = $s.res; batch = $s.batch; accum = $s.accum }
}
foreach ($s in $SCHEDULE) {
    $RUNS += @{ variant = "node21_talebi_zeroinit"; input = $s.res; target = 224;    batch = $s.batch; accum = $s.accum }
}

function Run($label, $args_) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    "=== $label ===" | Add-Content $LOG
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -u scripts/train.py @args_ 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LOG -Append
    $ErrorActionPreference = $prev
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$label failed with exit code $LASTEXITCODE, continuing" -ForegroundColor Red
        "$label failed with exit code $LASTEXITCODE" | Add-Content $LOG
    }
}

Write-Host "Running smoke test..." -ForegroundColor Yellow
& python scripts/diagnostics/smoke_test_node21_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed. Aborting sweep." }

$RUN_START = Get-Date
foreach ($cfg in $RUNS) {
    $label = "convnext_fpn / $($cfg.variant) input=$($cfg.input) target=$($cfg.target) batch=$($cfg.batch) accum=$($cfg.accum) s$SEED"
    Run $label @(
        "--backbone",    "convnext_fpn",
        "--variant",     $cfg.variant,
        "--seed",        "$SEED",
        "--datasets",    "both",
        "--target-size", "$($cfg.target)",
        "--input-size",  "$($cfg.input)",
        "--batch-size",  "$($cfg.batch)",
        "--grad-accum",  "$($cfg.accum)",
        "--lr",          "2e-4",
        "--epochs",      "50",
        "--patience",    "15",
        "--val-frac",    "0.20",
        "--run-group",   $RUN_GROUP,
        "--augment",
        "--froc"
    )
    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] cumulative elapsed: {0:hh\:mm}" -f $elapsed) -ForegroundColor DarkGray
}

Write-Host "`nv3 res sweep complete. Log: $LOG" -ForegroundColor Green
Write-Host "After it finishes:" -ForegroundColor Yellow
Write-Host "  1. python scripts/eval/run_size_stratified_all.py    (adds size-strat JSON for new runs)" -ForegroundColor DarkGray
Write-Host "  2. python scripts/eval/aggregate_v2_results.py       (extends aggregated JSON)" -ForegroundColor DarkGray
Write-Host "  3. TODO: plot_v3_resolution_sweep.py                  (line plot: sens vs input res)" -ForegroundColor DarkGray
