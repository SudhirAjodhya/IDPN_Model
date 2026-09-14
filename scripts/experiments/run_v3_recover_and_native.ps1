# run_v3_recover_and_native.ps1
# Two-part overnight sweep:
#
# Part A  — RECOVERY of the 6 checkpoints lost to the v3 output_dir overwrite bug.
#           Same v3 sweep matrix (input=target=X for bilinear, input=X target=224 for Talebi)
#           but re-run at low res only; 512 and 1024 already survived. Uses datasets=both
#           for continuity with the original v3 sweep protocol.
#
# Part B  — NATIVE-INPUT SWEEP. Talebi loaded at NODE21's native 1024 (no bilinear pre-
#           downsampling by cv2.resize) then LDM does the entire compression 1024 -> T.
#           Isolates Talebi's downsampling capability, unconfounded by pre-model bilinear
#           info loss. Uses datasets=node21 because JSRT native is 2048 -> would still
#           get pre-downsampled by cv2.resize(2048->1024).
#
#           Paired bilinear controls at same input=1024 target=T so we can compute the
#           Talebi-vs-bilinear delta at each target.
#
# 14 runs total, s42, ~9h. Fits overnight.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$RUN_GROUP = "v3_res_sweep"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "recover_and_native_$STAMP.log"

$SEED = 42

# Part A — recovery. datasets="both" to match original v3 sweep.
$RUNS_A = @(
    @{ variant = "bilinear";                input = 224; target = 224; batch = 8; accum = 1; datasets = "both" },
    @{ variant = "bilinear";                input = 256; target = 256; batch = 8; accum = 1; datasets = "both" },
    @{ variant = "bilinear";                input = 384; target = 384; batch = 8; accum = 1; datasets = "both" },
    @{ variant = "node21_talebi_zeroinit";  input = 224; target = 224; batch = 8; accum = 1; datasets = "both" },
    @{ variant = "node21_talebi_zeroinit";  input = 256; target = 224; batch = 8; accum = 1; datasets = "both" },
    @{ variant = "node21_talebi_zeroinit";  input = 384; target = 224; batch = 8; accum = 1; datasets = "both" }
)

# Part B — native-input sweep. datasets="node21" so NODE21 (native 1024) stays untouched
# by the dataloader; LDM does the entire compression job. JSRT excluded because its
# native res (2048) would still trigger pre-downsampling at input=1024.
# batch=8 fine for all (backbone at 224-512 features).
$RUNS_B = @(
    @{ variant = "bilinear";                input = 1024; target = 224; batch = 8; accum = 1; datasets = "node21" },
    @{ variant = "bilinear";                input = 1024; target = 256; batch = 8; accum = 1; datasets = "node21" },
    @{ variant = "bilinear";                input = 1024; target = 384; batch = 8; accum = 1; datasets = "node21" },
    @{ variant = "bilinear";                input = 1024; target = 512; batch = 4; accum = 2; datasets = "node21" },
    @{ variant = "node21_talebi_zeroinit";  input = 1024; target = 224; batch = 8; accum = 1; datasets = "node21" },
    @{ variant = "node21_talebi_zeroinit";  input = 1024; target = 256; batch = 8; accum = 1; datasets = "node21" },
    @{ variant = "node21_talebi_zeroinit";  input = 1024; target = 384; batch = 8; accum = 1; datasets = "node21" },
    @{ variant = "node21_talebi_zeroinit";  input = 1024; target = 512; batch = 4; accum = 2; datasets = "node21" }
)

$RUNS = $RUNS_A + $RUNS_B

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
    $label = "convnext_fpn / $($cfg.variant) input=$($cfg.input) target=$($cfg.target) batch=$($cfg.batch) accum=$($cfg.accum) datasets=$($cfg.datasets) s$SEED"
    Run $label @(
        "--backbone",    "convnext_fpn",
        "--variant",     $cfg.variant,
        "--seed",        "$SEED",
        "--datasets",    $cfg.datasets,
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

Write-Host "`nRecover + native complete. Log: $LOG" -ForegroundColor Green
Write-Host "New naming: _i{input}_t{target} suffix on all output dirs -> no more overwrite collisions." -ForegroundColor Yellow
Write-Host "Part A checkpoints comparable to original v3 sweep (datasets=both)." -ForegroundColor DarkGray
Write-Host "Part B checkpoints test isolated Talebi compression from native 1024 (datasets=node21)." -ForegroundColor DarkGray
