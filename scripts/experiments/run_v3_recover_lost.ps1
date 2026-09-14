# run_v3_recover_lost.ps1
# Recover the 6 checkpoints lost to the output_dir overwrite bug in the first v3 sweep.
#
# Bug: original launcher wrote every run to {date}_v3_res_sweep/{variant}_..._s{seed}/
# without an input/target-size suffix, so bilinear{224,256,384,512} all overwrote each
# other and only the last (512) survived. Same for zeroinit.
#
# Fix: train.py now appends _i{input}_t{target} when they differ from argparse defaults.
# Re-run the 6 overwritten configs. Numbers themselves were captured in the original
# console log; this run restores the checkpoints for post-hoc size-strat + qualitative
# overlays.
#
# 6 runs, s42, ~4h. Uses same batch/accum schedule as first sweep.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$RUN_GROUP = "v3_res_sweep"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "recover_$STAMP.log"

$SEED = 42

# Only the 6 lost configs. Skip 512 (has surviving checkpoint) and 1024 (also survived).
$RUNS = @(
    @{ variant = "bilinear";                input = 224; target = 224; batch = 8; accum = 1 },
    @{ variant = "bilinear";                input = 256; target = 256; batch = 8; accum = 1 },
    @{ variant = "bilinear";                input = 384; target = 384; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit";  input = 224; target = 224; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit";  input = 256; target = 224; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit";  input = 384; target = 224; batch = 8; accum = 1 }
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

Write-Host "`nv3 recovery complete. Log: $LOG" -ForegroundColor Green
Write-Host "New checkpoint dirs now include _i{input}_t{target} suffix so they don't collide with the surviving 512/1024 runs." -ForegroundColor Yellow
