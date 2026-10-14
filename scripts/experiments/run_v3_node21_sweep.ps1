# run_v3_node21_sweep.ps1
# NODE21-only mirror of the original v3 resolution sweep + completion of the
# native-1024 bilinear controls that were left unfinished.
#
# Motivation: the earlier v3 sweep used datasets="both", which trained on JSRT
# alongside NODE21. JSRT samples buff NODE21 test sensitivity in ways that
# confound the pure-NODE21 story. This sweep re-runs the same protocol with
# datasets="node21" only.
#
# Also completes the two bilinear-at-input=1024 configs (target=384, target=512)
# that weren't run / were interrupted during the native_1024 sweep, giving us
# clean paired controls for the Talebi native-1024 numbers.
#
# 10 runs total, s42, ~9h. Fits overnight.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$RUN_GROUP = "v3_node21_sweep"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

$SEED = 42

# Arm A -- bilinear at input=target=X (mirror of v3 Arm A, datasets=node21).
# Skip X=1024 -- too expensive and we already have that config from v3.
# Batch/accum schedule matches earlier sweep to keep effective batch=8.
# Arm B -- Talebi zeroinit at input=X, target=224 (mirror of v3 Arm B, datasets=node21).
#   Backbone always at 224 -> Talebi runs are cheap regardless of input res.
# Arm C -- bilinear at input=1024 target=384/512 (complete missing native-1024 controls).
$RUNS = @(
    # Arm A -- bilinear input=target=X
    @{ variant = "bilinear";               input = 224;  target = 224; batch = 8; accum = 1 },
    @{ variant = "bilinear";               input = 256;  target = 256; batch = 8; accum = 1 },
    @{ variant = "bilinear";               input = 384;  target = 384; batch = 8; accum = 1 },
    @{ variant = "bilinear";               input = 512;  target = 512; batch = 4; accum = 2 },
    # Arm B -- Talebi zeroinit input=X target=224
    @{ variant = "node21_talebi_zeroinit"; input = 224;  target = 224; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit"; input = 256;  target = 224; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit"; input = 384;  target = 224; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit"; input = 512;  target = 224; batch = 8; accum = 1 },
    # Arm C -- complete native-1024 bilinear controls
    @{ variant = "bilinear";               input = 1024; target = 384; batch = 8; accum = 1 },
    @{ variant = "bilinear";               input = 1024; target = 512; batch = 4; accum = 2 }
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
    $label = "convnext_fpn / $($cfg.variant) input=$($cfg.input) target=$($cfg.target) batch=$($cfg.batch) accum=$($cfg.accum) datasets=node21 s$SEED"
    Run $label @(
        "--backbone",    "convnext_fpn",
        "--variant",     $cfg.variant,
        "--seed",        "$SEED",
        "--datasets",    "node21",
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

Write-Host "`nv3 node21-only sweep complete. Log: $LOG" -ForegroundColor Green
Write-Host "After it lands, combined with v3 recovery + v3 native_1024 we have:" -ForegroundColor Yellow
Write-Host "  * bilinear input=target=X, node21 (X in {224,256,384,512})           - Arm A" -ForegroundColor DarkGray
Write-Host "  * Talebi input=X target=224, node21 (X in {224,256,384,512,1024*})    - Arm B (* 1024 from native_1024)" -ForegroundColor DarkGray
Write-Host "  * bilinear + Talebi input=1024 target={224,256,384,512}, node21       - native downsampling test" -ForegroundColor DarkGray
