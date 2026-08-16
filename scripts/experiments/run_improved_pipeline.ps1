# run_improved_pipeline.ps1
# Adds four cumulative improvements to the pipeline:
#   1. CLAHE preprocessing on both loaders (always on, not augmentation)
#   2. Dropout(0.3) in the CenterNet head
#   3. Cosine LR schedule
#   4. Target resolution 384 (heatmap 96x96 = 3x more positive pixels
#      per nodule vs 224/hm=56, closer to Node21 leaderboard practice)
#
# Same recipe otherwise: datasets=both, input=2048, batch=2, patience=15,
# --augment, --froc, weight_decay 0.05.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "improved_v1"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

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
$VARIANTS = @("bilinear", "node21_talebi", "talebi")

foreach ($v in $VARIANTS) {
    $label = "$v datasets=both s42 improved"
    Run $label @(
        "--variant",     $v,
        "--datasets",    "both",
        "--seed",        "42",
        "--target-size", "384",
        "--input-size",  "2048",
        "--batch-size",  "2",
        "--epochs",      "50",
        "--patience",    "15",
        "--run-group",   $RUN_GROUP,
        "--augment",
        "--cosine-lr",
        "--froc"
    )
    $elapsed = (Get-Date) - $RUN_START
    Write-Host "[timing] elapsed: $($elapsed.ToString('hh\:mm'))" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Sweep complete." -ForegroundColor Green
Write-Host "Baseline NODE21 test sens was 0.44 to 0.52. Meaningful lift = 0.55+." -ForegroundColor Green


