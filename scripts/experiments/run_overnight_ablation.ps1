# run_overnight_ablation.ps1
# Answers two things:
#   1. Does datasets=both reproduce matrix_v1's ~0.46 Node21 test sens?
#      (Confirms the "JSRT augmentation is the whole gap" hypothesis)
#   2. Does Talebi beat bilinear cleanly on BOTH data regimes?
#      (The actual thesis claim - replicated across regimes tightens the story)
#
# All runs: seed 42, target=224, input=2048, batch=2, patience=15, --froc.
# ~5-6h per run at input=2048, batch=2 on RTX 5070 Ti. Total ~15-18h.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "ablation_v1"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

# (variant, datasets) tuples - order matters if you want partial results early.
$RUNS = @(
    @{ variant = "bilinear";       datasets = "both"   },   # 1. matrix_v1 replication
    @{ variant = "node21_talebi";  datasets = "both"   },   # 2. Talebi on Combined
    @{ variant = "node21_talebi";  datasets = "node21" }    # 3. Talebi on Node21-only (pairs with existing bilinear Node21-only run)
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

foreach ($cfg in $RUNS) {
    $label = "$($cfg.variant)  datasets=$($cfg.datasets)  s42"
    Run $label @(
        "--variant",     $cfg.variant,
        "--datasets",    $cfg.datasets,
        "--seed",        "42",
        "--target-size", "224",
        "--input-size",  "2048",
        "--batch-size",  "2",
        "--epochs",      "50",
        "--patience",    "15",
        "--run-group",   $RUN_GROUP,
        "--augment",
        "--froc"
    )
}

Write-Host "Ablation complete. Compare NODE21 test sens across all runs to check matrix_v1 replication + Talebi delta." -ForegroundColor Green


