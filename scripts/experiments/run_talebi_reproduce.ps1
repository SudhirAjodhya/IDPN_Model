# run_talebi_reproduce.ps1
# Overnight runner for the Talebi reproduction.
# Order: smoke test  ->  seed s in {0,1,2}: Phase A, Phase B  ->  summary + plots.
# Approx 8-14h on RTX 5070 Ti at 60 epochs.

$ErrorActionPreference = "Stop"
Set-Location -Path (Resolve-Path (Join-Path $PSScriptRoot "../.."))

$PYTHON = "torch_env\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) { $PYTHON = "python" }

$env:PYTHONUNBUFFERED = "1"

$SEEDS  = @(0, 1, 2)
$EPOCHS = 60

$LOG_DIR = "outputs\talebi_reproduce\results"
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_DIR "run_$STAMP.log"

function Run($label, $args_) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    "=== $label ===`n" | Add-Content $LOG
    & $PYTHON scripts/train_talebi_reproduce.py @args_ 2>&1 | Tee-Object -FilePath $LOG -Append
    if ($LASTEXITCODE -ne 0) { throw "$label failed (exit $LASTEXITCODE)" }
}

Run "SMOKE TEST" @("--smoke")

foreach ($seed in $SEEDS) {
    Run "Phase A, seed $seed" @("--phase", "A", "--seed", $seed, "--epochs", $EPOCHS)
    Run "Phase B, seed $seed" @("--phase", "B", "--seed", $seed, "--epochs", $EPOCHS)
}

Run "Summary" @("--summarize")

# Generate plots
Write-Host "`nGenerating plots..." -ForegroundColor Cyan
& $PYTHON -c "from src.visualization.classification_curves import generate_all_plots; from pathlib import Path; generate_all_plots(Path('outputs/talebi_reproduce/results'))"

Write-Host "`nAll runs done. Log: $LOG" -ForegroundColor Green


