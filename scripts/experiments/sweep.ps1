<#
.SYNOPSIS
    Batch runner for the refactored Nodule Detection pipeline.
    Runs all variants x seeds sequentially, then generates run_summary.json.

.PARAMETER RunGroup
    Keyword for the outer output folder and MLflow run name prefix.
    e.g. "salience-probes", "saliency-ablation"

.PARAMETER Variants
    Comma-separated list of variants to run.
    Available variants: bilinear, talebi, saliency_talebi, inverse_bone_talebi, talebi_aux, alpha, gamma, delta, lip, blurpool

.PARAMETER Heads
    Comma-separated list of detection heads. Default: centernet

.PARAMETER Seeds
    Comma-separated list of seeds. Default: 42

.PARAMETER Losses
    Comma-separated list of loss functions. Default: focal_v5

.PARAMETER Epochs
    Number of training epochs per run. Default: 50

.PARAMETER TestSeed
    Fixed seed for test split (never change). Default: 0

.PARAMETER CosineLR
    Switch to enable cosine LR schedule.

.PARAMETER GradAccum
    Gradient accumulation steps. Default: 1

.PARAMETER AuxLambda
    Auxiliary loss weight for talebi_aux variant. Default: 0.1

.EXAMPLE
    # Overnight salience-probe batch (4 runs x ~90 min)
    .\sweep.ps1 -RunGroup "salience-probes" `
                 -Variants "bilinear,inverse_bone_talebi,talebi_aux,saliency_talebi" `
                 -Heads "centernet" `
                 -Seeds "42" `
                 -Losses "focal_v5" `
                 -Epochs 50 `
                 -CosineLR
#>
param(
    [string]$RunGroup   = "default",
    [string]$Variants   = "bilinear",
    [string]$Heads      = "centernet",
    [string]$Seeds      = "42",
    [string]$Losses     = "focal_v5",
    [int]$Epochs        = 50,
    [int]$TestSeed      = 0,
    [switch]$CosineLR,
    [int]$GradAccum     = 1,
    [float]$AuxLambda   = 0.1
)

$ROOT_DIR = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path; Set-Location $ROOT_DIR; $PYTHON = Join-Path $ROOT_DIR "torch_env\Scripts\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Host "WARNING: torch_env python not found at $PYTHON, falling back to global python" -ForegroundColor Yellow
    $PYTHON = "python"
}

$variantList = $Variants -split ","
$headList    = $Heads    -split ","
$seedList    = $Seeds    -split ","
$lossList    = $Losses   -split ","
$totalRuns   = $variantList.Count * $headList.Count * $seedList.Count * $lossList.Count
$runIdx      = 0
$failed      = @()

$sep = "=" * 70
Write-Host $sep
Write-Host "Experiment batch runner"
Write-Host "Run group : $RunGroup"
Write-Host "Variants  : $($variantList -join ', ')"
Write-Host "Heads     : $($headList -join ', ')"
Write-Host "Seeds     : $($seedList -join ', ')"
Write-Host "Losses    : $($lossList -join ', ')"
Write-Host "Epochs    : $Epochs"
Write-Host "TestSeed  : $TestSeed"
Write-Host "CosineLR  : $CosineLR"
Write-Host "GradAccum : $GradAccum"
Write-Host "AuxLambda : $AuxLambda"
Write-Host "Total runs: $totalRuns"
Write-Host $sep

$startTime = Get-Date

foreach ($loss in $lossList) {
    foreach ($head in $headList) {
        foreach ($v in $variantList) {
            foreach ($s in $seedList) {
                $runIdx++
                $elapsed = [int]((Get-Date) - $startTime).TotalMinutes
                Write-Host ""
                Write-Host "--- Run $runIdx / $totalRuns : variant=$v head=$head loss=$loss seed=$s [${elapsed}m elapsed] ---"

                $pyArgs = @(
                    "scripts/train.py",
                    "--variant",    $v,
                    "--head",       $head,
                    "--seed",       $s,
                    "--run-group",  $RunGroup,
                    "--epochs",     $Epochs,
                    "--loss",       $loss,
                    "--test-seed",  $TestSeed,
                    "--grad-accum", $GradAccum,
                    "--aux-lambda", $AuxLambda
                )
                if ($CosineLR) { $pyArgs += "--cosine-lr" }

                & $PYTHON @pyArgs

                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[FAILED] variant=$v head=$head loss=$loss seed=$s exited with code $LASTEXITCODE" -ForegroundColor Red
                    $failed += "${v}_${head}_${loss}_s${s}"
                } else {
                    Write-Host "[OK] variant=$v head=$head loss=$loss seed=$s" -ForegroundColor Green
                }
            }
        }
    }
}

$totalElapsed = [int]((Get-Date) - $startTime).TotalMinutes
Write-Host ""
Write-Host $sep
Write-Host "All runs complete in ${totalElapsed}m."

if ($failed.Count -gt 0) {
    Write-Host "Failed runs: $($failed -join ', ')" -ForegroundColor Red
}

$outputBase = Join-Path $ROOT_DIR "outputs"
$outputBase = Join-Path $outputBase "v8"

$runGroupDirs = Get-ChildItem -Path $outputBase -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "_${RunGroup}$" }

if (-not $runGroupDirs -or $runGroupDirs.Count -eq 0) {
    Write-Host "No output directory found matching *_${RunGroup} in $outputBase" -ForegroundColor Yellow
} else {
    $runGroupDir = ($runGroupDirs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    Write-Host "Generating run summary for: $runGroupDir"
    & $PYTHON -c @"
import sys; sys.path.insert(0, r'$ROOT_DIR')
from src.utils.run_summary import generate_run_summary
generate_run_summary(r'$runGroupDir')
"@
}

Write-Host $sep
Write-Host "To view MLflow results:"
Write-Host "  cd $ROOT_DIR"
Write-Host "  $PYTHON -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlruns.db"
Write-Host "  # Open http://localhost:5000 and filter by experiment nodule_v8"


