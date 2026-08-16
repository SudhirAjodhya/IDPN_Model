# run_frozen_backbone_control.ps1
# Diagnostic: freeze the pretrained ConvNeXt-T backbone (backbone_lr_factor 0.0)
# and train ONLY the resizer + head. Isolates "does the resizer contribute
# detection signal" from "does the backbone adapt to accommodate the resizer."
#
# Hypothesis under test: pretrained ConvNeXt-T was trained with bilinear
# resizing on ImageNet, so a trainable backbone can adapt-around whatever the
# resizer outputs. Freezing it prevents that adaptation. If Talebi wins with
# the frozen backbone but ties/loses with a trainable one, the confound is
# backbone adaptation eating Talebi's contribution.
#
# Also finishes the missing talebi s123 from ablation_v2 first so we have
# clean n=3 for the primary comparison alongside these frozen-backbone results.
#
# Order:
#   1. talebi s123 (trainable backbone) — closes ablation_v2 n=3
#   2. frozen bilinear s42            (baseline for the frozen-backbone regime)
#   3. frozen node21_talebi s42       (paper-faithful under frozen backbone)
#   4. frozen talebi s42              (zero-init under frozen backbone)
#
# ~3h for run 1 + ~3-4h for each frozen run = ~14h total. Fits an overnight.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$LOG_ROOT = Join-Path "outputs" "frozen_backbone_control"
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
        Write-Host "!! $label failed with exit code $LASTEXITCODE — continuing to next" -ForegroundColor Red
        "!! $label failed with exit code $LASTEXITCODE — continuing to next" | Add-Content $LOG
    }
}

Write-Host "Running smoke test..." -ForegroundColor Yellow
& python scripts/diagnostics/smoke_test_node21_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed. Aborting sweep." }

$RUN_START = Get-Date

# 1. Fill the missing talebi s123 in ablation_v2 (trainable backbone, patience 15, 50 epochs).
Run "talebi  datasets=both  s123  (fill ablation_v2)" @(
    "--variant",     "talebi",
    "--datasets",    "both",
    "--seed",        "123",
    "--target-size", "224",
    "--input-size",  "2048",
    "--batch-size",  "2",
    "--epochs",      "50",
    "--patience",    "15",
    "--run-group",   "ablation_v2",
    "--augment",
    "--froc"
)
$elapsed = (Get-Date) - $RUN_START
Write-Host "[timing] elapsed: $($elapsed.ToString('hh\:mm'))" -ForegroundColor DarkGray

# 2-4. Frozen-backbone control at seed 42.
# Shorter runs: 30 epochs + patience 15. Frozen-backbone converges faster
# because only the resizer + head learn (much fewer effective parameters).
$FROZEN_RUNS = @(
    @{ variant = "bilinear"      },
    @{ variant = "node21_talebi" },
    @{ variant = "talebi"        }
)
foreach ($cfg in $FROZEN_RUNS) {
    $label = "$($cfg.variant)  datasets=both  s42  frozen-backbone"
    Run $label @(
        "--variant",             $cfg.variant,
        "--datasets",            "both",
        "--seed",                "42",
        "--target-size",         "224",
        "--input-size",          "2048",
        "--batch-size",          "2",
        "--epochs",              "30",
        "--patience",            "15",
        "--backbone-lr-factor",  "0.0",
        "--run-group",           "frozen_backbone",
        "--augment",
        "--froc"
    )
    $elapsed = (Get-Date) - $RUN_START
    Write-Host "[timing] elapsed: $($elapsed.ToString('hh\:mm'))" -ForegroundColor DarkGray
}

Write-Host "" -ForegroundColor Green
Write-Host "Sweep complete." -ForegroundColor Green
Write-Host "Analysis targets:" -ForegroundColor Green
Write-Host "  1. talebi s123 fills ablation_v2 -> compute paired-seed Delta across s42,s7,s123" -ForegroundColor Green
Write-Host "  2. frozen-backbone bilinear vs Talebi -> does resizer contribute when backbone can't adapt?" -ForegroundColor Green


