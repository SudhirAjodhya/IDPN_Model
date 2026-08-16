# run_improved_multiseed.ps1
# Multi-seed replication of the 2026-08-13 improved_v1 result.
# improved_v1 at s42 showed node21_talebi cleanly beat bilinear +0.055 on NODE21 test.
# This runs s7 + s123 for all three variants to confirm the delta at n=3.
#
# All runs: datasets=both, input=2048, target=384, batch=2, patience=15,
# --augment, --cosine-lr, --froc. Same as improved_v1.
#
# Order chosen so the most-informative row lands earliest:
#   1. node21_talebi s7      (paper-faithful, tests if the win survives)
#   2. bilinear s7           (baseline pair for row 1)
#   3. node21_talebi s123    (paper-faithful, second seed)
#   4. bilinear s123         (baseline pair for row 3)
#   5. talebi s7             (zero-init variant, less priority)
#   6. talebi s123           (zero-init variant, less priority)
#
# ~2.5-3h per run at target=384 batch=2 on RTX 5070 Ti. Total ~15-18h.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "improved_v1"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_multiseed_$STAMP.log"

$RUNS = @(
    @{ variant = "node21_talebi"; seed = 7   },
    @{ variant = "bilinear";      seed = 7   },
    @{ variant = "node21_talebi"; seed = 123 },
    @{ variant = "bilinear";      seed = 123 },
    @{ variant = "talebi";        seed = 7   },
    @{ variant = "talebi";        seed = 123 }
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
    $label = "$($cfg.variant) datasets=both s$($cfg.seed) improved"
    Run $label @(
        "--variant",     $cfg.variant,
        "--datasets",    "both",
        "--seed",        "$($cfg.seed)",
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
Write-Host "Multi-seed complete." -ForegroundColor Green
Write-Host "Analyse: bilinear vs node21_talebi across s42/s7/s123 with paired-seed delta." -ForegroundColor Green
Write-Host "If mean delta positive AND consistent across seeds, +0.055 headline holds." -ForegroundColor Green


