# run_v2_phase1.ps1
# v2 recipe change + full backbone/pretraining matrix, 1 seed, overnight sweep.
#
# Two things get exercised at once:
#   1. New training recipe (batch=8, LR=2e-4, no cosine, val_frac=0.20)
#   2. Backbone/pretraining matrix (ConvNeXt|ResNet50 x pretrained|scratch)
#
# Combined into Phase 1 (was going to phase; user wanted the full matrix on night one).
# If any recipe issue shows up it will show up on convnext_fpn (pretrained) row,
# which is the direct comparison to v1 s42 - use it as the sanity anchor.
#
# 8 runs total (4 backbones x 2 resizers) x seed 42. At ~1.5-2h/run: ~12-16h.
#
# Ordering: pretrained pairs first (most informative + fastest to converge),
# scratch pairs last (may not converge in 50 epochs; expected floor).
#
# Sanity anchors (compare to improved_v1 s42):
#   convnext_fpn bilinear   combined test sens ~= v1 (0.629) +/- 0.05  -> recipe OK
#   convnext_fpn talebi     combined test sens ~= v1 (0.684) +/- 0.05
# Scratch numbers: absolute sens will be much lower (~0.15-0.35), that is expected on 4k train images.
# What matters for scratch: the RELATIVE ordering bilinear vs talebi in each pair.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "v2_phase1"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

# 4 backbones x 2 resizers = 8 runs. Pretrained pairs first.
$RUNS = @(
    @{ backbone = "convnext_fpn";           variant = "bilinear"      },
    @{ backbone = "convnext_fpn";           variant = "node21_talebi" },
    @{ backbone = "resnet50_fpn";           variant = "bilinear"      },
    @{ backbone = "resnet50_fpn";           variant = "node21_talebi" },
    @{ backbone = "convnext_fpn_scratch";   variant = "bilinear"      },
    @{ backbone = "convnext_fpn_scratch";   variant = "node21_talebi" },
    @{ backbone = "resnet50_fpn_scratch";   variant = "bilinear"      },
    @{ backbone = "resnet50_fpn_scratch";   variant = "node21_talebi" }
)
$SEED = 42

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
    $label = "$($cfg.backbone) / $($cfg.variant) v2_phase1 s$SEED batch=8 lr=2e-4 val_frac=0.20 no-cosine"
    Run $label @(
        "--backbone",    $cfg.backbone,
        "--variant",     $cfg.variant,
        "--seed",        "$SEED",
        "--datasets",    "both",
        "--target-size", "384",
        "--input-size",  "2048",
        "--batch-size",  "8",
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

Write-Host "`nv2 phase 1 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Sanity anchor: outputs/v8/*_v2_phase1/{bilinear,node21_talebi}_centernet_focal_v5_s42/metrics.json" -ForegroundColor Yellow
Write-Host "                (bare = convnext_fpn pretrained; _<bb> suffix for others)" -ForegroundColor DarkGray


