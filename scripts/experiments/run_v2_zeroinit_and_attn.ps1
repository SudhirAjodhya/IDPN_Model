# run_v2_zeroinit_and_attn.ps1
# Overnight sweep: extend the zero-init foundation from 3 -> 5 seeds AND get a
# first read on the attention-subtraction variant at 3 seeds paired with the
# existing bilinear + zeroinit comparison anchors.
#
# 5 runs, ~5h at v2 recipe (batch=8, LR=2e-4, no cosine, val_frac=0.20).
#
# What we get:
#   - zeroinit at 5 seeds total: s42/7/123 (existing v2_zeroinit) + s5/99 (new)
#     -> locks the 2/3-positive foundation claim to 4/5 or 5/5 if consistent
#   - zeroinit_attn at 3 seeds: s42/7/123, directly pairable against existing
#     bilinear and zeroinit at the same seeds -> clean Delta_attn_vs_zeroinit
#
# Ordering: attn runs first so we see the first-look result earliest in the log.
# If attn is a total dud we'll notice in the first 1-2 runs. Zeroinit extensions
# run after (safer bet, less need to check partway).

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "v2_zeroinit_and_attn"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

$RUNS = @(
    @{ variant = "node21_talebi_zeroinit_attn"; seed = 42  },
    @{ variant = "node21_talebi_zeroinit_attn"; seed = 7   },
    @{ variant = "node21_talebi_zeroinit_attn"; seed = 123 },
    @{ variant = "node21_talebi_zeroinit";      seed = 5   },
    @{ variant = "node21_talebi_zeroinit";      seed = 99  }
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
    $label = "convnext_fpn / $($cfg.variant) $RUN_GROUP s$($cfg.seed) batch=8 lr=2e-4 val_frac=0.20 no-cosine"
    Run $label @(
        "--backbone",    "convnext_fpn",
        "--variant",     $cfg.variant,
        "--seed",        "$($cfg.seed)",
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

Write-Host "`nv2 zeroinit + attn sweep complete. Log: $LOG" -ForegroundColor Green
Write-Host "Reference numbers to compare against (combined-test sens):" -ForegroundColor Yellow
Write-Host "  bilinear v2:         s42=0.542  s7=0.610  s123=0.616" -ForegroundColor DarkGray
Write-Host "  zeroinit v2:         s42=0.627  s7=0.655  s123=0.605  mean=0.629 (+0.040 vs bilinear)" -ForegroundColor DarkGray
Write-Host "New results feed into: 5-seed zeroinit picture + 3-seed attn picture." -ForegroundColor DarkGray


