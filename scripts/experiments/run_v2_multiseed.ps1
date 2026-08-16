# run_v2_multiseed.ps1
# Load-bearing claim confirmation: does v2-recipe Talebi beat bilinear on ConvNeXt-pretrained
# across seeds? s42 already ran in v2_phase1 (talebi +0.057 combined, +0.082 NODE21). This adds
# s7 and s123 to reach the 3-seed threshold. Everything else (ResNet, scratch, attention) is
# blocked until this foundation is confirmed.
#
# 4 runs: {bilinear, node21_talebi} x {s7, s123}, all ConvNeXt-pretrained. ~1h/run at v2.
# Ordering: pairs interleaved by seed so a partial-completion result still has one full pair.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "v2_multiseed"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

$RUNS = @(
    @{ variant = "bilinear";      seed = 7   },
    @{ variant = "node21_talebi"; seed = 7   },
    @{ variant = "bilinear";      seed = 123 },
    @{ variant = "node21_talebi"; seed = 123 }
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
    $label = "convnext_fpn / $($cfg.variant) v2_multiseed s$($cfg.seed) batch=8 lr=2e-4 val_frac=0.20 no-cosine"
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

Write-Host "`nv2 multiseed complete. Log: $LOG" -ForegroundColor Green
Write-Host "Verdict inputs -- combined-test sens, seed 42 (v2_phase1):" -ForegroundColor Yellow
Write-Host "  bilinear s42 = 0.542   |   node21_talebi s42 = 0.599   (Delta +0.057)" -ForegroundColor DarkGray
Write-Host "New results at s7 + s123 pair with those to give the 3-seed picture." -ForegroundColor DarkGray


