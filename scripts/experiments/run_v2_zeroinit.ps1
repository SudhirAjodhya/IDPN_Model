# run_v2_zeroinit.ps1
# Tests the hypothesis: paper-faithful Talebi's random-init output_conv adds noise
# on top of bilinear at step 1, so training wastes time cancelling noise before the
# residual can help. Near-zero init (kaiming * 0.01) starts residual ~= 0 -> model
# begins strictly at bilinear and can only improve.
#
# Verified at init: residual magnitude 0.547 (random-init) -> 0.013 (near-zero-init).
#
# 3 runs: node21_talebi_zeroinit at s42, s7, s123. ~3-4h total.
# Compare to bilinear v2 (same seeds) already run in v2_multiseed:
#   bilinear s42 = 0.542    s7 = 0.610    s123 = 0.616
#
# Verdict: if zeroinit Talebi beats bilinear 3/3, foundation is validated and
# we can proceed to attention-block design. If mean Delta stays ~0 like paper-faithful,
# init is not the deciding factor -- something else about the architecture or task fit
# is capping Talebi's gains on CXR nodule detection.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$RUN_GROUP = "v2_zeroinit"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

$SEEDS = @(42, 7, 123)

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
foreach ($seed in $SEEDS) {
    $label = "convnext_fpn / node21_talebi_zeroinit v2_zeroinit s$seed batch=8 lr=2e-4 val_frac=0.20 no-cosine"
    Run $label @(
        "--backbone",    "convnext_fpn",
        "--variant",     "node21_talebi_zeroinit",
        "--seed",        "$seed",
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

Write-Host "`nv2 zeroinit complete. Log: $LOG" -ForegroundColor Green
Write-Host "Compare to bilinear v2_multiseed:" -ForegroundColor Yellow
Write-Host "  s42=0.542  s7=0.610  s123=0.616  (combined test sens)" -ForegroundColor DarkGray


