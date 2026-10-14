# run_v3_native_1024.ps1
# Native-input sweep. Talebi loaded at NODE21's native 1024 (no bilinear pre-
# downsampling by cv2.resize) then LDM does the entire compression 1024 -> T.
# Isolates Talebi's downsampling capability, unconfounded by pre-model bilinear
# info loss.
#
# datasets="node21" because JSRT native is 2048 -> would still get pre-downsampled
# by cv2.resize(2048->1024) at input=1024.
#
# 8 runs (4 Talebi + 4 bilinear controls at matched (input=1024, target=T)) so
# we can compute Talebi-vs-bilinear delta at each target for isolated LDM comparison.
#
# ~5h. Backbone at 224-512 features (cheap); LDM at 1024 features (dominant cost).

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$RUN_GROUP = "v3_native_1024"
$LOG_ROOT = Join-Path "outputs" $RUN_GROUP
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

$SEED = 42

# Talebi-only. Bilinear controls skipped -- existing v3 bilinear runs (input=target=T)
# use cv2.resize(1024->T) which is algorithmically equivalent to F.interpolate(1024->T).
# The <0.005 numerical difference between cv2 and PyTorch bilinear isn't worth 4 extra
# runs. Fair Δ comparison uses: v3 recovery bilinear NODE21 sens at each target.
$RUNS = @(
    @{ variant = "node21_talebi_zeroinit";  target = 224; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit";  target = 256; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit";  target = 384; batch = 8; accum = 1 },
    @{ variant = "node21_talebi_zeroinit";  target = 512; batch = 4; accum = 2 }
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
    $label = "convnext_fpn / $($cfg.variant) input=1024 target=$($cfg.target) batch=$($cfg.batch) accum=$($cfg.accum) datasets=node21 s$SEED"
    Run $label @(
        "--backbone",    "convnext_fpn",
        "--variant",     $cfg.variant,
        "--seed",        "$SEED",
        "--datasets",    "node21",
        "--target-size", "$($cfg.target)",
        "--input-size",  "1024",
        "--batch-size",  "$($cfg.batch)",
        "--grad-accum",  "$($cfg.accum)",
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

Write-Host "`nNative-1024 sweep complete. Log: $LOG" -ForegroundColor Green
Write-Host "New comparison: paired (bilinear vs zeroinit) at input=1024 target={224,256,384,512}." -ForegroundColor Yellow
