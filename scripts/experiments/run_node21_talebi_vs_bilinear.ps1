# run_node21_talebi_vs_bilinear.ps1
# Node21-only head-to-head: bilinear vs node21_talebi (paper-faithful TalebiPaperResizer).
# One seed (42). ConvNeXt-T backbone, target 512 (matches matrix_v1 defaults).
# Approx ~4h per run on RTX 5070 Ti -> ~8h total.
# Follow with scripts/eval_size_stratified.py per checkpoint for Plot B numbers.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$VARIANTS    = @("bilinear", "node21_talebi")
$SEED        = 42
$RUN_GROUP   = "node21_only_v1"
$EPOCHS      = 50
$PATIENCE    = 15
$TARGET_SIZE = 224   # paper Talebi standard. Set to 256 for the alt run if desired.
$INPUT_SIZE  = 1024  # Node21 native res — no upsample, ~4x less loader work than the legacy 2048.

$LOG_ROOT = Join-Path "outputs" "node21_only_v1"
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
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
}

# Build the Node21 .npy cache once — idempotent, skips already-cached files.
# Loader auto-detects the sibling _npy dir. ~5-10x IO speedup vs SimpleITK per __getitem__.
$N21_CSV    = "data_sets/node21/cxr_images/original_data/metadata.csv"
$N21_IMAGES = "data_sets/node21/cxr_images/original_data/images"
Write-Host "Ensuring Node21 npy cache..." -ForegroundColor Yellow
& python scripts/data/cache_node21_npy.py --node21-csv $N21_CSV --node21-images $N21_IMAGES --workers 6
if ($LASTEXITCODE -ne 0) { throw "Node21 npy cache build failed." }

# Smoke test — fails fast if the pipeline is broken.
Write-Host "Running smoke test..." -ForegroundColor Yellow
& python scripts/diagnostics/smoke_test_node21_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed. Aborting sweep." }

foreach ($variant in $VARIANTS) {
    Run "$variant s$SEED" @(
        "--variant",     $variant,
        "--datasets",    "node21",
        "--seed",        "$SEED",
        "--epochs",      "$EPOCHS",
        "--patience",    "$PATIENCE",
        "--target-size", "$TARGET_SIZE",
        "--input-size",  "$INPUT_SIZE",
        "--batch-size",  "2",
        "--run-group",   $RUN_GROUP,
        "--augment",
        "--froc"
    )
}

Write-Host "Sweep complete. Next: run scripts/eval_size_stratified.py per checkpoint." -ForegroundColor Green


