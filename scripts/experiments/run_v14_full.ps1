# run_v14_full.ps1
# Warm the VinDr caches, then run the v14 sweep. One command, correct order.
#
# WHY THIS WRAPPER EXISTS
#   The first attempt at v14 ran uncached and was dataloader-bound: a VinDr film cost
#   ~370 ms (DICOM/PNG decode 47%, CLAHE at native resolution 33%, normalisation 14%)
#   against NODE21's 2.7 ms, so ~15.4 of every 19.2 min per epoch was spent loading and
#   the GPU sat idle roughly 80% of the time. Warming the cache first takes a film to
#   ~14 ms. Forgetting the warm step silently reverts that, so it is done here.
#
#   The caches are keyed on input_size because the cached array is already resized.
#   Arm V needs input 1024; Arm N needs 2048. Both must exist before the sweep starts.
#
# DISK
#   1024 cache ~18 GB, 2048 cache ~74 GB. The dataset refuses to cache when the projected
#   size does not fit and falls back to uncached loading, so a short drive degrades speed
#   rather than crashing. This script checks up front so that is a decision, not a surprise.
#
#   The intermediate PNG16 archive is NOT needed - these caches supersede it, and films are
#   read straight from the original DICOM. Do NOT delete train/ or train_png16_manifest.csv:
#   they are the only regeneration path.
#
# USAGE
#   .\scripts\experiments\run_v14_full.ps1 -DryRun
#   .\scripts\experiments\run_v14_full.ps1
#   .\scripts\experiments\run_v14_full.ps1 -Arms V        # skip the expensive Arm N
#   .\scripts\experiments\run_v14_full.ps1 -SkipWarm      # caches already built

param(
    [string] $Arms               = "VNC",
    [int[]]  $Seeds              = @(42),
    [int]    $InputSize          = 1024,
    [int]    $NativeVindrSize    = 2048,
    [double] $VindrNegativeRatio = 1.0,
    [int]    $Workers            = 12,
    [switch] $SkipWarm,
    [switch] $DryRun,
    [switch] $Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\.."
Set-Location -Path $repoRoot

$VINDR_ROOT  = "D:/Downloads/vinbigdata-chest-xray-abnormalities-detection"
$PYTHON      = ".\torch_env\Scripts\python.exe"
$invariant   = [cultureinfo]::InvariantCulture
$ratioText   = $VindrNegativeRatio.ToString($invariant)

if (-not (Test-Path (Join-Path $VINDR_ROOT "train"))) {
    throw "VinDr DICOM source missing at $VINDR_ROOT/train. Nothing can be rebuilt without it."
}

# Which input sizes actually need a cache, given the arms requested.
$neededSizes = @()
if ($Arms -match "V") { $neededSizes += $InputSize }
if ($Arms -match "N") { $neededSizes += $NativeVindrSize }
$neededSizes = $neededSizes | Select-Object -Unique

# Expected film count after subsampling: 4394 box-bearing films plus the same number of
# box-free ones. Every one of the 18,231 boxes lives on a positive film, so the negatives
# dropped here cost no supervision at all.
$expectedFilms = if ($VindrNegativeRatio -gt 0) { 8788 } else { 15000 }

Write-Host ""
Write-Host "v14 full run | arms=$Arms seeds=$($Seeds -join ',') | vindr negatives ${ratioText}x positives" -ForegroundColor Cyan

foreach ($size in $neededSizes) {
    $cacheDir = Join-Path $VINDR_ROOT "train_png16_cache_i$size"
    $have     = (Get-ChildItem -Path $cacheDir -Filter "*.npy" -ErrorAction SilentlyContinue).Count
    $needGB   = [math]::Round($expectedFilms * $size * $size * 2 / 1GB, 1)
    $freeGB   = [math]::Round((Get-PSDrive D).Free / 1GB, 1)

    if ($have -ge $expectedFilms) {
        Write-Host "  [ok]   cache i$size complete: $have films" -ForegroundColor DarkGray
        continue
    }

    Write-Host "  [warm] cache i$size has $have/$expectedFilms films, needs ~$needGB GB (free $freeGB GB)" -ForegroundColor Yellow

    if ($needGB -gt $freeGB) {
        throw ("Not enough disk for the i$size cache: needs ~$needGB GB, $freeGB GB free. " +
               "Deleting $VINDR_ROOT/test (~31 GB, the unlabelled Kaggle split) would help.")
    }

    if ($SkipWarm) {
        Write-Host "  [warn] -SkipWarm set: the sweep will populate this cache during epoch 1, slowly." -ForegroundColor Red
        continue
    }

    if ($DryRun) {
        Write-Host "  [dry]  would warm cache i$size" -ForegroundColor DarkGray
        continue
    }

    & $PYTHON -m scripts.warm_vindr_cache --input-size $size --negative-ratio $ratioText --workers $Workers

    if ($LASTEXITCODE -ne 0) { throw "Cache warm failed for input size $size." }
}

Write-Host ""

# Splatted as a HASHTABLE, not an array. Array splatting passes elements positionally,
# which silently binds -Arms's value to the first positional parameter (-Seeds) instead.
$sweepArgs = @{
    Arms               = $Arms
    Seeds              = $Seeds
    InputSize          = $InputSize
    NativeVindrSize    = $NativeVindrSize
    VindrNegativeRatio = $VindrNegativeRatio
}
if ($DryRun) { $sweepArgs["DryRun"] = $true }
if ($Resume) { $sweepArgs["Resume"] = $true }

& ".\scripts\experiments\run_v14_vindr_data_volume.ps1" @sweepArgs
