# run_v19_focal_norm_fix.ps1
# Re-run the top of the resolution ladder under the corrected focal normaliser.
#
# WHAT WAS WRONG
#   src/losses/fcos.py normalised the focal term by the positive-cell count, floored at 1:
#
#       return loss.sum() / target.sum().clamp(min=1)
#
#   That is the reference RetinaNet/FCOS convention and it assumes every training image
#   contains an object. COCO satisfies that; this pipeline does not. With
#   vindr_negative_ratio 1.0 and batch_size 4, a large share of batches are entirely
#   nodule-free, and on those the numerator sums over every cell while the denominator
#   stays pinned at 1. So their gradient scaled with GRID AREA while nodule-bearing
#   batches stayed correctly normalised.
#
#   MEASURED at identical per-cell error, batch of 2 clean films:
#
#       target   grid       clean-batch loss   vs t224   batch-with-nodule
#       224      56x56          8.48            1.0x          1.35
#       256      64x64         11.08            1.3x          1.10
#       384      96x96         24.93            2.9x          1.10
#       512      128x128       44.32            5.2x          1.10
#       768      192x192       99.73           11.8x          1.10
#       1024     256x256      177.30           20.9x          1.20
#
#   Nodule-free batches suppress detections, so this worked AGAINST resolution the whole
#   way up the ladder, 21x harder at 1024 than at 224. At 1024 it was strong enough that
#   the bilinear arm never trained: val_loss ran 70-325 across 19 epochs (the magnitudes
#   the table predicts for clean batches) against the learned arm's 0.5-2.0 on the same
#   setting, and early stopping fired with the best checkpoint at epoch 4.
#
#   That is what produced the +0.237 at target 1024 - the headline number in v18. It most
#   likely measures which arm escapes a mis-scaled loss, not which resizer preserves
#   information. The learned arm has a full-learning-rate module sitting directly on the
#   input while the ImageNet backbone crawls at backbone_lr_factor 0.1, so it gets
#   confidently-background by epoch 0 and the clean-batch term collapses. Bilinear has no
#   free parameters ahead of the slow backbone and never escapes.
#
# THE FIX
#   The floor now scales with grid area, anchored at the target-224 grid:
#
#       floor = (grid_side / 56) ** 2
#
#   Exactly 1.0 at target 224, so every run banked at that target is bit-for-bit
#   unaffected. Above it the floor grows with the grid and holds the clean-batch loss
#   constant at 8.48 for every target from 224 to 1024 (asserted in the module's _demo).
#   Nodule-bearing batches are untouched: their positive count clears the floor.
#
# WHAT THIS RUN DECIDES
#   If bilinear at target 1024 recovers to roughly its target 768 level, the cause is
#   confirmed, +0.237 comes out of the results, and the ladder's shape below 768 stands.
#   If it does NOT recover, the loss was not the whole story and the collapse needs a
#   different explanation - in which case do not write up either reading yet.
#
# SCOPE - WHY ONLY 768 AND 1024
#   The defect is a smooth ramp, not a cliff, so 384 (2.9x) and 512 (5.2x) are distorted
#   too and are NOT strictly comparable to these runs. They are a follow-up, not part of
#   this decision. Target 224 needs nothing: the fix is a no-op there.
#
#   Do NOT pool these runs with the v18 group in any paired test. Different loss, so the
#   arms do not share a score scale with anything banked earlier. seed_pooled_nodule_test
#   will refuse the pooling anyway on the run-group guard - let it.
#
# COST
#   12 runs: {bilinear, node21_talebi_zeroinit} x {768, 1024} x seeds {42, 7, 123}.
#   The v18 equivalents ran 19-33 epochs, so budget roughly two days.
#   -Resume skips any cell that already wrote a metrics.json under this run group, so a
#   cancelled sweep can be restarted without repeating completed cells.

param(
    [string]   $Head               = "fcos",
    [int]      $InputSize          = 1024,
    [int]      $VindrInputSize     = 2048,
    [int[]]    $Targets            = @(1024, 768),
    [int[]]    $Seeds              = @(42, 7, 123),
    [string[]] $Variants           = @("bilinear", "node21_talebi_zeroinit"),
    [double]   $VindrNegativeRatio = 1.0,
    [string]   $SelectOn           = "val_sens_matched",
    [int]      $MaxEpochs          = 50,
    [int]      $Patience           = 15,
    [string]   $RunGroup           = "v19_focal_norm_fix",
    [switch]   $DryRun,
    [switch]   $Resume
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$LOG_ROOT = Join-Path "outputs" $RunGroup
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }

$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG   = Join-Path $LOG_ROOT "run_$STAMP.log"

$VINDR_ROOT = "D:/Downloads/vinbigdata-chest-xray-abnormalities-detection"
$invariant  = [cultureinfo]::InvariantCulture
$ratioText  = $VindrNegativeRatio.ToString($invariant)

if (-not (Test-Path (Join-Path $VINDR_ROOT "train"))) {
    throw "VinDr DICOM source missing at $VINDR_ROOT/train."
}

# The whole point of this run group is the corrected normaliser. If the working tree has
# been reverted, every run below would silently reproduce v18 and waste two days.
$lossSource = Get-Content "src/losses/fcos.py" -Raw

if ($lossSource -notmatch "REFERENCE_GRID_SIDE") {
    throw "src/losses/fcos.py does not carry the normaliser fix. Aborting - these runs would reproduce v18."
}

# Micro-batch keyed on the resizer's INPUT resolution, matching v15/v18 exactly so these
# runs stay comparable to the ones they replace on every axis except the loss.
function Get-Schedule {
    param([int] $ResizerInput)

    if ($ResizerInput -ge 2560) { return @{ batch = 2; accum = 4 } }
    if ($ResizerInput -ge 2048) { return @{ batch = 4; accum = 2 } }

    return @{ batch = 8; accum = 1 }
}

$resizerInput = if ($VindrInputSize -gt $InputSize) { $VindrInputSize } else { $InputSize }
$schedule     = Get-Schedule -ResizerInput $resizerInput

# Targets descend so the cell that decides the question runs first
$RUNS = @()

foreach ($target in $Targets) {
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            $RUNS += @{
                variant = $variant
                seed    = $seed
                target  = $target
                batch   = $schedule.batch
                accum   = $schedule.accum
            }
        }
    }
}

Write-Host ""
Write-Host "v19 focal normaliser fix -- $($RUNS.Count) runs" -ForegroundColor Cyan
Write-Host "  targets   $($Targets -join ', ')" -ForegroundColor DarkGray
Write-Host "  seeds     $($Seeds -join ', ')" -ForegroundColor DarkGray
Write-Host "  variants  $($Variants -join ', ')" -ForegroundColor DarkGray
Write-Host "  node21 @ $InputSize px, vindr @ $VindrInputSize px, batch $($schedule.batch) x accum $($schedule.accum)" -ForegroundColor DarkGray
Write-Host ""

if ($DryRun) {
    foreach ($cfg in $RUNS) {
        Write-Host ("  {0,-24} s{1,-4} t{2,-5} node21@{3} vindr@{4}" -f `
                    $cfg.variant, $cfg.seed, $cfg.target, $InputSize, $VindrInputSize)
    }

    Write-Host ""
    Write-Host "Dry run -- nothing executed." -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    param($Config)

    # Must mirror train.py's _res_tag + _recipe_tag exactly or -Resume accepts the wrong run
    $resTag    = "_i${InputSize}_t$($Config.target)"
    $recipeTag = "_vindr"

    if ($VindrInputSize -ne $InputSize) { $recipeTag += "_vin$VindrInputSize" }
    if ($VindrNegativeRatio -gt 0)      { $recipeTag += "_vnr$ratioText" }

    $leaf = "$($Config.variant)_${Head}_focal_v5_s$($Config.seed)${resTag}${recipeTag}"

    return Get-ChildItem -Path "outputs" -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq $leaf -and
            $_.Parent.Name -like "*_$RunGroup" -and
            (Test-Path (Join-Path $_.FullName "metrics.json"))
        } |
        Select-Object -First 1
}

function Invoke-Run {
    param([string] $Label, $TrainArgs)

    Write-Host "=== $Label ===" -ForegroundColor Cyan
    "=== $Label ===" | Add-Content $LOG

    # A crashed run leaves orphaned DataLoader workers holding ~1.5 GB of commit each.
    # Snapshot so anything this run leaks can be reaped precisely.
    $preExisting = @(Get-Process python3.13 -ErrorAction SilentlyContinue |
                     Select-Object -ExpandProperty Id)

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    & python -u scripts/train.py @TrainArgs 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LOG -Append

    $ErrorActionPreference = $prev

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$Label failed with exit code $LASTEXITCODE, continuing" -ForegroundColor Red
        "$Label FAILED exit=$LASTEXITCODE" | Add-Content $LOG
    }

    $leaked = @(Get-Process python3.13 -ErrorAction SilentlyContinue |
                Where-Object { $preExisting -notcontains $_.Id })

    if ($leaked.Count -gt 0) {
        $leakedGB = ($leaked | Measure-Object PagedMemorySize64 -Sum).Sum / 1GB
        Write-Host ("  [cleanup] reaping {0} orphaned workers holding {1:N1} GB" -f `
                    $leaked.Count, $leakedGB) -ForegroundColor Yellow
        $leaked | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    }
}

& python scripts/diagnostics/smoke_test_node21_pipeline.py

if ($LASTEXITCODE -ne 0) { throw "Smoke test failed -- not starting a two-day sweep." }

$RUN_START = Get-Date
$completed = 0

foreach ($cfg in $RUNS) {
    $completed++
    $label = "$($cfg.variant) s$($cfg.seed) t$($cfg.target) ($completed/$($RUNS.Count))"

    if ($Resume) {
        $existing = Get-ExistingRun -Config $cfg

        if ($existing) {
            Write-Host "SKIP $label -- already complete at $($existing.FullName)" -ForegroundColor DarkYellow
            "SKIP $label" | Add-Content $LOG
            continue
        }
    }

    $trainArgs = @(
        "--backbone",             "convnext_fpn",
        "--variant",              $cfg.variant,
        "--head",                 $Head,
        "--seed",                 "$($cfg.seed)",
        "--datasets",             "node21_vindr",
        "--input-size",           "$InputSize",
        "--target-size",          "$($cfg.target)",
        "--vindr-input-size",     "$VindrInputSize",
        "--vindr-negative-ratio", $ratioText,
        "--batch-size",           "$($cfg.batch)",
        "--grad-accum",           "$($cfg.accum)",
        "--lr",                   "2e-4",
        "--epochs",               "$MaxEpochs",
        "--patience",             "$Patience",
        "--select-on",            $SelectOn,
        "--aug-profile",          "v3",
        "--val-frac",             "0.20",
        "--run-group",            $RunGroup,
        "--augment",
        "--froc",
        "--attribution"
    )

    Invoke-Run -Label $label -TrainArgs $trainArgs

    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:N1} h" -f `
                $completed, $RUNS.Count, $elapsed.TotalHours) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v19 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Read in this order:" -ForegroundColor Yellow
Write-Host "  1. bilinear t1024 val_loss -- if it is still 70-325, the loss was not the cause" -ForegroundColor DarkGray
Write-Host "  2. bilinear t1024 sensitivity vs its own t768 -- recovery confirms the diagnosis" -ForegroundColor DarkGray
Write-Host "  3. seed_pooled_nodule_test.py --run-group v19_focal_norm_fix -- NEVER pooled with v18" -ForegroundColor DarkGray
