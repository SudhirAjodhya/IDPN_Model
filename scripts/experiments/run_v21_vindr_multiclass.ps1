# run_v21_vindr_multiclass.ps1
# Resolution sweep on VinDr ALONE, 14 pathology classes, scored as image-level multi-label.
#
# WHAT THIS IS AND IS NOT
#   Every other run group in this project trains on NODE21 (optionally with VinDr boxes
#   added to TRAIN only) and reports nodule detection on a fixed 517-film NODE21 test set.
#   This one is VinDr end to end: its own 60/20/20 split, its own val and test, 14 classes,
#   and no NODE21 anywhere. The two sets of numbers answer different questions and must
#   never be put on the same axis.
#
#   The question here is whether the learned resizer's behaviour across resolution holds up
#   across fourteen pathologies rather than one, and whether it helps or hurts the rare ones.
#
# THE SPLIT
#   60/20/20 by FILM, stratified rarest-class-first because VinDr is multi-label and a film
#   usually carries several findings. Measured test counts:
#
#       Aortic enlargement  613   Nodule/Mass        165
#       Cardiomegaly        460   Infiltration       122
#       Pleural thickening  396   Calcification       90
#       Pulmonary fibrosis  323   ILD                 77
#       Lung Opacity        264   Consolidation       70
#       Other lesion        227   Atelectasis         37  <- thin
#       Pleural effusion    206   Pneumothorax        19  <- thin
#
#   Twelve of fourteen classes clear 70 test films. Atelectasis and Pneumothorax do not and
#   no split ratio fixes it - only 186 and 96 such films exist. They are reported and
#   flagged, never quoted alone.
#
#   AUGMENTATION IS TRAIN-ONLY, which is what lets the split be 60/20/20 rather than
#   70/15/15: fewer unique training films, compensated by augmentation, and the originals
#   freed up go to val and test. VinDrDataset augments inside __getitem__, so augmentation
#   is already stochastic per epoch - a film is a different variant every draw. On top of
#   that, --oversample draws rare-class films more often per epoch, sqrt-scaled, so
#   Pneumothorax lands at about 5.6x rather than the 31.9x full equalisation would need.
#
#   Augmented copies NEVER enter val or test. Four augmented versions of one film are not
#   four patients, so they would inflate the row count without moving the sample size an
#   interval is computed from.
#
# IMAGE SOURCE
#   The PNG16 archive was deleted once the resize caches were written, so
#   train_png16_cache_i<input> IS the image source. It holds 8,788 films: ALL 4,394
#   positives plus 4,394 negatives. Every class therefore keeps its full count and only
#   the negative pool is smaller than the CSV's 15,000 would allow.
#
# METRICS
#   Image-level multi-label. Each film is reduced to one score per class by taking the max
#   of sqrt(cls x centerness) over the grid. Micro-F1 and micro-AUC are the headline;
#   macro sits beside them because micro is dominated by the common classes and would hide
#   a model that never predicts Pneumothorax at all.
#
#   The decision threshold is FITTED on validation and applied unchanged to test.
#   Checkpoints are selected on validation micro-AUC, which is threshold-free, so selection
#   and reporting cannot drift apart the way val_sens and val_sens_matched did in v14.
#
# WHY INPUT 2048 AND NOT 1024
#   VinDr is natively high resolution and the resizer must be the thing that decimates it,
#   not the dataloader. MEASURED over the 15,000-film manifest, native long side is p5
#   2430, median 2863, p95 3188, max 3408:
#
#       input   films upsampled   median decimation DONE BY THE LOADER
#       1024          0.0%                2.80x
#       2048          0.6%                1.40x
#
#   At input 1024 the loader throws away a 2.80x decimation with INTER_AREA - which
#   anti-aliases properly - before the resizer under test sees anything. That hands the
#   hard part of the problem to cv2 and leaves the resizer a pre-solved one, which is the
#   opposite of what this project is measuring. vindr.py's own docstring says the ~12x
#   decimation to target 224 is deliberate, "the harsher Nyquist case the learned resizer
#   is being asked to handle"; routing through 1024 removes exactly that.
#
#   Unlike NODE21 - where input 2048 upsamples about half the test split and the extra
#   resolution is therefore partly synthetic - VinDr genuinely carries 2048: only 0.6% of
#   films are below it. The i2048 cache already holds all 8,788 films including every one
#   of the 4,394 positives, so this costs nothing to switch to.
#
#   Decimation at input 2048: 9.14x to target 224, 8.00x to 256, 5.33x to 384, 4.00x to 512.
#
# COST
#   8 runs: {bilinear, node21_talebi_zeroinit} x targets {224, 256, 384, 512}, seed 42.
#   5,272 training films against NODE21's 3,591, and the resizer's conv stack runs at 2048,
#   so budget noticeably longer per epoch than the nodule ladder.

param(
    [int]      $InputSize     = 2048,
    [int[]]    $Targets       = @(224, 256, 384, 512),
    [int[]]    $Seeds         = @(42),
    [string[]] $Variants      = @("bilinear", "node21_talebi_zeroinit"),
    [int]      $MaxEpochs     = 50,
    [int]      $Patience      = 15,
    [int]      $SplitSeed     = 0,
    # SELECTION RULE. Both of these change which checkpoint a run keeps, so they must be
    # identical across every run that will be compared. Defaults reproduce v21/v22.
    # MEASURED on the 16 v21/v22 curves: raw-max selection inflates the chosen value by
    # +0.0045 against a resizer effect of +0.0023; a 5-epoch trailing mean cuts that to
    # +0.0027 and saves about 12 epochs per run.
    [int]      $SelectSmoothing = 1,
    [double]   $MinDelta       = 0.0,
    [switch]   $NoOversample,
    [string]   $RunGroup      = "v21_vindr_multiclass",
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
$CACHE      = Join-Path $VINDR_ROOT "train_png16_cache_i$InputSize"

if (-not (Test-Path $CACHE)) {
    throw "No VinDr cache at $CACHE. The PNG16 archive is gone, so this cache is the only image source at input $InputSize. Warm it first: python -m scripts.warm_vindr_cache --input-size $InputSize"
}

$cachedFilms = @(Get-ChildItem -Path $CACHE -Filter *.npy -ErrorAction SilentlyContinue).Count

if ($cachedFilms -lt 8000) {
    throw "Cache at $CACHE holds only $cachedFilms films. Expected ~8788. Re-warm it before sweeping."
}

# MEASURED on the 17.1 GB card at input 2048 with the 14-class head, bf16 autocast:
#
#     target  variant   batch   peak GB   % GPU      step   per-image
#     224     bilinear      8      1.20      7%     0.053s     0.007s
#     224     talebi        8      8.22     48%     0.270s     0.034s
#     256     bilinear      8      1.60      9%     0.055s     0.007s
#     256     talebi        8      8.45     49%     0.281s     0.035s
#     384     bilinear      8      2.81     16%     0.056s     0.007s
#     384     talebi        8      8.90     52%     0.489s     0.061s
#     512     bilinear      8      4.19     25%     0.079s     0.010s
#     512     talebi        8      8.92     52%     0.373s     0.047s
#
# Batch 8 fits with half the card spare and no accumulation needed. An earlier version of
# this file copied the nodule runners' "input >= 2048 means batch 4" rule, which was
# derived at targets 768-2048 where the HEAD is the expensive part. Here the heads are
# 56x56 to 128x128 and cost almost nothing, so that rule did not apply and was halving the
# batch for no reason.
#
# Step time is reported per image as well as per step on purpose. At target 2048 a batch
# that fit at 97% of the card ran at 14.2 s/step against 0.36 s at half the batch - a 39x
# penalty from allocator thrashing, not compute. Fitting is not the same as being usable,
# so any future change here is measured on both columns.
#
# Note the arms are NOT equally priced: the learned resizer costs 5-7x bilinear's step
# time because its conv stack runs at the full 2048 input before the resize. The Talebi
# runs dominate the sweep's wall clock.
$BATCH = 8
$ACCUM = 1

$RUNS = @()

foreach ($target in $Targets) {
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            $RUNS += @{ variant = $variant; seed = $seed; target = $target }
        }
    }
}

Write-Host ""
Write-Host "v21 VinDr 14-class sweep -- $($RUNS.Count) runs" -ForegroundColor Cyan
Write-Host "  input     $InputSize px  ($cachedFilms films cached)" -ForegroundColor DarkGray
Write-Host "  targets   $($Targets -join ', ')" -ForegroundColor DarkGray
Write-Host "  seeds     $($Seeds -join ', ')" -ForegroundColor DarkGray
Write-Host "  variants  $($Variants -join ', ')" -ForegroundColor DarkGray
Write-Host "  split     60/20/20 by film, seed $SplitSeed, stratified rarest-class-first" -ForegroundColor DarkGray
Write-Host "  sampling  $(if ($NoOversample) { 'plain' } else { 'rare-class oversampling, sqrt-scaled' })" -ForegroundColor DarkGray
Write-Host "  selection micro-AUC, $(if ($SelectSmoothing -gt 1) { "$SelectSmoothing-epoch trailing mean" } else { 'raw per-epoch max' }), min-delta $MinDelta" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  NOTE Atelectasis (37 test films) and Pneumothorax (19) are reported but" -ForegroundColor Yellow
Write-Host "       cannot support a per-class claim. Micro scores are the headline." -ForegroundColor Yellow
Write-Host ""

if ($DryRun) {
    foreach ($cfg in $RUNS) {
        Write-Host ("  {0,-24} s{1,-4} i{2} -> t{3,-5} batch {4} x accum {5}" -f `
                    $cfg.variant, $cfg.seed, $InputSize, $cfg.target, $BATCH, $ACCUM)
    }

    Write-Host ""
    Write-Host "Dry run -- nothing executed." -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    param($Config)

    # Must mirror train_vindr.py's leaf name exactly or -Resume accepts the wrong run
    $leaf = "$($Config.variant)_fcos_vindr14_s$($Config.seed)_i${InputSize}_t$($Config.target)"

    return Get-ChildItem -Path "outputs/vindr" -Directory -Recurse -ErrorAction SilentlyContinue |
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

    & python -u scripts/train_vindr.py @TrainArgs 2>&1 |
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

# Fail before an eight-run sweep rather than during it
& python -m src.data.vindr_split

if ($LASTEXITCODE -ne 0) { throw "VinDr split self-check failed -- not starting the sweep." }

& python -m src.evaluation.vindr_multilabel_metrics

if ($LASTEXITCODE -ne 0) { throw "Multi-label metric self-check failed -- not starting the sweep." }

$RUN_START = Get-Date
$completed = 0

foreach ($cfg in $RUNS) {
    $completed++
    $label = "$($cfg.variant) s$($cfg.seed) i$InputSize t$($cfg.target) ($completed/$($RUNS.Count))"

    if ($Resume) {
        $existing = Get-ExistingRun -Config $cfg

        if ($existing) {
            Write-Host "SKIP $label -- already complete at $($existing.FullName)" -ForegroundColor DarkYellow
            "SKIP $label" | Add-Content $LOG
            continue
        }
    }

    $trainArgs = @(
        "--variant",       $cfg.variant,
        "--backbone",      "convnext_fpn",
        "--input-size",    "$InputSize",
        "--target-size",   "$($cfg.target)",
        "--batch-size",    "$BATCH",
        "--grad-accum",    "$ACCUM",
        "--lr",            "2e-4",
        "--epochs",        "$MaxEpochs",
        "--patience",      "$Patience",
        "--seed",          "$($cfg.seed)",
        "--split-seed",    "$SplitSeed",
        "--select-smoothing", "$SelectSmoothing",
        "--min-delta",     $MinDelta.ToString([cultureinfo]::InvariantCulture),
        "--aug-profile",   "v3",
        "--num-workers",   "6",
        "--run-group",     $RunGroup,
        "--augment"
    )

    if (-not $NoOversample) { $trainArgs += "--oversample" }

    Invoke-Run -Label $label -TrainArgs $trainArgs

    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:N1} h" -f `
                $completed, $RUNS.Count, $elapsed.TotalHours) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v21 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Read in this order:" -ForegroundColor Yellow
Write-Host "  1. micro-AUC vs target, both arms. That is the sweep." -ForegroundColor DarkGray
Write-Host "  2. macro-AUC beside it -- a gap between them means rare-class collapse." -ForegroundColor DarkGray
Write-Host "  3. per-class AUC, ignoring Atelectasis and Pneumothorax as thin." -ForegroundColor DarkGray
Write-Host "  4. NEVER put these on an axis with the NODE21 nodule results." -ForegroundColor DarkGray
