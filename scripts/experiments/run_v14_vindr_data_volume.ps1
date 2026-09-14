# run_v14_vindr_data_volume.ps1
# Does the residual fail to concentrate on lesions because it is starved of data?
#
# THE QUESTION
#   Residual enrichment sits at 0.677-0.856 in every cell measured so far - every
#   variant, every resolution, 11-plus runs. The residual consistently puts LESS energy
#   on nodules than on matched control patches, while the loss demands 130-230x MORE
#   there. One standing explanation is data volume: NODE21 offers 1,476 boxes across
#   3,591 training films, and the v12 training curves show train loss falling
#   monotonically while validation loss climbs from about epoch 12, which reads as
#   overfitting rather than undertraining.
#
#   VinDr-CXR supplies 18,231 fused boxes across 4,394 abnormal films (plus 10,606
#   negatives), roughly 12.4x NODE21's box supervision. Every class is collapsed to one
#   generic foreground class - this is box supervision for the resizer, not pathology
#   classification.
#
# WHY TARGET 224
#   1024 -> 224 is a 4.6x decimation with no low-pass filter anywhere in the pipeline,
#   which is where bilinear violates Nyquist hardest and where a learned resizer has the
#   most headroom to beat it. Targets 384 and 512 decimate 2.7x and 2.0x, where there is
#   far less for a resizer to recover.
#
# THE GRID -- 2x2, data volume x resizer
#
#            | NODE21 only            | NODE21 + VinDr
#   ---------|------------------------|----------------
#   bilinear | 0.263 (v12, s42)       | run here
#   zeroinit | 0.220 (v12, s42)       | run here
#
#   The NODE21-only cells already exist from v12 at seed 42, same head, same input and
#   target size, same batch. The pipeline is deterministic at fixed seed (trap 4.7) and
#   node21.py was NOT modified for this work, so re-running them would reproduce them
#   exactly. They are reused by default; -WithControls re-runs them anyway.
#
#   READ ENRICHMENT FIRST. It is the primary readout and the only one with the
#   statistical power to resolve a change here.
#
# WHAT THIS EXPERIMENT CAN AND CANNOT SHOW
#   CAN: whether more box supervision moves residual enrichment off 0.7, and whether the
#   validation loss divergence from epoch 12 closes.
#
#   CANNOT: make the detection metric less noisy. The +-0.041 floor is binomial SE on
#   147 test nodules, and the test split is untouched by this change. Enrichment is
#   scored over every GT box against 8 matched controls each, so its n is orders of
#   magnitude larger.
#
#   A null here is a strong result, not a wasted run: it converts "maybe the dataset was
#   too small" from an open objection into a measured and closed one.
#
# TRAPS THAT APPLY TO READING THESE RUNS
#   * Detection numbers are NOT comparable to the v11/v12 grid. A model trained on 14
#     pathologies emits boxes on cardiomegaly and aortic enlargement in NODE21 test
#     films, which score as false positives against nodule-only ground truth and force a
#     higher threshold at a matched 0.25 FP/img budget. Both arms eat this equally, so
#     the bilinear-vs-zeroinit DELTA stays controlled - the absolute number does not.
#     Keep these runs in their own block in every figure.
#   * An epoch is ~5.2x more optimizer steps here than in a NODE21-only run, because the
#     pooled training set is 18,591 films against 3,591. Early stopping on NODE21 val
#     sensitivity absorbs most of this, but CHECK THE TRAINING CURVE before comparing
#     anything (trap 13.3: an epoch budget is not transferable across a change of this
#     size). One run peaking on its final epoch means the budget truncated it.
#   * VinDr images are downscaled with INTER_AREA, not INTER_LINEAR. Native size is
#     2128-3408 px, so reaching input_size decimates 1.3-1.7x, and a plain bilinear
#     kernel there would put aliasing in the dataloader as well as in the resizer under
#     test. node21.py is deliberately UNCHANGED, so NODE21 films still take the legacy
#     INTER_LINEAR path and stay bit-identical to previous runs.
#   * VinDr is almost entirely non-square where NODE21 is 39% non-square, so the squash
#     default distorts a far larger share of the pooled training set. Left at the default
#     so these runs stay comparable to their v12 controls; worth an -Letterbox arm later.
#
# COST  ~12h per pooled run at target 224 (t224 is the slowest target - low resolution
#   early-stops later - and the pooled set is 5.2x the films). 2 runs, roughly 25h.
#   -WithControls adds 2 NODE21-only runs at ~2.4h each.
#
# PREREQUISITE  scripts/preprocess_vindr.py must have completed. The dataset will fail
#   loudly on a missing PNG rather than silently skipping films.
#
# USAGE
#   .\scripts\experiments\run_v14_vindr_data_volume.ps1 -DryRun
#   .\scripts\experiments\run_v14_vindr_data_volume.ps1
#   .\scripts\experiments\run_v14_vindr_data_volume.ps1 -WithControls
#   .\scripts\experiments\run_v14_vindr_data_volume.ps1 -Seeds 42,7 -Resume

param(
    [string] $Arms            = "V",
    [int[]]  $Seeds           = @(42),
    [string] $Head            = "fcos",
    [int]    $TargetSize      = 224,
    [int]    $InputSize       = 1024,
    [int]    $NativeVindrSize = 2048,
    [double] $VindrNegativeRatio = 1.0,
    [int]    $MaxEpochs       = 50,
    [int]    $Patience        = 15,
    [string] $SelectOn        = "val_sens",
    [string] $RunGroup        = "v14_vindr_data_volume",
    [switch] $WithControls,
    [switch] $DryRun,
    [switch] $Resume
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$LOG_ROOT = Join-Path "outputs" $RunGroup
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }

$STAMP    = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG      = Join-Path $LOG_ROOT "run_$STAMP.log"
$MANIFEST = Join-Path $LOG_ROOT "manifest_$STAMP.json"

# Fail before burning a night on a half-converted archive.
$VINDR_IMAGES = "D:/Downloads/vinbigdata-chest-xray-abnormalities-detection/train_png16"
$VINDR_CSV    = "D:/Downloads/vinbigdata-chest-xray-abnormalities-detection/train.csv"

$VINDR_DICOM = "D:/Downloads/vinbigdata-chest-xray-abnormalities-detection/train"

if (-not (Test-Path $VINDR_CSV)) { throw "VinDr annotations not found: $VINDR_CSV" }

# The PNG16 archive is OPTIONAL. The per-input-size cache supersedes it and is built
# straight from DICOM by scripts/warm_vindr_cache.py, so only the DICOM source is
# genuinely required. A warm cache means ~14 ms/film; cold means ~370 ms, which is what
# made the first attempt at this sweep dataloader-bound.
if (-not (Test-Path $VINDR_DICOM)) {
    throw "VinDr DICOM source not found: $VINDR_DICOM. Nothing can be rebuilt without it."
}

# Only the sizes the requested arms will actually load. Arm V uses $InputSize; Arm N adds
# $NativeVindrSize. Checking a size no arm needs produces a warning about work nobody asked
# for, which trains the reader to ignore these lines.
$sizesInUse = @()
if ($Arms -match "V" -or $Arms -match "N") { $sizesInUse += $InputSize }
if ($Arms -match "N") { $sizesInUse += $NativeVindrSize }

foreach ($size in ($sizesInUse | Select-Object -Unique)) {
    $cacheDir = "D:/Downloads/vinbigdata-chest-xray-abnormalities-detection/train_png16_cache_i$size"
    $cached = (Get-ChildItem -Path $cacheDir -Filter "*.npy" -ErrorAction SilentlyContinue).Count
    $ratioText = $VindrNegativeRatio.ToString([cultureinfo]::InvariantCulture)

    if ($cached -lt 8000) {
        # Parenthesised BEFORE -f: without it the operator binds to the trailing string
        # literal only, and the {0}/{1} placeholders print verbatim.
        Write-Host (("  [warn] cache at input {0} holds only {1} films. First epoch will be " +
                     "SLOW while it populates. Prebuild with:") -f $size, $cached) -ForegroundColor Yellow
        Write-Host ("         python -m scripts.warm_vindr_cache --input-size {0} --negative-ratio {1}" -f `
                    $size, $ratioText) -ForegroundColor DarkGray
    } else {
        Write-Host "  [ok] cache at input $size holds $cached films" -ForegroundColor DarkGray
    }
}

# Micro-batch keyed on target, matching v8 onward so trap 13.1 (bf16 evaluation is
# batch-size dependent) does not put a batch difference inside the comparison.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

$RUNS = @()

function Add-Run {
    param([string] $Arm, [string] $Variant, [int] $Seed, [string] $Datasets,
          [int] $VindrInputSize = 0)

    $sched = $BATCH_FOR_TARGET[$TargetSize]
    $batch = $sched.batch
    $accum = $sched.accum

    # The Talebi resizer runs its conv stack at INPUT resolution, so a VinDr batch at
    # 2048 is 4x the activation memory of one at 1024 (measured: 3.16 GB at 2700 batch 2
    # against 1.15 GB at 1024). Halve the micro-batch and double accumulation so the
    # effective batch stays 8 and no LR-scaling confound enters the comparison.
    if ($VindrInputSize -ge 2048) {
        $batch = [int]($batch / 2)
        $accum = $accum * 2
    }

    $script:RUNS += @{
        arm = $Arm; variant = $Variant; seed = $Seed; datasets = $Datasets
        vindr_input_size = $VindrInputSize
        batch = $batch; accum = $accum
    }
}

# ---- Arm V: pooled at MATCHED resolution. Varies data volume alone. ----
if ($Arms -match "V") {
    foreach ($seed in $Seeds) {
        foreach ($variant in @("bilinear", "node21_talebi_zeroinit")) {
            Add-Run -Arm "V" -Variant $variant -Seed $seed -Datasets "node21_vindr"
        }
    }
}

# ---- Arm N: pooled with VinDr at NATIVE resolution, so the resizer sees ~12x ----
#   VinDr enters at $NativeVindrSize while NODE21 stays at $InputSize, which makes the
#   resizer perform a 12.1x decimation to 224 on VinDr films against 4.6x on NODE21.
#   Measured over 15 films under the CLAHE pipeline, bilinear carries 1.32x more aliasing
#   error at 12.1x than at 4.6x, so this is the case where a learned resizer has the most
#   to repair. Requires single-source batches; ConvNeXt only.
#
#   Read Arm N against Arm V, not against Arm C: N differs from V by decimation ratio
#   alone, which is the comparison that isolates it.
if ($Arms -match "N") {
    foreach ($seed in $Seeds) {
        foreach ($variant in @("bilinear", "node21_talebi_zeroinit")) {
            Add-Run -Arm "N" -Variant $variant -Seed $seed -Datasets "node21_vindr" `
                    -VindrInputSize $NativeVindrSize
        }
    }
}

# ---- Arm C: NODE21-only controls, reused from v12 unless explicitly re-run ----
if ($WithControls -or $Arms -match "C") {
    foreach ($seed in $Seeds) {
        foreach ($variant in @("bilinear", "node21_talebi_zeroinit")) {
            Add-Run -Arm "C" -Variant $variant -Seed $seed -Datasets "node21"
        }
    }
}

$manifestObj = [ordered]@{
    run_group         = $RunGroup
    started           = $STAMP
    arms              = $Arms
    native_vindr_size = $NativeVindrSize
    vindr_negative_ratio = $VindrNegativeRatio
    head              = $Head
    input_size     = $InputSize
    target_size    = $TargetSize
    seeds          = $Seeds
    max_epochs     = $MaxEpochs
    patience       = $Patience
    select_on      = $SelectOn
    attribution    = $true
    primary_metric = "residual enrichment (detection is secondary and arm-vs-arm only)"
    evaluation     = "box IoU (0.25 primary, 0.5 reported), per-image sampling"
    comparable_to  = "each other; NOT to the v11/v12 grid on absolute detection"
    vindr_images   = $VINDR_IMAGES
    n_runs         = $RUNS.Count
    runs           = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v14 VinDr data volume -- $($RUNS.Count) runs | head=$Head $InputSize->$TargetSize" -ForegroundColor Cyan
Write-Host "  VinDr source: DICOM (the PNG16 archive is superseded by the per-size caches)" -ForegroundColor DarkGray
Write-Host "  READ ENRICHMENT FIRST. Absolute detection is NOT comparable to v11/v12." -ForegroundColor Yellow

if ($VindrNegativeRatio -gt 0) {
    Write-Host ("  VinDr negatives capped at {0}x positives: 8788 films, ALL 18231 boxes kept." -f `
        $VindrNegativeRatio.ToString([cultureinfo]::InvariantCulture)) -ForegroundColor DarkGray
    Write-Host "  Pooled train ~12379 films (~3.4x NODE21-only). CHECK THE TRAINING CURVE." -ForegroundColor Yellow
} else {
    Write-Host "  All 15000 VinDr films: pooled train 18591 (~5.2x). CHECK THE TRAINING CURVE." -ForegroundColor Yellow
}

if (-not ($Arms -match "C" -or $WithControls)) {
    Write-Host "  WARNING: no Arm C. v12 controls predate the 2026-08-27 INTER_AREA fix." -ForegroundColor Red
}

Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    # InvariantCulture, or a machine with a comma decimal separator prints "4,6x".
    $inv   = [cultureinfo]::InvariantCulture
    $ratio = if ($cfg.vindr_input_size) { $cfg.vindr_input_size / $TargetSize } else { $InputSize / $TargetSize }
    $ratioText = [math]::Round($ratio, 1).ToString($inv)

    $res = if ($cfg.vindr_input_size) {
        "vindr@$($cfg.vindr_input_size) (${ratioText}x)"
    } else {
        "matched (${ratioText}x)"
    }

    Write-Host ("  [{0}] {1,-26} s{2,-5} {3,-13} {4,-22} batch={5}/accum={6}" -f `
        $cfg.arm, $cfg.variant, $cfg.seed, $cfg.datasets, $res, $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
}
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run -- nothing executed. Manifest written to $MANIFEST" -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    param($Config)

    # Must mirror train.py's _recipe_tag exactly, or -Resume will treat a matched-resolution
    # pooled run as satisfying a native-resolution one.
    $resTag    = "_i${InputSize}_t${TargetSize}"
    $recipeTag = ""
    if ($Config.datasets -eq "node21_vindr") { $recipeTag += "_vindr" }
    if ($Config.vindr_input_size -and $Config.vindr_input_size -ne $InputSize) {
        $recipeTag += "_vin$($Config.vindr_input_size)"
    }
    if ($Config.datasets -eq "node21_vindr" -and $VindrNegativeRatio -gt 0) {
        $recipeTag += "_vnr" + $VindrNegativeRatio.ToString([cultureinfo]::InvariantCulture)
    }
    $leaf = "$($Config.variant)_${Head}_focal_v5_s$($Config.seed)${resTag}${recipeTag}"

    return Get-ChildItem -Path "outputs" -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq $leaf -and
            $_.Parent.Name -like "*_$RunGroup" -and
            (Test-Path (Join-Path $_.FullName "metrics.json"))
        } |
        Select-Object -First 1
}

function Get-CommitHeadroomGB {
    $counters = Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' -ErrorAction SilentlyContinue

    if (-not $counters) { return [double]::MaxValue }

    $committed = ($counters.CounterSamples | Where-Object Path -match 'Committed').CookedValue
    $limit     = ($counters.CounterSamples | Where-Object Path -match 'Limit').CookedValue

    return ($limit - $committed) / 1GB
}

function Invoke-Run {
    param([string] $Label, $TrainArgs)

    Write-Host "=== $Label ===" -ForegroundColor Cyan
    "=== $Label ===" | Add-Content $LOG

    # A crashed run leaves its DataLoader workers orphaned, each holding ~1.5 GB of commit.
    # 33 of them once held 56 GB of a 90 GB limit and made every subsequent run die
    # instantly with WinError 1455 ("paging file is too small"). Snapshot what exists now
    # so anything this run leaks can be identified precisely afterwards.
    $preExisting = @(Get-Process python3.13 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)

    $headroom = Get-CommitHeadroomGB

    if ($headroom -lt 8) {
        Write-Host ("  [warn] only {0:N1} GB of commit headroom. A run needs roughly 15 GB; " -f $headroom) -ForegroundColor Red
        Write-Host "         expect WinError 1455. Check for orphaned python3.13 processes." -ForegroundColor Red
    }

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

    # Reap only processes that appeared during THIS run and outlived it. Anything the user
    # had running beforehand is in $preExisting and is left alone.
    $leaked = @(Get-Process python3.13 -ErrorAction SilentlyContinue |
                Where-Object { $preExisting -notcontains $_.Id })

    if ($leaked.Count -gt 0) {
        $leakedGB = ($leaked | Measure-Object PagedMemorySize64 -Sum).Sum / 1GB
        Write-Host ("  [cleanup] reaping {0} orphaned workers holding {1:N1} GB of commit" -f `
                    $leaked.Count, $leakedGB) -ForegroundColor Yellow
        "$Label leaked $($leaked.Count) workers, reaped" | Add-Content $LOG
        $leaked | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    }
}

Write-Host "Running smoke test..." -ForegroundColor Yellow
& python scripts/diagnostics/smoke_test_node21_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed. Aborting sweep." }

$RUN_START = Get-Date
$completed = 0

foreach ($cfg in $RUNS) {
    $completed++
    $label = "[$($cfg.arm)] $($cfg.variant) s$($cfg.seed) $($cfg.datasets) ($completed/$($RUNS.Count))"

    if ($Resume) {
        $existing = Get-ExistingRun -Config $cfg

        if ($existing) {
            Write-Host "SKIP $label -- already complete at $($existing.FullName)" -ForegroundColor DarkYellow
            "SKIP $label" | Add-Content $LOG
            continue
        }
    }

    $trainArgs = @(
        "--backbone",    "convnext_fpn",
        "--variant",     $cfg.variant,
        "--head",        $Head,
        "--seed",        "$($cfg.seed)",
        "--datasets",    $cfg.datasets,
        "--input-size",  "$InputSize",
        "--target-size", "$TargetSize",
        "--batch-size",  "$($cfg.batch)",
        "--grad-accum",  "$($cfg.accum)",
        "--lr",          "2e-4",
        "--epochs",      "$MaxEpochs",
        "--patience",    "$Patience",
        "--select-on",   $SelectOn,
        "--aug-profile", "v3",
        "--val-frac",    "0.20",
        "--run-group",   $RunGroup,
        "--augment",
        "--froc",
        "--attribution"
    )

    if ($cfg.vindr_input_size) {
        $trainArgs += @("--vindr-input-size", "$($cfg.vindr_input_size)")
    }

    if ($cfg.datasets -eq "node21_vindr" -and $VindrNegativeRatio -gt 0) {
        $trainArgs += @("--vindr-negative-ratio",
                        $VindrNegativeRatio.ToString([cultureinfo]::InvariantCulture))
    }

    Invoke-Run -Label $label -TrainArgs $trainArgs

    # TotalHours, not "hh\:mm" - the latter drops the days component, so a 30-hour
    # cumulative prints as "06:06" and reads like the timer went backwards.
    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:N1} h" -f $completed, $RUNS.Count, $elapsed.TotalHours) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v14 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Read in this order:" -ForegroundColor Yellow
Write-Host "  1. residual enrichment in each metrics.json -- did it move off ~0.7?" -ForegroundColor DarkGray
Write-Host "  2. the train/val loss curves -- did the epoch-12 divergence close?" -ForegroundColor DarkGray
Write-Host "  3. detection delta bilinear vs zeroinit -- WITHIN this block only" -ForegroundColor DarkGray
