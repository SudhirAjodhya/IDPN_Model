# run_v20_input_2048.ps1
# Extend the input-resolution axis to 2048 at the HIGH targets, and in doing so test the
# mechanism proposed for the target-1024 result.
#
# WHAT ALREADY EXISTS AT INPUT 2048
#   v17_input_resolution covers targets 224 / 256 / 384 / 512, seed 42 only, both arms.
#   Nothing above 512. This run group adds 768 and 1024, optionally 2048.
#
# WHY TARGET 1024 IS THE CELL THAT MATTERS, AND WHY IT RUNS FIRST
#   At input 1024, target 1024 makes F.interpolate a bit-exact identity - verified,
#   max |ldm(x) - x| = 0.000e+00 over every test film. The bilinear arm therefore receives
#   native unmodified pixels, and it is the ONLY configuration in the study where that is
#   true. Measured consequence (v19, 3 seeds, last third of training):
#
#       target  variant   train loss   val loss   val/train   val FP/img
#       768     bilinear     1.000       1.149       1.1         0.40
#       768     talebi       1.032       0.444       0.4         0.33
#       1024    bilinear     1.084       9.617       8.9         1.39
#       1024    talebi       1.047       0.472       0.5         0.35
#
#   Bilinear at 1024 fits the TRAINING set as well as the arm that beats it by 26 points
#   (1.084 against 1.047). It fails only on held-out films, by producing 3.5x the false
#   positives. Its validation FP/img is pinned at 0.39-0.41 at every target from 224 to
#   768 and triples at 1024 alone. That is overfitting, not a failure to learn.
#
#   PROPOSED MECHANISM. At every other target the resize itself low-passes, which strips
#   per-pixel sensor noise and scanner texture before the backbone sees them. Both arms
#   get that, so it cancels from the comparison. At target 1024 it vanishes for bilinear
#   only. The learned arm still has one: its output attenuates high frequencies to 28% of
#   what its own 3.198x gain would produce.
#
#   WHAT THIS RUN GROUP TESTS. At input 2048, target 1024 is a genuine 2x downsample, so
#   bilinear DOES low-pass there. If the mechanism is right, bilinear at 2048 -> 1024 does
#   NOT collapse: its val FP/img should sit near 0.40 rather than 1.39, and the gap to the
#   learned arm should fall back to the +0.00 to +0.05 seen everywhere else on the ladder.
#
#   If bilinear collapses at 2048 -> 1024 TOO, the mechanism is wrong and the cause is
#   something about the 256x256 head grid rather than about the input being unfiltered.
#   Either outcome is informative, which is why this is worth running before the blur
#   control.
#
# CAVEAT THAT MUST TRAVEL WITH ANY 2048 RESULT
#   NODE21 has no native square resolution. MEASURED over all 4,882 films: only 37% are
#   1024x1024, the rest run to 4280x3520, median 2048x1864, 57% non-square. At input 2048
#   roughly HALF the NODE21 test split is UPSAMPLED from native. So "input 2048" is partly
#   synthetic resolution for NODE21, and a gain there is not evidence that real extra
#   detail helps. VinDr is the dataset that genuinely carries 2048 (long side 2429 at p5,
#   median 2863), and it is already at 2048 in every cell here.
#
#   This is the documented reason NODE21 was held at 1024 from v15 onward. It is raised
#   here because the supervisor asked for the 2048 input axis and v17 already established
#   the setting - not because the upsampling concern was resolved.
#
# TARGET 2048 - THE BACKBONE ACTUALLY RECEIVES 2048x2048
#   The default target is 2048, so the resizer hands the backbone a full 2048 film and the
#   FCOS head predicts on a 512x512 grid - 262,144 cells against 1-2 nodules per film, 4x
#   the cell count of the target-1024 runs.
#
#   EXPECT THE BILINEAR ARM TO STRUGGLE HERE, and know why before reading the result. At
#   input 2048, target 2048 makes F.interpolate an exact identity, which is precisely the
#   condition that produced the target-1024 anomaly - bilinear receives native unmodified
#   pixels with no low-pass anywhere in its path. If the regularisation mechanism is right,
#   this cell reproduces that collapse and probably amplifies it. That is a prediction the
#   run tests, not a reason to skip it, but a large gap here is NOT independent evidence
#   for the learned resizer: it is the same phenomenon at a larger grid.
#
#   Target 1024 is kept in the default list as the paired control, since at input 2048 it
#   is a genuine 2x downsample and bilinear DOES low-pass there.
#
# COST
#   Default 4 runs: {bilinear, node21_talebi_zeroinit} x {2048, 1024}, seed 42.
#   Target 2048 runs first, then 1024 as its control.
#   Add seeds with -Seeds 42,7,123 once the seed-42 direction is known.
#   -Resume skips any cell that already wrote a metrics.json under this run group.

param(
    [string]   $Head               = "fcos",
    [int]      $InputSize          = 2048,
    [int]      $VindrInputSize     = 2048,
    [int[]]    $Targets            = @(2048, 1024),
    [int[]]    $Seeds              = @(42),
    [string[]] $Variants           = @("bilinear", "node21_talebi_zeroinit"),
    [double]   $VindrNegativeRatio = 1.0,
    [string]   $SelectOn           = "val_sens_matched",
    [int]      $MaxEpochs          = 50,
    [int]      $Patience           = 15,
    [int]      $BatchOverride      = 0,
    [int]      $AccumOverride      = 0,
    [string]   $RunGroup           = "v20_input_2048",
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

# These runs are only comparable to v19 if they carry the same normaliser. Without the
# fix, a 512x512 head grid at target 2048 would distort nodule-free batches by 84x.
$lossSource = Get-Content "src/losses/fcos.py" -Raw

if ($lossSource -notmatch "REFERENCE_GRID_SIDE") {
    throw "src/losses/fcos.py does not carry the normaliser fix. Aborting - these runs would not be comparable to v19."
}

# MEASURED on the 17.1 GB card, synthetic batches, bf16 autocast:
#
#     target  variant   batch   peak GB   % GPU      step
#     2048    bilinear      2     13.03     76%     0.304s
#     2048    talebi        2     16.53     97%    14.182s   <-- thrashing, unusable
#     2048    bilinear      1      6.75     39%     0.160s
#     2048    talebi        1      8.50     50%     0.364s
#     1024    bilinear      2      3.86     23%     0.101s
#     1024    talebi        2      5.15     30%     0.209s
#
# BATCH 2 AT TARGET 2048 IS A TRAP. It does not OOM - it fits at 97% of the card and then
# spends its time evicting, 14.18 s/step against 0.364 s at batch 1. That is a 39x
# slowdown from dropping the batch, i.e. the run would start cleanly and then never
# finish (roughly 24 h/epoch against 75 min). Do not raise the batch at this target
# because it "fits".
#
# WHY THE LEARNED ARM COSTS SO MUCH MORE HERE. Its conv stack runs at INPUT resolution,
# before the resize, so it holds 2048x2048x16 activations through the whole residual path
# regardless of target. Bilinear holds nothing. At input 2048 that is a 43x peak-memory
# multiple at batch 2. This is a real property of the method, not an implementation
# artefact, and belongs in the discussion if the thesis argues for learned downsampling
# at high input resolution.
#
# grad_accum holds the effective batch at 8 throughout, matching v15/v18/v19, so no
# LR-scaling confound enters when comparing across run groups.
function Get-Schedule {
    param([int] $Target)

    if ($Target -ge 2048) { return @{ batch = 1; accum = 8 } }

    return @{ batch = 4; accum = 2 }
}

$RUNS = @()

foreach ($target in $Targets) {
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            $schedule = Get-Schedule -Target $target

            if ($BatchOverride -gt 0) { $schedule.batch = $BatchOverride }
            if ($AccumOverride -gt 0) { $schedule.accum = $AccumOverride }

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
Write-Host "v20 input 2048 -- $($RUNS.Count) runs" -ForegroundColor Cyan
Write-Host "  node21 @ $InputSize px, vindr @ $VindrInputSize px" -ForegroundColor DarkGray
Write-Host "  targets   $($Targets -join ', ')" -ForegroundColor DarkGray
Write-Host "  seeds     $($Seeds -join ', ')" -ForegroundColor DarkGray
Write-Host "  variants  $($Variants -join ', ')" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  NOTE at input 2048 roughly half the NODE21 test split is UPSAMPLED from" -ForegroundColor Yellow
Write-Host "       native. Only VinDr genuinely carries this resolution." -ForegroundColor Yellow
Write-Host ""

if ($DryRun) {
    foreach ($cfg in $RUNS) {
        Write-Host ("  {0,-24} s{1,-4} t{2,-5} batch {3} x accum {4}" -f `
                    $cfg.variant, $cfg.seed, $cfg.target, $cfg.batch, $cfg.accum)
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

    # A crashed run leaves orphaned DataLoader workers holding ~1.5 GB of commit each, and
    # 2048-px films make each worker heavier. Snapshot so this run's leaks can be reaped.
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

if ($LASTEXITCODE -ne 0) { throw "Smoke test failed -- not starting the sweep." }

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
Write-Host "v20 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Read in this order:" -ForegroundColor Yellow
Write-Host "  1. bilinear i2048 t1024 val FP/img over the last third of training." -ForegroundColor DarkGray
Write-Host "     ~0.40 means the low-pass mechanism holds. ~1.4 means it does not." -ForegroundColor DarkGray
Write-Host "  2. the bilinear-vs-talebi gap at t1024. Back to +0.00-0.05 confirms it." -ForegroundColor DarkGray
Write-Host "  3. NEVER pool with v17 - that group predates the normaliser fix." -ForegroundColor DarkGray
