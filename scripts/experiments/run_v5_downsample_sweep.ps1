# run_v5_downsample_sweep.ps1
# Downsampling sweep: does a learned resizer beat bilinear, and does the answer depend
# on how far the image is downsampled vs. what resolution it starts from?
#
# THREE ARMS (12 unique runs per seed):
#
#   ARM A -- bilinear, input=1024 -> target {512, 384, 256, 224}
#     Reference. F.interpolate does the downsample; backbone runs at target.
#
#   ARM B -- Talebi zeroinit, input=1024 -> target {512, 384, 256, 224}
#     Paired against Arm A: identical input and backbone resolution, so the ONLY
#     difference is who performs the downsample. This is the clean causal comparison.
#
#   ARM C -- Talebi zeroinit, input {1024, 512, 384, 256, 224} -> target 224
#     Backbone compute pinned at 224. Tests how much information the resizer can carry
#     from a higher-resolution input into a fixed small feature map.
#
#   ARM D -- Talebi zeroinit WIDE (n_filters 64), input {1024, 512, 384, 256, 224} -> 224
#     Arm C repeated at 4x resizer width. Arm C declines monotonically in FROC AUC as
#     input rises (0.883 at 224->224 down to 0.767 at 1024->224, five points, all same
#     direction). This arm separates "16 channels is too narrow to carry information
#     through a large downsample factor" from "the architecture cannot do it at any
#     width". n_filters=64 is a DEVIATION from Talebi Figure 3 (k7n16s1) and must be
#     reported as an ablation, never as "Talebi".
#
#   Arm B and Arm C both contain zeroinit input=1024 target=224. It runs ONCE and is
#   reused. Arm D shares no config with any other arm. Total: 17 runs per seed.
#
# RECIPE NOTES -- read before comparing these numbers to anything older:
#   - --select-on val_sens, patience 15. The 2026-08-18 attempt at val_loss selection
#     FAILED and is recorded here so it is not retried: focal_v5's val loss is dominated
#     by the ~25k negative heatmap cells against 2-6 positive peaks, so the lowest-loss
#     checkpoint is the most conservative one, not the best detector. Measured across
#     that sweep, val_loss and val sensitivity were POSITIVELY correlated in 11/12 runs
#     (mean r = +0.247) -- minimising val loss actively selects lower sensitivity. The
#     selected epochs landed at 3-10 with sensitivity 0.33-0.61 while the same runs
#     peaked at 0.65-0.76. val_sens is noisy, but it is at least measuring the target.
#   - --aug-profile v3. The stronger v4 augmentation dropped NODE21 FROC AUC
#     0.803 -> 0.767 on s42 and was confounded with positive oversampling in that run.
#     It stays out of this sweep until the aug-only disambiguation settles it.
#   - No --pos-frac, for the same reason.
#
# COST: the 12-run version measured 4h44m end to end on this machine. Arm D adds ~2h.
#   MEMORY WARNING for Arm D at input=1024: input_conv runs at full input resolution, so
#   64 channels at 1024^2 is 4x the activation footprint of the n_filters=16 arm (roughly
#   5.4 GB vs 1.3 GB for the LDM alone at batch 8). It should fit in 16 GB but has not
#   been verified. If it OOMs, that single run fails and the sweep continues -- rerun it
#   at batch 4 / accum 2, and note in the writeup that its BatchNorm statistics were
#   computed at a different micro-batch than the rest of Arm D.
#   Seeds multiply this directly: -Seeds 42,7,123 is roughly three nights. Run one seed
#   first, confirm the ordering holds, then decide which configs deserve more seeds.
#
# Ordering: cheapest targets first, so an interrupted night still yields usable configs.
#
# USAGE
#   .\scripts\experiments\run_v5_downsample_sweep.ps1
#   .\scripts\experiments\run_v5_downsample_sweep.ps1 -DryRun
#   .\scripts\experiments\run_v5_downsample_sweep.ps1 -Seeds 42,7,123
#   .\scripts\experiments\run_v5_downsample_sweep.ps1 -Resume

param(
    [int[]]  $Seeds      = @(42),
    [string] $SelectOn   = "val_sens",
    [int]    $Patience   = 15,
    [string] $AugProfile = "v3",
    [bool]   $IncludeWide = $true,
    [string] $RunGroup   = "v5_downsample_sweep",
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

# Micro-batch keyed on TARGET, since the backbone dominates memory and always runs at
# target resolution. grad_accum holds the effective batch at 8 everywhere so LR stays
# 2e-4 across the sweep with no LR-scaling confound. Talebi at input=1024 is known to
# fit at batch 8 from the v3/v4 runs, so input resolution does not drive this table.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

$RUNS = @()
$SEEN = @{}

function Add-Run {
    param([string] $Arm, [string] $Variant, [int] $InputRes, [int] $TargetRes, [int] $Seed)

    # Dedupe: Arm B and Arm C overlap at input=1024 target=224
    $key = "$Variant|$InputRes|$TargetRes|$Seed"

    if ($SEEN.ContainsKey($key)) {
        Write-Host "  (dedup) [$Arm] $Variant $InputRes->$TargetRes s$Seed already queued by arm $($SEEN[$key])" -ForegroundColor DarkYellow
        return
    }

    $SEEN[$key] = $Arm
    $sched = $BATCH_FOR_TARGET[$TargetRes]

    $script:RUNS += @{
        arm     = $Arm
        variant = $Variant
        input   = $InputRes
        target  = $TargetRes
        seed    = $Seed
        batch   = $sched.batch
        accum   = $sched.accum
    }
}

foreach ($seed in $Seeds) {

    # Arms A and B interleaved so each paired comparison completes together. If the night
    # is cut short you still have matched pairs rather than a full arm and half of another.
    foreach ($t in @(224, 256, 384, 512)) {
        Add-Run -Arm "A" -Variant "bilinear"               -InputRes 1024 -TargetRes $t -Seed $seed
        Add-Run -Arm "B" -Variant "node21_talebi_zeroinit" -InputRes 1024 -TargetRes $t -Seed $seed
    }

    # Arm C: input varies, target pinned at 224
    foreach ($i in @(224, 256, 384, 512, 1024)) {
        Add-Run -Arm "C" -Variant "node21_talebi_zeroinit" -InputRes $i -TargetRes 224 -Seed $seed
    }

    # Arm D: Arm C at 4x resizer width. Same micro-batch as Arm C so the two are directly
    # comparable -- differing batch size would change the LDM's BatchNorm statistics and
    # confound the exact width comparison this arm exists to make.
    if ($IncludeWide) {
        foreach ($i in @(224, 256, 384, 512, 1024)) {
            Add-Run -Arm "D" -Variant "node21_talebi_zeroinit_wide" -InputRes $i -TargetRes 224 -Seed $seed
        }
    }
}

# Manifest written up front so the sweep is reconstructable even if the log is lost
$manifestObj = [ordered]@{
    run_group       = $RunGroup
    started         = $STAMP
    seeds           = $Seeds
    select_on       = $SelectOn
    patience        = $Patience
    aug_profile     = $AugProfile
    pos_frac        = $null
    lr              = "2e-4"
    datasets        = "node21"
    backbone        = "convnext_fpn"
    effective_batch = 8
    include_wide    = $IncludeWide
    n_runs          = $RUNS.Count
    runs            = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v5 downsample sweep -- $($RUNS.Count) runs across seeds [$($Seeds -join ',')]" -ForegroundColor Cyan
Write-Host "  select_on=$SelectOn patience=$Patience aug_profile=$AugProfile pos_frac=none" -ForegroundColor DarkGray
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    Write-Host ("  [{0}] {1,-24} {2,4} -> {3,-4} s{4} batch={5} accum={6}" -f $cfg.arm, $cfg.variant, $cfg.input, $cfg.target, $cfg.seed, $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
}
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run -- nothing executed. Manifest written to $MANIFEST" -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    <#
        Locate a completed run for this config. train.py's output dir is
        outputs/<date>_<run_group>/<variant>_<head>_<loss>_s<seed>[_i<in>_t<out>], and the
        date prefix varies, so match on the leaf name and the run-group suffix of its parent.
    #>
    param($Config)

    $resTag = ""
    if (-not ($Config.input -eq 2048 -and $Config.target -eq 512)) {
        $resTag = "_i$($Config.input)_t$($Config.target)"
    }

    $leaf = "$($Config.variant)_centernet_focal_v5_s$($Config.seed)$resTag"

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

    # A failed run must not abort the remaining nights' work
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
}

Write-Host "Running smoke test..." -ForegroundColor Yellow
& python scripts/diagnostics/smoke_test_node21_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed. Aborting sweep." }

$RUN_START = Get-Date
$completed = 0

foreach ($cfg in $RUNS) {
    $completed++
    $label = "[$($cfg.arm)] $($cfg.variant) $($cfg.input)->$($cfg.target) s$($cfg.seed) ($completed/$($RUNS.Count))"

    if ($Resume) {
        $existing = Get-ExistingRun -Config $cfg

        if ($existing) {
            Write-Host "SKIP $label -- already complete at $($existing.FullName)" -ForegroundColor DarkYellow
            "SKIP $label" | Add-Content $LOG
            continue
        }
    }

    Invoke-Run -Label $label -TrainArgs @(
        "--backbone",    "convnext_fpn",
        "--variant",     $cfg.variant,
        "--seed",        "$($cfg.seed)",
        "--datasets",    "node21",
        "--input-size",  "$($cfg.input)",
        "--target-size", "$($cfg.target)",
        "--batch-size",  "$($cfg.batch)",
        "--grad-accum",  "$($cfg.accum)",
        "--lr",          "2e-4",
        "--epochs",      "50",
        "--patience",    "$Patience",
        "--select-on",   $SelectOn,
        "--aug-profile", $AugProfile,
        "--val-frac",    "0.20",
        "--run-group",   $RunGroup,
        "--augment",
        "--froc"
    )

    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:hh\:mm}" -f $completed, $RUNS.Count, $elapsed) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v5 downsample sweep complete. Log: $LOG" -ForegroundColor Green
Write-Host "Reconstruct results with:" -ForegroundColor Yellow
Write-Host "  1. python scripts/eval/run_size_stratified_all.py" -ForegroundColor DarkGray
Write-Host "  2. python scripts/eval/aggregate_v2_results.py" -ForegroundColor DarkGray
Write-Host "  Each run's metrics.json carries select_on, aug_profile, pos_frac, input_size," -ForegroundColor DarkGray
Write-Host "  target_size and backbone; froc_data.json carries the full operating curve." -ForegroundColor DarkGray
