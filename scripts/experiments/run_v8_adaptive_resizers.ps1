# run_v8_adaptive_resizers.ps1
# Resizer architectures built from the 2026-08-21 attribution measurement.
#
# WHAT THE MEASUREMENT SAID
#   dL/d(residual) is 130-230x concentrated on nodules versus matched background, and is
#   statistically identical whether or not a learnable resizer occupies the slot. The
#   trained Talebi residual answers with enrichment 0.62-0.85 - it puts LESS energy on
#   nodules than on matched control patches. The instruction is fine; the response is not.
#
#   Cause: receptive field. Talebi's post-interpolate stack sees ~11 target cells against
#   nodules of 9.8 (t=224), 16.9 (t=384) and 22.5 (t=512) - a view 0.5x to 1.1x the size
#   of the object it must distinguish from a vessel or rib crossing. It cannot be
#   selective, so it learns a spatially uniform transform, and uniform transforms deposit
#   energy on high-frequency structure rather than low-contrast nodules.
#
# EVERY RUN CARRIES --attribution, so residual enrichment lands in metrics.json next to
# sensitivity. That is the point: enrichment says whether an intervention fixed the
# MEASURED DEFICIT, independently of whether detection moved. Detection deltas here are
# ~0.02-0.08 against a seed spread of 0.020 (fcos) to 0.082 (centernet_det); enrichment
# has a per-nodule SEM around 0.04 on n=147 and is far more sensitive.
#
# ============================================================================
# ARM M -- mechanism 2x2: receptive field x selectivity
# ============================================================================
#   {zeroinit, ldm_dilated, ldm_gated, ldm_dilated_gated} x seeds {42, 7}, fcos, 1024->384
#
#   zeroinit         RF 11  no gate    the current architecture, as control
#   ldm_dilated      RF 63  no gate    receptive field alone
#   ldm_gated        RF 11  gate       selectivity alone
#   ldm_dilated_gated RF 63 gate       both
#
#   READ: residual enrichment first. If dilation alone lifts it toward 1.0, receptive
#   field was the binding constraint. If gating alone does, selectivity was. If only both
#   together do, a gate is useless without the context to gate on - which is the
#   prediction the measurement actually makes.
#
#   8 runs.
#
# ============================================================================
# ARM W -- a different family: learn WHERE to sample, not what to add
# ============================================================================
#   {bilinear, zeroinit, ldm_warp} x seeds {42, 7}, fcos, 1024->384
#
#   The literature's successful learned downsamplers are non-uniform RESAMPLERS - they
#   move sampling locations rather than editing values at fixed ones (content-adaptive
#   resampler, Sun & Chen 2019; two-plane perspective prior, Ghosh et al. 2023, reporting
#   large small-object gains over naive downsampling). Talebi is not of that family.
#   WarpResizer predicts a smooth offset field into grid_sample, so pixel budget can be
#   spent non-uniformly. At init the grid is the identity, so it starts exactly at
#   bilinear (verified to 2.4e-07 on a ramp).
#
#   6 runs, of which bilinear and zeroinit are shared with Arm M's config only if the
#   target matches - they do not, so both are run here.
#
# ============================================================================
# ARM L -- aspect ratio, a defect independent of everything above
# ============================================================================
#   {bilinear, zeroinit} x {squash, letterbox} x seed 42, fcos, 1024->384
#
#   39% of NODE21 films are non-square natively (up to 3920x3612). The loader has always
#   resized straight to a square, distorting nodules by a per-image factor - a 3000x2000
#   film squashes a circular nodule to aspect 0.667. The NODE21 reference pipelines
#   preserve aspect and pad. Square films (61%, the 1024 resamples) are unaffected either
#   way, so this is a test on the non-square minority.
#
#   Run on BOTH variants: if letterboxing helps bilinear as much as the resizer, it is a
#   data-pipeline fix and not part of the resizer story.
#
#   4 runs.
#
# ============================================================================
# COST  M ~4h (8 x ~30min), W ~3h (6 x ~30min), L ~2h (4 x ~30min). All three ~9h.
#
# COMPARABILITY  fcos head, IoU matching, per-image sampling: comparable to the v7 fcos
#   runs and to nothing before v6. Arm L's letterbox cells are comparable only to each
#   other and to their own squash controls.
#
# USAGE
#   .\scripts\experiments\run_v8_adaptive_resizers.ps1 -DryRun
#   .\scripts\experiments\run_v8_adaptive_resizers.ps1                # M + W
#   .\scripts\experiments\run_v8_adaptive_resizers.ps1 -Arms MWL      # everything
#   .\scripts\experiments\run_v8_adaptive_resizers.ps1 -Arms M        # mechanism only
#   .\scripts\experiments\run_v8_adaptive_resizers.ps1 -Resume

param(
    [string] $Arms       = "MW",
    [int[]]  $Seeds      = @(42, 7),
    [string] $Head       = "fcos",
    [int]    $TargetSize = 384,
    [int]    $InputSize  = 1024,
    [string] $SelectOn   = "val_sens",
    [int]    $Patience   = 15,
    [string] $RunGroup   = "v8_adaptive_resizers",
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

# Micro-batch keyed on target; grad_accum holds the effective batch at 8 so LR stays 2e-4
# with no LR-scaling confound. Identical across every cell of an arm - the resizers carry
# BatchNorm, so a differing micro-batch would put a BN-statistics difference inside the
# comparison itself.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

$RUNS = @()
$SEEN = @{}

function Add-Run {
    param([string] $Arm, [string] $Variant, [int] $Seed, [bool] $Letterbox = $false)

    # Letterbox must be in the key: train.py encodes it in the output path as _lb, and two
    # runs differing only by it are genuinely different runs.
    $key = "$Variant|$Head|$InputSize|$TargetSize|$Seed|$Letterbox"

    if ($SEEN.ContainsKey($key)) {
        Write-Host "  (dedup) [$Arm] $Variant s$Seed lb=$Letterbox already queued by arm $($SEEN[$key])" -ForegroundColor DarkYellow
        return
    }

    $SEEN[$key] = $Arm
    $sched = $BATCH_FOR_TARGET[$TargetSize]

    $script:RUNS += @{
        arm = $Arm; variant = $Variant; seed = $Seed; letterbox = $Letterbox
        batch = $sched.batch; accum = $sched.accum
    }
}

# ---- Arm M: receptive field x selectivity, with the current architecture as control ----
if ($Arms -match "M") {
    foreach ($seed in $Seeds) {
        foreach ($variant in @("node21_talebi_zeroinit", "node21_ldm_dilated",
                               "node21_ldm_gated", "node21_ldm_dilated_gated")) {
            Add-Run -Arm "M" -Variant $variant -Seed $seed
        }
    }
}

# ---- Arm W: non-uniform resampling against both baselines ----
if ($Arms -match "W") {
    foreach ($seed in $Seeds) {
        foreach ($variant in @("bilinear", "node21_talebi_zeroinit", "node21_ldm_warp")) {
            Add-Run -Arm "W" -Variant $variant -Seed $seed
        }
    }
}

# ---- Arm L: aspect ratio, on both variants so a pipeline fix is distinguishable ----
if ($Arms -match "L") {
    foreach ($variant in @("bilinear", "node21_talebi_zeroinit")) {
        foreach ($letterbox in @($false, $true)) {
            Add-Run -Arm "L" -Variant $variant -Seed 42 -Letterbox $letterbox
        }
    }
}

$manifestObj = [ordered]@{
    run_group     = $RunGroup
    started       = $STAMP
    arms          = $Arms
    head          = $Head
    input_size    = $InputSize
    target_size   = $TargetSize
    seeds         = $Seeds
    select_on     = $SelectOn
    patience      = $Patience
    attribution   = $true
    evaluation    = "box IoU (0.25 primary, 0.5 reported), per-image sampling"
    comparable_to = "v7 fcos runs; nothing before v6"
    n_runs        = $RUNS.Count
    runs          = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v8 adaptive resizers -- $($RUNS.Count) runs | arms=$Arms head=$Head $InputSize->$TargetSize" -ForegroundColor Cyan
Write-Host "  every run records residual enrichment in metrics.json (--attribution)" -ForegroundColor Yellow
Write-Host "  READ ENRICHMENT FIRST: it measures whether the deficit was fixed." -ForegroundColor Yellow
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    $lb = if ($cfg.letterbox) { "letterbox" } else { "squash   " }
    Write-Host ("  [{0}] {1,-26} s{2,-5} {3} batch={4}/accum={5}" -f `
        $cfg.arm, $cfg.variant, $cfg.seed, $lb, $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
}
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run -- nothing executed. Manifest written to $MANIFEST" -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    param($Config)

    $resTag = "_i${InputSize}_t${TargetSize}"
    $lbTag  = if ($Config.letterbox) { "_lb" } else { "" }
    $leaf   = "$($Config.variant)_${Head}_focal_v5_s$($Config.seed)${resTag}${lbTag}"

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
    $lb = if ($cfg.letterbox) { " letterbox" } else { "" }
    $label = "[$($cfg.arm)] $($cfg.variant) s$($cfg.seed)$lb ($completed/$($RUNS.Count))"

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
        "--datasets",    "node21",
        "--input-size",  "$InputSize",
        "--target-size", "$TargetSize",
        "--batch-size",  "$($cfg.batch)",
        "--grad-accum",  "$($cfg.accum)",
        "--lr",          "2e-4",
        "--epochs",      "50",
        "--patience",    "$Patience",
        "--select-on",   $SelectOn,
        "--aug-profile", "v3",
        "--val-frac",    "0.20",
        "--run-group",   $RunGroup,
        "--augment",
        "--froc",
        "--attribution"
    )

    if ($cfg.letterbox) { $trainArgs += "--letterbox" }

    Invoke-Run -Label $label -TrainArgs $trainArgs

    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:hh\:mm}" -f $completed, $RUNS.Count, $elapsed) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v8 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Analyse with:" -ForegroundColor Yellow
Write-Host "  python scripts/eval/analyse_v8.py" -ForegroundColor DarkGray
