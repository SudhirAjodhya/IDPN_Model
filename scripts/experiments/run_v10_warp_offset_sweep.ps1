# run_v10_warp_offset_sweep.ps1
# Does the warp resizer's null come from the displacement bound being too tight?
#
# WHY THIS SWEEP
#   node21_ldm_warp caps its sampling displacement at max_offset=0.05, which in
#   grid_sample's [-1,1] convention is 2.5% of image width - 9.6 target cells at t384,
#   against a median nodule of 19.1 cells. That bound has never been varied.
#
#   scripts/diagnostics/warp_label_displacement.py measured the trained offset head
#   reaching 8.34 of that 9.6 bound, so it is roughly 87% saturated. The constraint
#   plausibly binds, and Jin et al. (ICLR 2022), whose deformation module is
#   architecturally close to WarpResizer, allow far more aggressive magnification.
#
#   node21_ldm_warp_off15 and _off30 are identical to node21_ldm_warp in every respect
#   except max_offset (0.15 and 0.30). All three start at exactly the identity grid, so
#   they stay comparable with each other and with bilinear.
#
# THE GATE - DO NOT SKIP
#   fcos_targets.py:71 places supervision by boxes * (heatmap/input), a pure linear
#   rescale. That is exact for bilinear and the additive-residual family. For a warp it is
#   only an approximation, valid while label displacement is small relative to nodule size.
#   Measured at max_offset=0.05 it is 5-11% of nodule diameter, which is fine. Displacement
#   scales with the bound: at 0.30 the bound is 3x a nodule diameter and the linear rule is
#   EXPECTED to break.
#
#   After this sweep, before reading any detection number:
#       python scripts/diagnostics/warp_label_displacement.py --all
#   If displacement/diameter exceeds ~0.25 for a variant, its sensitivity is not
#   interpretable until FCOS targets are built through grid[i,j] instead of boxes * scale.
#   That outcome is a finding, not a failure.
#
# BASELINE
#   bilinear at 1024->384 already has 3 seeds (v8). Not re-run here. Compare against
#   bilinear, never against talebi zeroinit - zeroinit sits BELOW bilinear, and measuring
#   against it once inflated an apparent gain that was really a return to parity.
#
# FLAGS THAT MUST MATCH v8/v9 OR THE RUNS ARE NOT COMPARABLE
#   --datasets node21   the default is BOTH, and JSRT training exposure inflates NODE21
#                       test sensitivity. All work from v3 onward is node21-only.
#   --augment           store_true, default OFF
#   --attribution       store_true, default OFF - without it residual enrichment, the most
#                       sensitive instrument in this project, is never recorded
#   --froc              store_true, default OFF
#   --lr 2e-4           the argparse default is 1e-4 and matches no existing run
#   --val-frac 0.20     the default is None
#
# COST  4 runs at roughly 85 min each, about 5.7 hours. Measured from the 14 v8 t384 fcos
#   runs in outputs/mlflow/mlruns.db, which ran 75-105 min for 21-26 epochs.
#
# READING IT  Report at MATCHED FALSE-POSITIVE BUDGET, never at a fixed score threshold.
#   At threshold 0.3, node21_ldm_gated read +0.031 against bilinear and node21_ldm_warp
#   read +0.020; at 0.25 FP/img both are negative. The entire difference was the operating
#   point.
#
# USAGE
#   .\scripts\experiments\run_v10_warp_offset_sweep.ps1 -DryRun
#   .\scripts\experiments\run_v10_warp_offset_sweep.ps1 -Seeds 42,7 -Resume
#   .\scripts\experiments\run_v10_warp_offset_sweep.ps1 -Variants node21_ldm_warp_off15 -Seeds 42

param(
    [string[]] $Variants = @("node21_ldm_warp_off15", "node21_ldm_warp_off30"),
    [int[]]    $Seeds    = @(42, 7),
    [int]      $Target   = 384,
    # NOT $Input: that is a PowerShell automatic variable holding the pipeline-input
    # enumerator, and declaring it as [int] makes parameter binding fail before the script
    # body runs, with "Cannot convert ArrayListEnumeratorSimple to System.Int32".
    [int]      $InputRes = 1024,
    [string]   $Head     = "fcos",
    [string]   $SelectOn = "val_sens",
    [int]      $Patience = 15,
    [string]   $RunGroup = "v10_warp_offset_sweep",
    [switch]   $DryRun,
    [switch]   $Resume
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$LOG_ROOT = Join-Path "outputs" $RunGroup
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }

$STAMP    = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG      = Join-Path $LOG_ROOT "run_$STAMP.log"
$MANIFEST = Join-Path $LOG_ROOT "manifest_$STAMP.json"

# Matches the v9 schedule at this target: the backbone dominates memory and always runs at
# target, and holding the effective batch at 8 keeps LR at 2e-4 with no LR-scaling confound.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

$sched = $BATCH_FOR_TARGET[$Target]
if (-not $sched) { throw "No batch schedule for target $Target" }

# Seed-major ordering so an interrupted night leaves complete variant pairs at one seed,
# rather than one variant at both seeds and nothing to compare it against.
$RUNS = @()
foreach ($seed in $Seeds) {
    foreach ($variant in $Variants) {
        $RUNS += @{
            variant = $variant; input = $InputRes; target = $Target
            seed = $seed; batch = $sched.batch; accum = $sched.accum
        }
    }
}

$manifestObj = [ordered]@{
    run_group     = $RunGroup
    started       = $STAMP
    head          = $Head
    variants      = $Variants
    seeds         = $Seeds
    select_on     = $SelectOn
    patience      = $Patience
    attribution   = $true
    datasets      = "node21"
    question      = "is the warp null caused by max_offset=0.05 being too tight a bound?"
    baseline      = "bilinear 1024->384, 3 seeds, already in v8 - not re-run here"
    gate          = "run scripts/diagnostics/warp_label_displacement.py --all BEFORE reading detection numbers"
    reporting     = "matched false-positive budget; never a fixed score threshold"
    comparable_to = "v8 and v9 fcos runs; nothing before v6"
    n_runs        = $RUNS.Count
    runs          = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v10 warp offset sweep -- $($RUNS.Count) runs | head=$Head seeds=[$($Seeds -join ',')]" -ForegroundColor Cyan
Write-Host "  baseline is BILINEAR 1024->384 (3 seeds, v8). Report at matched FP." -ForegroundColor Yellow
Write-Host "  GATE: run warp_label_displacement.py --all before reading detection numbers." -ForegroundColor Yellow
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    Write-Host ("  {0,-24} {1,4} -> {2,-4} s{3,-5} batch={4}/accum={5}" -f `
        $cfg.variant, $cfg.input, $cfg.target, $cfg.seed, $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
}
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run -- nothing executed. Manifest written to $MANIFEST" -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    param($Config)

    $leaf = "$($Config.variant)_${Head}_focal_v5_s$($Config.seed)_i$($Config.input)_t$($Config.target)"

    return Get-ChildItem -Path "outputs" -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq $leaf -and
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

    # python -u because piping to Tee-Object makes stdout a pipe, and CPython then
    # block-buffers it. ForEach-Object { "$_" } stringifies the merged stderr, so
    # PowerShell 5.1 does not wrap those lines in NativeCommandError.
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
    $label = "$($cfg.variant) $($cfg.input)->$($cfg.target) s$($cfg.seed) ($completed/$($RUNS.Count))"

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
        "--head",        $Head,
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
        "--aug-profile", "v3",
        "--val-frac",    "0.20",
        "--run-group",   $RunGroup,
        "--augment",
        "--froc",
        "--attribution"
    )

    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:hh\:mm}" -f $completed, $RUNS.Count, $elapsed) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v10 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Next, IN THIS ORDER:" -ForegroundColor Yellow
Write-Host "  python scripts/diagnostics/warp_label_displacement.py --all" -ForegroundColor DarkGray
Write-Host "     gate: if displacement/diameter > ~0.25, detection numbers are not usable" -ForegroundColor DarkGray
Write-Host "  python scripts/eval/analyse_size_resolution.py" -ForegroundColor DarkGray
Write-Host "     new runs write froc_by_size directly, so no backfill is needed" -ForegroundColor DarkGray
