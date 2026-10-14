# run_v9_warp_resolution.ps1
# Resolution sweep for the warping resizer, against bilinear, on the FCOS head.
#
# WHY WARP, AND WHY THIS SWEEP
#   v8 (1024->384, 3 seeds) found that ldm_warp REPAIRS the residual-alignment deficit -
#   enrichment 0.765 -> 1.435, reversing it rather than merely reducing it - while
#   detection lands exactly on bilinear (+0.001 / -0.000 / +0.003 at 0.25/0.5/1.0 FP,
#   FROC 0.788 vs 0.790). That falsifies "the residual does not concentrate on nodules,
#   therefore it cannot help", but only at one resolution. This sweep asks whether the
#   tie, and the enrichment repair, hold across the downsampling range.
#
# WHY FCOS AND NOT CENTERNET
#   Two reasons, and the second is the decisive one.
#     1. FCOS is simply better here: 0.772 vs 0.626 (bilinear) and 0.738 vs 0.524
#        (talebi zeroinit) at 1024->512.
#     2. FCOS seed spread is 0.020 against CenterNet's 0.082. Every effect in this sweep
#        is 0.01-0.03, so under CenterNet none of it would be readable.
#   All existing warp data is FCOS, so this extends one series rather than starting a
#   third incomparable one. Note the bilinear CenterNet resolution sweep looks complete
#   but is s42-only at every resolution except 1024->512.
#
# ARMS
#   R1  ldm_warp,  input=1024 -> target {224, 256, 384, 512}
#         How the tie behaves as the downsampling factor grows. 1024->224 is the most
#         aggressive point and the one where a resizer has most to prove.
#
#   R2  ldm_warp,  input {224, 256, 384, 512} -> target 224
#         Backbone pinned at 224 while the resizer is fed more input. Under the corrected
#         IoU metric this arm is interpretable; under the pre-v6 centre-distance metric it
#         was not, because the hit tolerance grew ~9x more lenient as input resolution fell.
#
#   B   bilinear,  input=1024 -> target {224, 256, 384, 512}
#         The matched control. 1024->384 (3 seeds) and 1024->512 (2 seeds) already exist
#         and -Resume skips them, so in practice this arm costs two runs per seed.
#
#   Comparing against bilinear, not against talebi zeroinit. zeroinit sits BELOW bilinear,
#   so measuring warp against it once inflated an apparent gain that was really a return
#   to parity.
#
# COST per seed, roughly: R1 ~2.5h (25+28+35+60 min), R2 ~1.7h (4 x 25 min at target 224),
#   B ~1h (the two missing targets). About 5h for one seed, 10h for two.
#
#   Start with -Seeds 42 if the shape matters more than the error bars; the FCOS spread of
#   0.020 makes a single seed readable for effects above roughly 0.04, but not below.
#
# EVERY RUN CARRIES --attribution, so residual enrichment is recorded next to sensitivity.
#   Enrichment is the mechanism readout and is far more sensitive than detection here: it
#   moved 0.765 -> 1.435 in v8 while detection moved 0.000.
#
# READING IT  Report at MATCHED FALSE-POSITIVE BUDGET, never at a fixed score threshold.
#   FCOS scores sqrt(cls*centerness) and a fixed 0.3 threshold sits far right on the curve;
#   in v8 that made a flat-to-worse variant look like the winner.
#
# USAGE
#   .\scripts\experiments\run_v9_warp_resolution.ps1 -DryRun
#   .\scripts\experiments\run_v9_warp_resolution.ps1 -Seeds 42 -Resume
#   .\scripts\experiments\run_v9_warp_resolution.ps1 -Seeds 42,7 -Resume
#   .\scripts\experiments\run_v9_warp_resolution.ps1 -Arms R1 -Seeds 42 -Resume

param(
    [string] $Arms     = "R1R2B",
    [int[]]  $Seeds    = @(42),
    [string] $Head     = "fcos",
    [string] $SelectOn = "val_sens",
    [int]    $Patience = 15,
    [string] $RunGroup = "v9_warp_resolution",
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

# Micro-batch keyed on target, since the backbone dominates memory and always runs there.
# grad_accum holds the effective batch at 8 so LR stays 2e-4 across the sweep with no
# LR-scaling confound. warp and bilinear share the schedule at every target, which matters:
# an unequal micro-batch would put a BatchNorm-statistics difference inside the comparison.
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

    $key = "$Variant|$Head|$InputRes|$TargetRes|$Seed"

    if ($SEEN.ContainsKey($key)) {
        Write-Host "  (dedup) [$Arm] $Variant $InputRes->$TargetRes s$Seed already queued by arm $($SEEN[$key])" -ForegroundColor DarkYellow
        return
    }

    $SEEN[$key] = $Arm
    $sched = $BATCH_FOR_TARGET[$TargetRes]

    $script:RUNS += @{
        arm = $Arm; variant = $Variant; input = $InputRes; target = $TargetRes
        seed = $Seed; batch = $sched.batch; accum = $sched.accum
    }
}

foreach ($seed in $Seeds) {

    # B and R1 interleaved by target so each matched pair completes together - an
    # interrupted night then leaves usable pairs rather than one full arm and half of another
    foreach ($t in @(224, 256, 384, 512)) {
        if ($Arms -match "B")  { Add-Run -Arm "B"  -Variant "bilinear"        -InputRes 1024 -TargetRes $t -Seed $seed }
        if ($Arms -match "R1") { Add-Run -Arm "R1" -Variant "node21_ldm_warp" -InputRes 1024 -TargetRes $t -Seed $seed }
    }

    # R2: input varies, target pinned at 224
    if ($Arms -match "R2") {
        foreach ($i in @(224, 256, 384, 512)) {
            Add-Run -Arm "R2" -Variant "node21_ldm_warp" -InputRes $i -TargetRes 224 -Seed $seed
        }
    }
}

$manifestObj = [ordered]@{
    run_group     = $RunGroup
    started       = $STAMP
    arms          = $Arms
    head          = $Head
    seeds         = $Seeds
    select_on     = $SelectOn
    patience      = $Patience
    attribution   = $true
    baseline      = "bilinear (NOT talebi zeroinit - zeroinit sits below bilinear)"
    reporting     = "matched false-positive budget; never a fixed score threshold"
    evaluation    = "box IoU (0.25 primary, 0.5 reported), per-image sampling"
    comparable_to = "v7 and v8 fcos runs; nothing before v6"
    n_runs        = $RUNS.Count
    runs          = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v9 warp resolution -- $($RUNS.Count) runs | arms=$Arms head=$Head seeds=[$($Seeds -join ',')]" -ForegroundColor Cyan
Write-Host "  baseline is BILINEAR. Report at matched FP, not a fixed score threshold." -ForegroundColor Yellow
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    Write-Host ("  [{0,-2}] {1,-20} {2,4} -> {3,-4} s{4,-5} batch={5}/accum={6}" -f `
        $cfg.arm, $cfg.variant, $cfg.input, $cfg.target, $cfg.seed, $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
}
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run -- nothing executed. Manifest written to $MANIFEST" -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    <#
        Existing bilinear runs at 1024->384 and 1024->512 live in the v7 and v8 groups, so
        -Resume searches ALL fcos run groups rather than only this one. A config is a config;
        recipe defaults are identical across these groups.
    #>
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
Write-Host "v9 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Analyse with:" -ForegroundColor Yellow
Write-Host "  python scripts/eval/analyse_v9.py" -ForegroundColor DarkGray
