# run_v6_box_detection.ps1
# First sweep on the complete CenterNet detector (--head centernet_det).
#
# NOTHING FROM v1-v5b IS COMPARABLE TO THESE NUMBERS. Three things changed at once,
# deliberately, on 2026-08-19:
#
#   1. The model now predicts boxes. The earlier head emitted only CenterNet's centre
#      heatmap - one of its three branches - so it could not produce a box and IoU/AP
#      were not computable. Size and offset branches restored per Zhou et al. (2019).
#
#   2. Matching is IoU-based, not centre-distance. The old hit test was a constant 85.7
#      input-space pixels (15.0 mm / 0.175 mm/px), correct only at input_size=2048; at
#      224 it was ~38% of image width. IoU is a ratio of areas and carries no length
#      unit, so it is identical at every resolution - the confound is gone by construction.
#
#   3. Sampling is per-image, not per-annotation. Multi-nodule films previously appeared
#      once per nodule with the others labelled background - 342 of 1476 positives (23%)
#      carried a contradictory label. Train split goes 3764 rows -> 3591 images.
#
# Size bins are now derived from image width (CHEST_WIDTH_MM / input_size) rather than
# the .mha spacing headers: 61% of NODE21 films are 1024x1024 resamples whose headers
# were never updated and imply a 146 mm chest against 358 mm for natively-sized films.
# Under the old binning the "S" bin was 96% drawn from that resample subset, so it
# tracked image provenance rather than nodule size.
#
# ARMS
#
#   A  bilinear,               input=1024 -> target {224, 256, 384, 512}
#   B  node21_talebi_zeroinit, input=1024 -> target {224, 256, 384, 512}
#        The load-bearing pair. Identical input, identical backbone feature resolution,
#        the only difference is who performs the downsample. A is the matched control
#        for B - a B number alone compares to nothing.
#
#   C  node21_talebi_zeroinit, input {224, 256, 384, 512} -> target 224
#        INFORMATION-PRESERVATION test. Backbone compute is pinned at 224 while the
#        resizer is fed progressively more input. If the learned resizer preserves
#        detail that bilinear discards, more input should help - or at least not hurt.
#        Under the old centre-distance metric this arm declined monotonically, but that
#        metric grew ~9x more lenient as input resolution fell, so the trend was
#        probably an artifact. IoU matching is resolution-free, so this arm is
#        interpretable for the first time.
#
#   F  bilinear,               input {224, 256, 384, 512} -> target 224
#        MATCHED CONTROL FOR ARM C, and the reason C alone proves nothing. Raising the
#        resizer's input also changes how much the dataloader's cv2.resize destroyed
#        BEFORE the resizer ran, so C on its own cannot separate "the learned resizer
#        preserves information" from "more pixels help whoever downsamples". F is the
#        same line with bilinear in place of Talebi. The claim is the INTERACTION:
#        Talebi rising while bilinear does not.
#
#   E  bilinear,               input=target=X for X in {224, 256, 384, 512}
#        Optional no-downsample reference: the resizer is a near no-op, so this is the
#        backbone alone at resolution X. Useful as an upper anchor, not part of the
#        claim. (Letter E, not D, so run directories from the 2026-08-19 ABCD sweep
#        keep their original arm meaning in the manifests.)
#
#   Arm C at input=224 and input=1024 duplicate runs already produced by the ABCD sweep
#   (talebi 224->224 and talebi 1024->224). -Resume skips them.
#
# SCOPE ADVICE
#   Every number restarts tonight. A seeded answer on A/B beats a single-seed map of
#   configurations that cannot be told apart from noise: the binomial SE at n=147 test
#   nodules is +/-0.040, so any single-seed delta under ~0.08 is uninterpretable.
#   Default is A/B at two seeds (16 runs, ~7h at the ~25 min/run this config measured):
#   a seeded answer on the load-bearing pair. Use -Arms ABCD -Seeds 42 for the full 2x2
#   at one seed (16 runs, ~7h) when the structure matters more than the error bars.
#
# USAGE
#   .\scripts\experiments\run_v6_box_detection.ps1 -DryRun
#   .\scripts\experiments\run_v6_box_detection.ps1                      # A/B, seeds 42,7
#   .\scripts\experiments\run_v6_box_detection.ps1 -Arms C -Seeds 42 -Resume  # preservation arm only
#   .\scripts\experiments\run_v6_box_detection.ps1 -Resume

param(
    [int[]]  $Seeds      = @(42, 7),
    [string] $Arms       = "AB",
    [string] $SelectOn   = "val_sens",
    [int]    $Patience   = 15,
    [string] $AugProfile = "v3",
    [string] $RunGroup   = "v6_box_detection",
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

# Micro-batch keyed on target; grad_accum holds the effective batch at 8 everywhere so
# LR stays 2e-4 with no LR-scaling confound. Arms A and B MUST share a schedule at the
# same target - the Talebi resizer contains BatchNorm, so a different micro-batch would
# put a BN-statistics difference inside the exact comparison the sweep exists to make.
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

    $key = "$Variant|$InputRes|$TargetRes|$Seed"

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

    # A and B interleaved so each paired comparison completes together - an interrupted
    # night then leaves matched pairs rather than one full arm and half of another.
    foreach ($t in @(224, 256, 384, 512)) {
        if ($Arms -match "A") { Add-Run -Arm "A" -Variant "bilinear"               -InputRes 1024 -TargetRes $t -Seed $seed }
        if ($Arms -match "B") { Add-Run -Arm "B" -Variant "node21_talebi_zeroinit" -InputRes 1024 -TargetRes $t -Seed $seed }
    }

    # Arm C: input varies, target pinned at 224 - the information-preservation test
    if ($Arms -match "C") {
        foreach ($i in @(224, 256, 384, 512)) {
            Add-Run -Arm "C" -Variant "node21_talebi_zeroinit" -InputRes $i -TargetRes 224 -Seed $seed
        }
    }

    # Arm F: bilinear on Arm C's line - the matched control
    if ($Arms -match "F") {
        foreach ($i in @(224, 256, 384, 512)) {
            Add-Run -Arm "F" -Variant "bilinear" -InputRes $i -TargetRes 224 -Seed $seed
        }
    }

    # Arm E: no-downsample reference, input = target
    if ($Arms -match "E") {
        foreach ($t in @(224, 256, 384, 512)) {
            Add-Run -Arm "E" -Variant "bilinear" -InputRes $t -TargetRes $t -Seed $seed
        }
    }
}

$manifestObj = [ordered]@{
    run_group       = $RunGroup
    started         = $STAMP
    head            = "centernet_det"
    matching        = "box IoU (0.25 primary, 0.5 also reported)"
    sampling        = "per-image, all boxes"
    comparable_to   = "nothing before 2026-08-19"
    arms            = $Arms
    seeds           = $Seeds
    select_on       = $SelectOn
    patience        = $Patience
    aug_profile     = $AugProfile
    lr              = "2e-4"
    datasets        = "node21"
    backbone        = "convnext_fpn"
    effective_batch = 8
    n_runs          = $RUNS.Count
    runs            = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v6 box detection -- $($RUNS.Count) runs | arms=$Arms seeds=[$($Seeds -join ',')]" -ForegroundColor Cyan
Write-Host "  head=centernet_det  matching=IoU  sampling=per-image" -ForegroundColor DarkGray
Write-Host "  NOT comparable to any earlier run group" -ForegroundColor Yellow
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
    param($Config)

    $resTag = ""
    if (-not ($Config.input -eq 2048 -and $Config.target -eq 512)) {
        $resTag = "_i$($Config.input)_t$($Config.target)"
    }

    # head name is part of the output directory, so v6 runs never collide with v5b
    $leaf = "$($Config.variant)_centernet_det_focal_v5_s$($Config.seed)$resTag"

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
        "--head",        "centernet_det",
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
Write-Host "v6 box detection complete. Log: $LOG" -ForegroundColor Green
Write-Host "Each run's metrics.json carries box_metrics at IoU 0.25 and 0.5:" -ForegroundColor Yellow
Write-Host "  sensitivity, fp_per_image, average_precision, froc_auc, by_size (S/M/L)" -ForegroundColor DarkGray
Write-Host "froc_data.json is tagged criterion=box_iou_0.25 - do not mix with pre-v6 FROC files." -ForegroundColor DarkGray
