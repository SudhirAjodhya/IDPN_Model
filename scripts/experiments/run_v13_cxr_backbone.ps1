# run_v13_cxr_backbone.ps1
# Does a CHEST X-RAY pretrained encoder change the bilinear-vs-Talebi conclusion?
#
# THE QUESTION
#   Every result in this project so far uses an ImageNet-pretrained ConvNeXt-tiny, which
#   leaves a standing objection: natural-image filters reading grayscale radiographs. This
#   sweep repeats the core comparison on a DenseNet-121 pretrained on real chest
#   radiographs (TorchXRayVision), across the same four target resolutions.
#
#   Two things can happen and both are worth having:
#     - the learned resizer still ties or loses  -> the null survives a domain-matched
#       encoder, which is a much harder result to attack
#     - the ordering changes                     -> pretraining domain was a confound, and
#       that is a finding in itself
#
# THE CONFOUND, STATED
#   This changes architecture AND pretraining domain together: DenseNet-121 (~8M) against
#   ConvNeXt-tiny (~28M), CXR against ImageNet. A difference from the ConvNeXt runs cannot
#   be attributed to pretraining alone.
#
#   `densenet121_imagenet_fpn` is registered as the control that separates them - same
#   architecture, same parameter count, ImageNet weights. Run it with -Arms C once the main
#   sweep has landed if the CXR arm shows anything.
#
# WHY ZERO-INIT AND NOT RANDOM INIT
#   Zero-init makes the resizer start as exactly bilinear, so training can only move away
#   from a known-good point. That property is independent of the backbone. A random-init
#   resizer feeding a domain-matched encoder is if anything MORE disruptive, because those
#   features are more specialised to real radiographic structure. The measurement agrees:
#   at t384 on ConvNeXt, random init was -0.022 against bilinear, zero-init -0.004.
#
# RESOLUTION NOTE
#   The checkpoints are `densenet121-res224-*`, trained at 224x224. DenseNet is fully
#   convolutional so every multiple of 32 runs, but 224 is the exactly in-distribution
#   setting - and also where this project's absolute numbers are weakest. Sweeping all four
#   is the point: it shows whether domain-matched pretraining at low resolution can reach
#   ImageNet pretraining at high resolution.
#
# PREREQUISITE
#   python -m pip install torchxrayvision
#   python scripts/diagnostics/check_cxr_backbone.py     (shape check, downloads weights)
#
#   `python -m pip`, not `pip`: the venv was moved and its pip.exe launcher has a stale
#   absolute path baked in. The script sets PYTHONIOENCODING=utf-8 before any download,
#   because torchxrayvision's progress bar is not cp1252-encodable and a crash mid-download
#   leaves a truncated weights file that every later load fails on.
#
# COST
#   Unmeasured for this backbone. The full model is 10.5M parameters against ConvNeXt's
#   31.0M, but DenseNet's concatenation-heavy activations mean it is not simply 3x faster.
#   The ConvNeXt medians are an upper bound: 224=142, 256=107, 384=87, 512=68 min, so
#   8 runs <= ~13.5 h.
#   The script reports real per-run timings as it goes.
#
# RUN ORDER: 384, 224, 512, 256, with the two arms interleaved at each target.
#   384 first because the ConvNeXt reference there has 3 seeds, so it is the cleanest
#   backbone comparison available. 224 second because it is in-distribution for these
#   weights. Interleaving means an interrupted run leaves complete PAIRS.
#
# USAGE
#   .\scripts\experiments\run_v13_cxr_backbone.ps1 -DryRun
#   .\scripts\experiments\run_v13_cxr_backbone.ps1 -Resume
#   .\scripts\experiments\run_v13_cxr_backbone.ps1 -Arms C -Resume     # ImageNet control

param(
    [string] $Arms     = "X",           # X = CXR-pretrained, C = ImageNet DenseNet control
    [int]    $Seed     = 42,
    [int[]]  $Targets  = @(384, 224, 512, 256),
    [int]    $InputRes = 1024,          # NOT $Input: that is a PowerShell automatic variable
    [string] $Head     = "fcos",
    [string] $SelectOn = "val_sens",
    [int]    $Patience = 15,
    [string] $RunGroup = "v13_cxr_backbone",
    [switch] $DryRun,
    [switch] $Resume
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")

$BACKBONE = if ($Arms -eq "C") { "densenet121_imagenet_fpn" } else { "densenet121_cxr_fpn" }

$LOG_ROOT = Join-Path "outputs" $RunGroup
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }

$STAMP    = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG      = Join-Path $LOG_ROOT "run_$STAMP.log"
$MANIFEST = Join-Path $LOG_ROOT "manifest_$STAMP.json"

# Same schedule as v9-v12 so the recipe stays comparable across backbones.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

# ConvNeXt medians, used only for the up-front estimate. DenseNet timings are unmeasured.
$MINUTES_FOR_TARGET = @{ 224 = 142; 256 = 107; 384 = 87; 512 = 68 }

# Interleaved so each target completes as a matched pair
$RUNS = @()
foreach ($target in $Targets) {
    $sched = $BATCH_FOR_TARGET[$target]
    if (-not $sched) { throw "No batch schedule for target $target" }

    foreach ($variant in @("bilinear", "node21_talebi_zeroinit")) {
        $RUNS += @{
            variant = $variant; backbone = $BACKBONE
            input = $InputRes; target = $target; seed = $Seed
            batch = $sched.batch; accum = $sched.accum
            estimate_min = $MINUTES_FOR_TARGET[$target]
        }
    }
}

# Summed by hand: Measure-Object -Property reads OBJECT properties, and these are hashtables
$totalMinutes = 0
foreach ($run in $RUNS) { $totalMinutes += $run.estimate_min }

$manifestObj = [ordered]@{
    run_group      = $RunGroup
    started        = $STAMP
    backbone       = $BACKBONE
    head           = $Head
    seed           = $Seed
    targets        = $Targets
    select_on      = $SelectOn
    patience       = $Patience
    attribution    = $true
    datasets       = "node21"
    question       = "does a CXR-pretrained encoder change the bilinear vs Talebi conclusion?"
    confound       = "architecture and pretraining domain change together; densenet121_imagenet_fpn (-Arms C) separates them"
    input_channels = if ($Arms -eq "C") { "3, ImageNet normalised" } else { "1, xrv units - no 3-channel replicate" }
    reporting      = "matched false-positive budget; never a fixed score threshold"
    reference      = "ConvNeXt-tiny ImageNet runs at the same targets, v8/v11/v12"
    estimate_min   = $totalMinutes
    n_runs         = $RUNS.Count
    runs           = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v13 CXR backbone -- $($RUNS.Count) runs | backbone=$BACKBONE seed=$Seed" -ForegroundColor Cyan
Write-Host ("  upper-bound estimate {0:N0} min (~{1:N1} h) from ConvNeXt timings; DenseNet unmeasured" -f `
    $totalMinutes, ($totalMinutes / 60)) -ForegroundColor DarkGray
Write-Host "  reference: ConvNeXt-tiny ImageNet runs at the same targets. Report at matched FP." -ForegroundColor Yellow
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    Write-Host ("  {0,-24} {1,4} -> {2,-4} s{3,-4} batch={4}/accum={5}  ~{6} min" -f `
        $cfg.variant, $cfg.input, $cfg.target, $cfg.seed, $cfg.batch, $cfg.accum, $cfg.estimate_min) -ForegroundColor DarkGray
}
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run -- nothing executed. Manifest written to $MANIFEST" -ForegroundColor Yellow
    exit 0
}

function Get-ExistingRun {
    param($Config)

    # Backbone is part of the directory name for non-default backbones, so a CXR run never
    # collides with the ConvNeXt run of the same variant, seed and resolution.
    $leaf = "$($Config.variant)_${Head}_focal_v5_s$($Config.seed)_i$($Config.input)_t$($Config.target)_$($Config.backbone)"

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

    # python -u because piping to Tee-Object makes stdout a pipe, which CPython block-buffers.
    # ForEach-Object { "$_" } stringifies merged stderr so PowerShell 5.1 does not wrap those
    # lines in NativeCommandError.
    & python -u scripts/train.py @TrainArgs 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LOG -Append

    $ErrorActionPreference = $prev

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$Label failed with exit code $LASTEXITCODE, continuing" -ForegroundColor Red
        "$Label FAILED exit=$LASTEXITCODE" | Add-Content $LOG
    }
}

# torchxrayvision's download progress bar writes U+2588, which the Windows console
# (cp1252) cannot encode. Without this the FIRST download of any checkpoint crashes
# mid-stream and leaves a TRUNCATED weights file that later loads fail on.
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Checking the backbone loads before spending the night on it..." -ForegroundColor Yellow
& python scripts/diagnostics/check_cxr_backbone.py
if ($LASTEXITCODE -ne 0) {
    throw "Backbone check failed. Install with 'python -m pip install torchxrayvision' (note: python -m pip, the venv's pip.exe launcher has a stale path). If it reports 'unexpected EOF', delete the partial file under ~/.torchxrayvision/models_data/ and retry."
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

    $runStart = Get-Date

    Invoke-Run -Label $label -TrainArgs @(
        "--backbone",    $cfg.backbone,
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
    Write-Host ("[timing] {0}/{1} done, this run {2:N0} min, cumulative {3:hh\:mm}" -f `
        $completed, $RUNS.Count, ((Get-Date) - $runStart).TotalMinutes, $elapsed) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v13 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Compare against the ConvNeXt reference with:" -ForegroundColor Yellow
Write-Host "  python scripts/eval/plot_run_diagnostics.py --target 384" -ForegroundColor DarkGray
Write-Host "  python scripts/eval/plot_supervisor_figures.py" -ForegroundColor DarkGray
Write-Host "  NOTE: those scripts currently filter to ConvNeXt runs. They need a backbone" -ForegroundColor DarkGray
Write-Host "  axis added before the CXR runs will appear - ask before reading them." -ForegroundColor DarkGray
