# run_v12_family_resolution_grid.ps1
# Fill the empty cells in the Talebi-family x resolution grid so the sweep chart can show
# every variant at every target.
#
# THE GRID BEFORE THIS SWEEP (FCOS, input 1024, seed count per cell)
#
#     variant                        224    256    384    512
#     bilinear                         1      1      3      2
#     node21_talebi                    -      -      2      -
#     node21_talebi_zeroinit           1      -      3      2
#     node21_talebi_zeroinit_attn      -      -      2      -
#
# This script fills the seven gaps at ONE seed each:
#     node21_talebi                 224, 256, 512
#     node21_talebi_zeroinit        256
#     node21_talebi_zeroinit_attn   224, 256, 512
#
# WHY SEED 42 AND WHY THAT MATTERS
#   Every existing single-seed cell (bilinear 224 and 256, zeroinit 224) is seed 42, so
#   using 42 here keeps those comparisons PAIRED - same split, same init draw, same
#   augmentation stream. A different seed would turn a paired comparison into an unpaired
#   one and add split variance on top of the effect being measured.
#
# COST, from the measured medians in outputs/mlflow/mlruns.db (fcos, 50-epoch runs)
#     target 224   142 min   (n=6, range 130-158)
#     target 256   107 min   (n=2, range 105-109)
#     target 384    87 min   (n=22, range 75-112)
#     target 512    68 min   (n=5, range 64-84)
#
#   Note the inversion: SMALLER targets cost MORE. They early-stop later because the
#   validation signal is noisier and convergence is slower at low resolution. Do not size
#   this sweep by assuming cost scales with resolution.
#
#   Total here: 512 (2 x 68) + 256 (3 x 107) + 224 (2 x 142) = 741 min, about 12.4 h.
#
# RUN ORDER IS DELIBERATE: 512, then 256, then 224.
#   512 is both the cheapest column and the most informative one - bilinear reaches 0.504
#   there, its best result anywhere, while zeroinit sits at 0.301. That -0.203 gap is the
#   largest effect in the project and is currently known for only one of three learned
#   variants. An interrupted night therefore buys the most valuable cells first.
#
# FLAGS THAT MUST MATCH v8/v11 OR THE RUNS ARE NOT COMPARABLE
#   --datasets node21   default is BOTH; JSRT exposure inflates NODE21 test sensitivity
#   --augment           store_true, default OFF
#   --attribution       store_true, default OFF - residual enrichment is the mechanism readout
#   --froc              store_true, default OFF
#   --lr 2e-4           the argparse default is 1e-4 and matches no existing run
#   --val-frac 0.20     the default is None
#
# READING IT  Matched false-positive budget, never a fixed score threshold. Compare against
#   BILINEAR, never against zeroinit - zeroinit sits below bilinear, so measuring against it
#   once inflated an apparent gain that was really a return to parity.
#
# AFTERWARDS
#   python scripts/eval/plot_supervisor_figures.py
#   Figures 1 and 4 pick the new cells up automatically.
#
# USAGE
#   .\scripts\experiments\run_v12_family_resolution_grid.ps1 -DryRun
#   .\scripts\experiments\run_v12_family_resolution_grid.ps1 -Resume
#   .\scripts\experiments\run_v12_family_resolution_grid.ps1 -Seed 7 -Resume

param(
    [int]    $Seed     = 42,
    [int]    $InputRes = 1024,          # NOT $Input: that is a PowerShell automatic variable
    [string] $Head     = "fcos",
    [string] $SelectOn = "val_sens",
    [int]    $Patience = 15,
    [string] $RunGroup = "v12_family_resolution_grid",
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

# Same schedule as v9-v11. The backbone dominates memory and always runs at target; holding
# the effective batch at 8 keeps LR at 2e-4 with no LR-scaling confound across the sweep.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

# Measured medians, for the up-front time estimate only
$MINUTES_FOR_TARGET = @{ 224 = 142; 256 = 107; 384 = 87; 512 = 68 }

# Exactly the seven empty cells, ordered 512 -> 256 -> 224 (see the header)
$PLAN = @(
    @{ target = 512; variants = @("node21_talebi", "node21_talebi_zeroinit_attn") },
    @{ target = 256; variants = @("node21_talebi", "node21_talebi_zeroinit",
                                  "node21_talebi_zeroinit_attn") },
    @{ target = 224; variants = @("node21_talebi", "node21_talebi_zeroinit_attn") }
)

$RUNS = @()
foreach ($column in $PLAN) {
    $sched = $BATCH_FOR_TARGET[$column.target]
    if (-not $sched) { throw "No batch schedule for target $($column.target)" }

    foreach ($variant in $column.variants) {
        $RUNS += @{
            variant = $variant; input = $InputRes; target = $column.target; seed = $Seed
            batch = $sched.batch; accum = $sched.accum
            estimate_min = $MINUTES_FOR_TARGET[$column.target]
        }
    }
}

# Summed by hand: Measure-Object -Property reads OBJECT properties, and $RUNS holds
# hashtables, whose entries are keys rather than properties. It fails rather than returning 0.
$totalMinutes = 0
foreach ($run in $RUNS) { $totalMinutes += $run.estimate_min }

$manifestObj = [ordered]@{
    run_group      = $RunGroup
    started        = $STAMP
    head           = $Head
    seed           = $Seed
    select_on      = $SelectOn
    patience       = $Patience
    attribution    = $true
    datasets       = "node21"
    purpose        = "fill the empty cells of the Talebi-family x resolution grid at one seed"
    seed_rationale = "42 matches every existing single-seed cell, keeping those comparisons paired"
    order          = "512 then 256 then 224: cheapest and most informative column first"
    baseline       = "bilinear at each target, already run - not repeated here"
    reporting      = "matched false-positive budget; never a fixed score threshold"
    comparable_to  = "v8, v9, v10 and v11 fcos runs; nothing before v6"
    estimate_min   = $totalMinutes
    n_runs         = $RUNS.Count
    runs           = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v12 family x resolution grid -- $($RUNS.Count) runs | head=$Head seed=$Seed" -ForegroundColor Cyan
Write-Host ("  estimate {0:N0} min, about {1:N1} h; finishing around {2}" -f `
    $totalMinutes, ($totalMinutes / 60), (Get-Date).AddMinutes($totalMinutes).ToString("HH:mm")) -ForegroundColor DarkGray
Write-Host "  baseline is BILINEAR at each target, already run. Report at matched FP." -ForegroundColor Yellow
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    Write-Host ("  {0,-30} {1,4} -> {2,-4} s{3,-4} batch={4}/accum={5}  ~{6} min" -f `
        $cfg.variant, $cfg.input, $cfg.target, $cfg.seed, $cfg.batch, $cfg.accum, $cfg.estimate_min) -ForegroundColor DarkGray
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

    # python -u because piping to Tee-Object makes stdout a pipe, which CPython then
    # block-buffers. ForEach-Object { "$_" } stringifies the merged stderr so PowerShell 5.1
    # does not wrap those lines in NativeCommandError.
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
Write-Host "v12 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Regenerate the figures with:" -ForegroundColor Yellow
Write-Host "  python scripts/eval/plot_supervisor_figures.py" -ForegroundColor DarkGray
Write-Host "  every cell will be single-seed outside target 384 - the figures label this," -ForegroundColor DarkGray
Write-Host "  and FCOS seed spread is 0.020, so treat differences under ~0.04 as unresolved." -ForegroundColor DarkGray
