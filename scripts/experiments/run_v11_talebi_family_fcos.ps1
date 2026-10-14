# run_v11_talebi_family_fcos.ps1
# Fill the two FCOS gaps in the Talebi family so the whole family can go on one chart.
#
# THE GAP THIS CLOSES
#   As of 2026-08-24 the supervisor figure can only show bilinear and
#   node21_talebi_zeroinit, because those are the only Talebi-family variants with FCOS
#   runs. The other two are stranded:
#
#     node21_talebi_zeroinit_attn  exists only on the centernet_det head, so it has to be
#       shown on a separate chart with the caveat that FCOS beats centernet_det by +0.146
#       (bilinear) and +0.214 (Talebi) at 1024->512, and that centernet_det's seed spread
#       is 0.082 against FCOS's 0.020 - larger than every difference on that chart.
#
#     node21_talebi  (random init) has NO usable runs at all. Its only runs use the
#       `centernet` head with no box metrics, which places them before v6 and therefore
#       inside the 85.7 px hit-tolerance era (trap 4.3) - 38% of image width at target
#       224. Those numbers cannot be shown to anyone.
#
#   Arm A alone lets the attention variant join the main figure. Arm T additionally
#   restores the zero-init vs random-init ablation, which is the comparison that motivated
#   zeroinit in the first place and is currently unsupportable on this evaluation.
#
# ARMS
#   A  node21_talebi_zeroinit_attn  1024->384    the ask: attention on the current head
#   T  node21_talebi                1024->384    random init, to recover the init ablation
#
#   The baseline (bilinear) and node21_talebi_zeroinit already have 3 seeds each at
#   1024->384 from v8 and are NOT re-run. -Resume skips anything already complete.
#
# COST  ~85 min per run, measured from the 14 v8 t384 fcos runs in outputs/mlflow/mlruns.db
#   (75-105 min for 21-26 epochs).
#     -Arms A  -Seeds 42,7   2 runs, ~2.8 h
#     -Arms AT -Seeds 42,7   4 runs, ~5.7 h
#
# FLAGS THAT MUST MATCH v8 OR THE RUNS ARE NOT COMPARABLE
#   --datasets node21   the default is BOTH, and JSRT training exposure inflates NODE21
#                       test sensitivity. All work from v3 onward is node21-only.
#   --augment           store_true, default OFF
#   --attribution       store_true, default OFF - without it residual enrichment is never
#                       recorded, and for this family that is the whole explanation of the
#                       result (zeroinit sits at 0.765: the residual AVOIDS nodules)
#   --froc              store_true, default OFF
#   --lr 2e-4           the argparse default is 1e-4 and matches no existing run
#   --val-frac 0.20     the default is None
#
# READING IT  Report at MATCHED FALSE-POSITIVE BUDGET, never at a fixed score threshold.
#   Compare against BILINEAR, not against zeroinit - zeroinit sits below bilinear, so
#   measuring a variant against it once inflated an apparent gain that was really a return
#   to parity.
#
# AFTERWARDS
#   python scripts/eval/plot_supervisor_figures.py
#   The new runs are picked up automatically and the attention variant moves out of
#   figure 3 and into figure 1. Update that script's docstring once it does.
#
# USAGE
#   .\scripts\experiments\run_v11_talebi_family_fcos.ps1 -DryRun
#   .\scripts\experiments\run_v11_talebi_family_fcos.ps1 -Seeds 42,7 -Resume
#   .\scripts\experiments\run_v11_talebi_family_fcos.ps1 -Arms AT -Seeds 42,7 -Resume

param(
    [string] $Arms     = "A",
    [int[]]  $Seeds    = @(42, 7),
    [int]    $Target   = 384,
    [int]    $InputRes = 1024,           # NOT $Input: that is a PowerShell automatic variable
    [string] $Head     = "fcos",
    [string] $SelectOn = "val_sens",
    [int]    $Patience = 15,
    [string] $RunGroup = "v11_talebi_family_fcos",
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

# Same schedule as v9/v10: the backbone dominates memory and always runs at target, and
# holding the effective batch at 8 keeps LR at 2e-4 with no LR-scaling confound.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

$sched = $BATCH_FOR_TARGET[$Target]
if (-not $sched) { throw "No batch schedule for target $Target" }

# Seed-major so an interrupted night leaves complete arms at one seed rather than one arm
# at both seeds with nothing to pair it against.
$RUNS = @()
foreach ($seed in $Seeds) {
    if ($Arms -match "A") {
        $RUNS += @{ arm = "A"; variant = "node21_talebi_zeroinit_attn"
                    input = $InputRes; target = $Target; seed = $seed
                    batch = $sched.batch; accum = $sched.accum }
    }
    if ($Arms -match "T") {
        $RUNS += @{ arm = "T"; variant = "node21_talebi"
                    input = $InputRes; target = $Target; seed = $seed
                    batch = $sched.batch; accum = $sched.accum }
    }
}

if ($RUNS.Count -eq 0) { throw "No runs queued. -Arms must contain A and/or T, got '$Arms'." }

$manifestObj = [ordered]@{
    run_group     = $RunGroup
    started       = $STAMP
    head          = $Head
    arms          = $Arms
    seeds         = $Seeds
    select_on     = $SelectOn
    patience      = $Patience
    attribution   = $true
    datasets      = "node21"
    purpose       = "give the Talebi family FCOS runs so all four variants share one chart"
    baseline      = "bilinear 1024->384, 3 seeds, already in v8 - not re-run here"
    reporting     = "matched false-positive budget; never a fixed score threshold"
    comparable_to = "v8, v9 and v10 fcos runs; nothing before v6"
    n_runs        = $RUNS.Count
    runs          = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v11 Talebi family on FCOS -- $($RUNS.Count) runs | arms=$Arms seeds=[$($Seeds -join ',')]" -ForegroundColor Cyan
Write-Host "  baseline is BILINEAR 1024->384 (3 seeds, v8), not re-run. Report at matched FP." -ForegroundColor Yellow
Write-Host "  estimate ~85 min per run, about $([math]::Round($RUNS.Count * 85 / 60, 1)) h total" -ForegroundColor DarkGray
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    Write-Host ("  [{0}] {1,-30} {2,4} -> {3,-4} s{4,-5} batch={5}/accum={6}" -f `
        $cfg.arm, $cfg.variant, $cfg.input, $cfg.target, $cfg.seed, $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
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
Write-Host "v11 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Regenerate the figures with:" -ForegroundColor Yellow
Write-Host "  python scripts/eval/plot_supervisor_figures.py" -ForegroundColor DarkGray
Write-Host "  the attention variant should now appear in figure 1 rather than figure 3" -ForegroundColor DarkGray
