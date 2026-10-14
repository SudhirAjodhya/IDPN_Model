# run_v15_selection_and_scale.ps1
# Resolve the v14 Arm N confound, then map the decimation-ratio axis properly.
#
# WHAT v14 LEFT UNRESOLVED
#   At VinDr input 2048 the learned resizer beat bilinear by +0.153 sensitivity at matched
#   0.25 FP/img (McNemar p=0.0003, 29 discordant nodules against 7). At VinDr input 1024
#   the same comparison was +0.006, and NODE21-only was -0.015. That is the strongest
#   signal this project has produced.
#
#   It is not yet safe to quote, for one specific reason: EVERY v14 run selected its
#   checkpoint on `val_sens` (sensitivity at a FIXED score threshold of 0.3) while being
#   REPORTED at a matched 0.25 FP/img budget. Those are different quantities, and the
#   epochs they favour diverge badly:
#
#       arm  variant   selected(val_sens)   best on matched FP
#       C    bilinear      12                   21
#       C    zeroinit       9                   15
#       V    bilinear      13                   23
#       V    zeroinit      17                   29
#       N    bilinear       6                   33     <-- 27 epochs early
#       N    zeroinit      11                   26
#
#   Arm N's BILINEAR baseline - the run that produces the headline gap - was checkpointed
#   at epoch 6 of 34, when its validation matched-FP sensitivity was 0.282 against 0.488
#   at epoch 33. So the gap may partly measure a bad checkpoint rather than a real effect.
#   Only best_model.pth is saved, so epoch 33 no longer exists and this CANNOT be settled
#   from what is on disk.
#
#   Note the two readings are entangled and both predict the same observation at n=1:
#     (a) artifact - bilinear's validation series was unstable, so the argmax landed early
#     (b) mechanism - mixed-scale training genuinely destabilises a parameterless resizer,
#         and the erratic selection is a CONSEQUENCE of that
#   Seeds plus a matched selection rule are what separate them.
#
# FIX APPLIED HERE
#   Every run uses --select-on val_sens_matched, so the selection criterion is the metric
#   being reported. This flag has existed all along and has never been used. It changes the
#   baseline too, so ALL arms move together or a new confound replaces the old one.
#
# WHY NODE21 STAYS AT INPUT 1024
#   Raising it is not possible cleanly. MEASURED over all 4,882 films: only 37% of NODE21
#   is 1024x1024; the rest runs to 4280x3520, median 2048x1864, 57% non-square. So there is
#   no square input_size that is native for it - at 2048 fifty-two percent of the TEST split
#   is UPSAMPLED, making the pipeline 1024 -> upsample 2048 -> downsample 224 for half the
#   evaluation set.
#
#   (NODE21 is commonly misremembered as uniformly 1024x1024. It only looks that way because
#   --input-size 1024 has been set in every run since v3.)
#
#   The v14 effect did not come from NODE21 being high-resolution - NODE21 was at 1024 in
#   every arm, including the winning one. Only VinDr's training input varied. VinDr's long
#   side is 2429 (p5) to 3408, median 2863, so it is the axis that can actually be raised,
#   and moving it leaves the fixed 517-film / 147-nodule evaluation untouched.
#
# ============================================================================
# ARM S -- selection fix plus seeds on the arm that carries the result
# ============================================================================
#   {bilinear, zeroinit} x seeds {42, 7, 123}, VinDr @ 2048, val_sens_matched.
#   This is the run that decides whether +0.153 is real. 6 runs.
#
# ARM C -- the same treatment for the NODE21-only control
#   {bilinear, zeroinit} x seeds, no VinDr, val_sens_matched. Required: a control selected
#   under the old rule cannot be compared against arms selected under the new one. 6 runs.
#
# ARM L -- decimation ladder on the axis that can be varied
#   {bilinear, zeroinit} x VinDr input {1024, 2700}, seed 42, val_sens_matched.
#   With Arm S at 2048 this gives 4.6x / 9.1x / 12.1x at fixed evaluation, which separates
#   "higher decimation helps" from "the 2048 point was a fluke". 4 runs.
#
#   2700 is VinDr's approximate native median, so it needs no upsampling. Its cache does
#   not exist yet and will not fit alongside the others - the dataset falls back to
#   uncached DICOM loading, which is acceptable because a high-resolution run is
#   compute-bound in the resizer's conv stack, not dataloader-bound.
#
# ============================================================================
# COST  Arm C ~3 h/run (18 h). Arm S ~10 h/run (60 h). Arm L: 1024 ~3 h x2, 2700 ~20 h x2
#       (46 h). Everything is ~124 h. DEFAULT IS "SC" AT SEED 42 ONLY (~26 h), which
#       resolves the selection confound before spending days on seeds.
#
# COMPARABILITY  NOT comparable to v14 or earlier on absolute numbers: the selection rule
#   changed. Compare v15 within itself. The v14 runs remain on disk for the
#   selection-rule comparison itself.
#
# USAGE
#   .\scripts\experiments\run_v15_selection_and_scale.ps1 -DryRun
#   .\scripts\experiments\run_v15_selection_and_scale.ps1                    # S+C, seed 42
#   .\scripts\experiments\run_v15_selection_and_scale.ps1 -Seeds 42,7,123    # add seeds
#   .\scripts\experiments\run_v15_selection_and_scale.ps1 -Arms SCL          # everything

param(
    [string]   $Arms             = "T",
    [int[]]    $Targets          = @(224, 384),
    [int[]]    $Seeds            = @(42),
    [string[]] $Variants         = @("bilinear", "node21_talebi_zeroinit"),
    [string] $Head               = "fcos",
    [int]    $TargetSize         = 224,
    [int]    $InputSize          = 1024,
    [int]    $VindrInputSize     = 2048,
    [int[]]  $LadderSizes        = @(1024, 2700),
    [double] $VindrNegativeRatio = 1.0,
    [string] $SelectOn           = "val_sens_matched",
    [int]    $MaxEpochs          = 50,
    [int]    $Patience           = 15,
    [string] $RunGroup           = "v15_selection_and_scale",
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

$VINDR_ROOT  = "D:/Downloads/vinbigdata-chest-xray-abnormalities-detection"
$invariant   = [cultureinfo]::InvariantCulture
$ratioText   = $VindrNegativeRatio.ToString($invariant)

if (-not (Test-Path (Join-Path $VINDR_ROOT "train"))) {
    throw "VinDr DICOM source missing at $VINDR_ROOT/train."
}

# Micro-batch keyed on the resizer's INPUT resolution, not the target. The Talebi conv
# stack runs at input resolution, so activation memory scales with it: measured 1.15 GB at
# 1024 batch 2 against 3.16 GB at 2700 batch 2. grad_accum holds the effective batch at 8
# so no LR-scaling confound enters.
function Get-Schedule {
    param([int] $ResizerInput)

    if ($ResizerInput -ge 2560) { return @{ batch = 2; accum = 4 } }
    if ($ResizerInput -ge 2048) { return @{ batch = 4; accum = 2 } }

    return @{ batch = 8; accum = 1 }
}

$RUNS = @()

function Add-Run {
    param([string] $Arm, [string] $Variant, [int] $Seed, [string] $Datasets,
          [int] $VindrInput = 0, [int] $Target = 0)

    if (-not $Target) { $Target = $TargetSize }

    # The largest resolution the resizer will see decides the memory schedule.
    $resizerInput = if ($VindrInput -gt $InputSize) { $VindrInput } else { $InputSize }
    $schedule = Get-Schedule -ResizerInput $resizerInput

    $script:RUNS += @{
        arm = $Arm; variant = $Variant; seed = $Seed; datasets = $Datasets
        vindr_input_size = $VindrInput; target = $Target
        batch = $schedule.batch; accum = $schedule.accum
    }
}

$VARIANTS = $Variants

# ---- Anti-aliasing must match across the arms in one run group ----
#   The learned resizer is skip(x) + residual(x), and its skip IS a classical downsample.
#   Anti-aliasing only the baseline leaves the learned arm the aliased one and reverses the
#   unfairness; anti-aliasing only the learned arm is the original unfairness. Either way
#   the measured gap stops being attributable to the learned residual.
#
#   Aliasing energy left by each classical kernel, decimating a 1024px chequerboard to 224
#   (scripts/eval/check_antialias_variants.py):
#       bilinear             std 0.16327
#       bilinear_antialias   std 0.00016   <- ~1000x less
#       bicubic              std 0.24284   <- WORSE than bilinear; sharper kernel folds more
#       bicubic_antialias    std 0.00002
#
#   So `bilinear` vs `node21_talebi_zeroinit_aa` would compare a learned resizer against a
#   baseline carrying 1000x the aliasing - a strawman. Refuse it rather than let it run.
$aaVariants    = @($VARIANTS | Where-Object { $_ -match "_aa$" -or $_ -match "^bicubic_antialias$" -or $_ -match "^bilinear_antialias$" })
$plainVariants = @($VARIANTS | Where-Object { $_ -eq "bilinear" -or $_ -eq "bicubic" -or $_ -eq "node21_talebi_zeroinit" -or $_ -eq "node21_talebi_zeroinit_attn" })

if ($aaVariants.Count -gt 0 -and $plainVariants.Count -gt 0) {
    throw ("Anti-aliasing mismatch in one run group.`n" +
           "  anti-aliased : $($aaVariants -join ', ')`n" +
           "  aliased      : $($plainVariants -join ', ')`n" +
           "Run them as SEPARATE run groups, or pair like with like:`n" +
           "  aliased  : -Variants bilinear,node21_talebi_zeroinit`n" +
           "  filtered : -Variants bilinear_antialias,node21_talebi_zeroinit_aa")
}

# ---- Arm T: what the resizer downsamples TO, swept. THE DEFAULT. ----
#   VinDr @ 2048 (the configuration that produced the v14 effect), NODE21 @ 1024, ONE
#   seed. Answers whether that effect is specific to target 224 or holds as the resizer is
#   asked to deliver more output resolution. One seed on purpose: this is a shape question,
#   not a significance question, and seeds come after a shape is worth resolving.
#
#   EVERY CELL IS SEPARATELY TRAINED, INCLUDING BILINEAR. That is not laziness about
#   reuse - it was measured on 2026-08-31. Taking the t224-trained checkpoints and simply
#   evaluating them at other targets (input fixed at 1024):
#
#       eval target   bilinear   zeroinit
#            224        0.438      0.591   <- both trained here
#            256        0.463      0.515
#            384        0.078      0.448
#            512        0.039      0.271
#
#   Bilinear collapses HARDER than the learned resizer, so it cannot be trained once and
#   reused across targets. Nothing about the target is baked into the resizer weights -
#   target_size appears only in two F.interpolate calls and a t224 checkpoint loads at any
#   target with zero missing keys - but the training does not transfer.
#
#   INPUT RESOLUTION IS HELD FIXED across the sweep for the same reason, in reverse. With
#   the target fixed at 224 and the input varied, bilinear is nearly immune (0.438 ->
#   0.426 at input 2048) while the learned resizer collapses (0.591 -> 0.046). Letting the
#   input move during an output sweep would put that on top of the effect being measured.
if ($Arms -match "T") {
    foreach ($target in $Targets) {
        foreach ($variant in $VARIANTS) {
            Add-Run -Arm "T" -Variant $variant -Seed $Seeds[0] -Datasets "node21_vindr" -VindrInput $VindrInputSize -Target $target
        }
    }
}

# ---- Arm S: the decisive arm. VinDr @ 2048, matched selection, seeds x targets. ----
#   Arm T established the shape with one seed: the learned resizer is ahead at all four
#   targets by +0.020 to +0.068, none of it individually significant. This arm replicates
#   it. Sweeps SEEDS crossed with TARGETS, so pass both -Seeds and -Targets:
#
#       .\...run_v15_selection_and_scale.ps1 -Arms S -Targets 224,384 -Seeds 42,7,123 -Resume
#
#   -Resume is what makes this affordable. The seed-42 cells at t224 and t384 are already
#   banked from arm T under this same run group, so the 12 nominal runs become 8 new ones.
#
#   ONLY SEEDS ARE VARIED, and that is now a measured decision rather than an assumption.
#   Training is bit-reproducible at a fixed seed: two 1-epoch runs of an identical command
#   returned train 2.0334 / val 0.7704 / sens@0.25FP 0.169 on both (2026-09-02). There is
#   NO rerun-noise component, so a seed is the only thing worth re-drawing, and the spread
#   across seeds is genuine seed effect rather than nondeterminism.
#
#   Corollary: DO NOT pick the best seed and report that. With no rerun noise the spread is
#   real, and selecting the top of it on the TEST set is the same class of error that
#   produced v14's +0.150 phantom. Report mean +/- spread over every seed run.
#
#   WHY ONLY TWO TARGETS. The test split holds 147 nodules, which floors the paired SE at
#   ~0.033 no matter how many seeds are run, so no single cell can reach significance and
#   seeding all four targets buys nothing extra. t224 (+0.048) and t384 (+0.068) are the
#   two largest effects; 384 is also where enrichment peaks. 8 new runs at ~4 h each.
#
#   NOTE num_workers. Worker count partitions sample indices and each worker holds its own
#   RNG stream, so the same seed at a different num_workers augments different films. It is
#   recorded in metrics.json from 2026-09-02 onward. Leave it at the default here: the
#   banked seed-42 cells this arm reuses were produced at that default.
if ($Arms -match "S") {
    foreach ($target in $Targets) {
        foreach ($seed in $Seeds) {
            foreach ($variant in $VARIANTS) {
                Add-Run -Arm "S" -Variant $variant -Seed $seed -Datasets "node21_vindr" `
                        -VindrInput $VindrInputSize -Target $target
            }
        }
    }
}

# ---- Arm C: NODE21-only control under the SAME selection rule ----
if ($Arms -match "C") {
    foreach ($seed in $Seeds) {
        foreach ($variant in $VARIANTS) {
            Add-Run -Arm "C" -Variant $variant -Seed $seed -Datasets "node21"
        }
    }
}

# ---- Arm L: decimation ladder, seed 42 only ----
if ($Arms -match "L") {
    foreach ($size in $LadderSizes) {
        foreach ($variant in $VARIANTS) {
            Add-Run -Arm "L" -Variant $variant -Seed 42 -Datasets "node21_vindr" `
                    -VindrInput $size
        }
    }
}

$manifestObj = [ordered]@{
    run_group            = $RunGroup
    started              = $STAMP
    arms                 = $Arms
    seeds                = $Seeds
    head                 = $Head
    input_size           = $InputSize
    targets              = $Targets
    target_size_default  = $TargetSize
    vindr_input_size     = $VindrInputSize
    ladder_sizes         = $LadderSizes
    vindr_negative_ratio = $VindrNegativeRatio
    select_on            = $SelectOn
    why                  = "v14 selected on val_sens at threshold 0.3 but reported at matched 0.25 FP; Arm N bilinear was checkpointed 27 epochs before its own matched-FP optimum"
    comparable_to        = "v15 internally ONLY - the selection rule changed"
    n_runs               = $RUNS.Count
    runs                 = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v15 selection + scale -- $($RUNS.Count) runs | arms=$Arms | select-on=$SelectOn" -ForegroundColor Cyan
Write-Host "  SELECTION RULE CHANGED: not comparable to v14 or earlier on absolute numbers." -ForegroundColor Yellow
Write-Host "  NODE21 stays at input $InputSize - raising it upsamples 52% of the test split." -ForegroundColor DarkGray
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

foreach ($cfg in $RUNS) {
    $res = if ($cfg.vindr_input_size) {
        $ratio = [math]::Round($cfg.vindr_input_size / $cfg.target, 1).ToString($invariant)
        "vindr@$($cfg.vindr_input_size) (${ratio}x)"
    } else {
        "node21 only"
    }

    Write-Host ("  [{0}] {1,-24} s{2,-5} t{3,-5} {4,-24} batch={5}/accum={6}" -f `
        $cfg.arm, $cfg.variant, $cfg.seed, $cfg.target, $res, $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
}
Write-Host ""

# Warn where a cache is missing, since that is the difference between ~3 h and ~10 h.
foreach ($size in ($RUNS | Where-Object { $_.vindr_input_size } |
                   ForEach-Object { $_.vindr_input_size } | Select-Object -Unique)) {
    $cacheDir = Join-Path $VINDR_ROOT "train_png16_cache_i$size"
    $cached = (Get-ChildItem -Path $cacheDir -Filter "*.npy" -ErrorAction SilentlyContinue).Count

    if ($cached -lt 8000) {
        # Parenthesised BEFORE -f: otherwise the operator binds only to the trailing
        # string literal and {0}/{1} print verbatim.
        Write-Host (("  [warn] no cache at VinDr input {0} ({1} films). Loading uncached at " +
                     "~300 ms/film. Prebuild if disk allows:") -f $size, $cached) -ForegroundColor Yellow
        Write-Host ("         python -m scripts.warm_vindr_cache --input-size {0} --negative-ratio {1}" -f `
                    $size, $ratioText) -ForegroundColor DarkGray
    } else {
        Write-Host "  [ok] cache at VinDr input $size holds $cached films" -ForegroundColor DarkGray
    }
}
Write-Host ""

if ($DryRun) {
    Write-Host "Dry run -- nothing executed. Manifest written to $MANIFEST" -ForegroundColor Yellow
    exit 0
}

function Get-CommitHeadroomGB {
    $counters = Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' -ErrorAction SilentlyContinue

    if (-not $counters) { return [double]::MaxValue }

    $committed = ($counters.CounterSamples | Where-Object Path -match 'Committed').CookedValue
    $limit     = ($counters.CounterSamples | Where-Object Path -match 'Limit').CookedValue

    return ($limit - $committed) / 1GB
}

function Get-ExistingRun {
    param($Config)

    # Must mirror train.py's _recipe_tag exactly or -Resume will accept the wrong run.
    $resTag = "_i${InputSize}_t$($Config.target)"
    $recipeTag = ""
    if ($Config.datasets -eq "node21_vindr") { $recipeTag += "_vindr" }
    if ($Config.vindr_input_size -and $Config.vindr_input_size -ne $InputSize) {
        $recipeTag += "_vin$($Config.vindr_input_size)"
    }
    if ($Config.datasets -eq "node21_vindr" -and $VindrNegativeRatio -gt 0) {
        $recipeTag += "_vnr" + $VindrNegativeRatio.ToString($invariant)
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

function Invoke-Run {
    param([string] $Label, $TrainArgs)

    Write-Host "=== $Label ===" -ForegroundColor Cyan
    "=== $Label ===" | Add-Content $LOG

    # A crashed run leaves orphaned DataLoader workers holding ~1.5 GB of commit each; 33
    # of them once held 56 GB of a 90 GB limit and killed every subsequent run with
    # WinError 1455. Snapshot so anything this run leaks can be reaped precisely.
    $preExisting = @(Get-Process python3.13 -ErrorAction SilentlyContinue |
                     Select-Object -ExpandProperty Id)

    $headroom = Get-CommitHeadroomGB

    if ($headroom -lt 8) {
        Write-Host ("  [warn] only {0:N1} GB commit headroom; expect WinError 1455" -f $headroom) -ForegroundColor Red
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

    $leaked = @(Get-Process python3.13 -ErrorAction SilentlyContinue |
                Where-Object { $preExisting -notcontains $_.Id })

    if ($leaked.Count -gt 0) {
        $leakedGB = ($leaked | Measure-Object PagedMemorySize64 -Sum).Sum / 1GB
        Write-Host ("  [cleanup] reaping {0} orphaned workers holding {1:N1} GB" -f `
                    $leaked.Count, $leakedGB) -ForegroundColor Yellow
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
    $label = "[$($cfg.arm)] $($cfg.variant) s$($cfg.seed) $($cfg.datasets)"

    if ($cfg.vindr_input_size) { $label += " vindr@$($cfg.vindr_input_size)" }

    $label += " ($completed/$($RUNS.Count))"

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
        "--target-size", "$($cfg.target)",
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
        $trainArgs += @("--vindr-negative-ratio", $ratioText)
    }

    Invoke-Run -Label $label -TrainArgs $trainArgs

    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:N1} h" -f `
                $completed, $RUNS.Count, $elapsed.TotalHours) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v15 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Read in this order:" -ForegroundColor Yellow
Write-Host "  1. paired_nodule_test.py on each arm -- McNemar, not the +-0.041 marginal SE" -ForegroundColor DarkGray
Write-Host "  2. does Arm S still show a gap once selection matches the reported metric?" -ForegroundColor DarkGray
Write-Host "  3. Arm L -- is the gap monotone in VinDr decimation ratio, or only at 2048?" -ForegroundColor DarkGray
