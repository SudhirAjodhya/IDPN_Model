# run_v7_recipe_and_fcos.ps1
# Two questions, one night.
#
# Everything here uses the v6 evaluation: box-predicting head, IoU matching, per-image
# sampling, width-derived size bins. Comparable to the v6 run groups, NOT to v1-v5b.
#
# ============================================================================
# ARM V -- does freezing the backbone reduce SEED VARIANCE?
# ============================================================================
#   node21_talebi_zeroinit, 1024 -> 224, centernet_det, seeds {42, 7, 123},
#   freeze-backbone-epochs 0 vs 3.  6 runs.
#
#   This is the highest-leverage question left. Measured seed spread at a single fixed
#   config is 0.109 (talebi 224->224: 0.381 at s42, 0.490 at s7) - larger than every
#   Talebi-vs-bilinear effect this project has measured. Nothing can be concluded until
#   that shrinks or is accepted as the floor.
#
#   Rationale for the intervention: the detection head starts randomly initialised, so
#   its first gradients are noise, and nothing holds the backbone still while that noise
#   is written into ImageNet features worth ~0.40 sensitivity here (pretrained resnet50
#   0.578 vs scratch 0.177). How much damage that does depends on the random draw, which
#   is a plausible source of run-to-run spread.
#
#   READ IT AS: the spread across seeds within each freeze setting, NOT the mean. A
#   higher mean with the same spread does not help; a lower spread does, even at an
#   unchanged mean.
#
# ============================================================================
# ARM I -- does DENSE box supervision help the learned resizer specifically?
# ============================================================================
#   {bilinear, node21_talebi_zeroinit} x {centernet_det, fcos}, 1024 -> 512,
#   seeds {42, 7}.  8 runs.
#
#   CenterNet supervises box regression at ONE cell per nodule; FCOS supervises every
#   cell inside the box. Measured on NODE21 that is 6x more supervised cells at
#   target=224 but 32x at target=512 - so this arm runs at 512, where the manipulation
#   is strongest. Testing it at 224 would be testing the hypothesis where it has least
#   to offer.
#
#   THE CLAIM IS THE INTERACTION, not the head main effect:
#     (fcos - centernet) for Talebi, MINUS (fcos - centernet) for bilinear.
#   If dense supervision lifts both equally it is a better detector recipe and says
#   nothing about learned downsampling. Only a larger lift for Talebi supports the
#   argument that the resizer was gradient-starved.
#
#   freeze-backbone-epochs is fixed at 0 across this arm so it stays comparable to the
#   existing v6 runs at 1024->512. Arm V decides whether freezing is worth adopting; it
#   must not be varied inside the arm that tests something else.
#
# ============================================================================
# ARM T -- the attention-subtraction variant, re-tested under a valid evaluation
# ============================================================================
#   {node21_talebi_zeroinit, node21_talebi_zeroinit_attn}, 1024 -> 384,
#   centernet_det, seeds {42, 7}.  4 runs.
#
#   The attention branch runs window + shifted-window attention at attn_res and is
#   SUBTRACTED from the conv branch before output_conv: the idea is that attention
#   captures globally-consistent anatomy, so removing it leaves nodule-scale detail in
#   the residual. Every number ever recorded for this variant comes from the invalidated
#   era - centre-distance matching, per-annotation labels, heatmap-only head - so it has
#   never actually been evaluated.
#
#   Run at target 384 rather than 224: the subtraction is meant to preserve fine detail,
#   and at 224 the heatmap is 56x56 with a ~2.5-cell nodule, too coarse to show it.
#
# ============================================================================
# OPTIONAL ARMS (off by default)
# ============================================================================
#   G  aug-profile v4 disambiguation, 1 run. Still open from 2026-08-18: the stronger
#      augmentation dropped FROC AUC 0.803 -> 0.767 but was confounded with --pos-frac,
#      so neither has been cleared. This is aug v4 with NO pos-frac against the v3
#      baseline at the same config.
#   L  backbone-lr-factor {0.01, 0.03} against the default 0.1, 2 runs. Tests whether
#      the pretrained backbone should move even more slowly, a weaker version of Arm V's
#      intervention.
#
# COST: V ~2.5h (6 x ~25min), I ~4.5h (8 x ~35min at target 512), T ~2h (4 x ~30min),
#   G ~25min, L ~50min.  V + I is roughly 7h; V + I + T is roughly 9h.
#
# USAGE
#   .\scripts\experiments\run_v7_recipe_and_fcos.ps1 -DryRun
#   .\scripts\experiments\run_v7_recipe_and_fcos.ps1                 # V + I
#   .\scripts\experiments\run_v7_recipe_and_fcos.ps1 -Arms VIGL      # everything
#   .\scripts\experiments\run_v7_recipe_and_fcos.ps1 -Resume

param(
    [string] $Arms       = "VI",
    [string] $SelectOn   = "val_sens",
    [int]    $Patience   = 15,
    [string] $AugProfile = "v3",
    [string] $RunGroup   = "v7_recipe_and_fcos",
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

# Micro-batch keyed on target; grad_accum holds the effective batch at 8 so LR stays
# 2e-4 everywhere. Within any arm the schedule is identical across cells - the Talebi
# resizer contains BatchNorm, so a differing micro-batch would put a BN-statistics
# difference inside the comparison itself.
$BATCH_FOR_TARGET = @{
    224 = @{ batch = 8; accum = 1 }
    256 = @{ batch = 8; accum = 1 }
    384 = @{ batch = 8; accum = 1 }
    512 = @{ batch = 4; accum = 2 }
}

$RUNS = @()
$SEEN = @{}

function Add-Run {
    param(
        [string] $Arm, [string] $Variant, [string] $Head,
        [int] $InputRes, [int] $TargetRes, [int] $Seed,
        [int] $Freeze = 0, [string] $Aug = "", [double] $BackboneLrFactor = 0.1
    )

    if (-not $Aug) { $Aug = $AugProfile }

    # Every knob that changes the run must be in the key, or two different runs collide
    $key = "$Variant|$Head|$InputRes|$TargetRes|$Seed|$Freeze|$Aug|$BackboneLrFactor"

    if ($SEEN.ContainsKey($key)) {
        Write-Host "  (dedup) [$Arm] $Variant/$Head $InputRes->$TargetRes s$Seed already queued by arm $($SEEN[$key])" -ForegroundColor DarkYellow
        return
    }

    $SEEN[$key] = $Arm
    $sched = $BATCH_FOR_TARGET[$TargetRes]

    $script:RUNS += @{
        arm = $Arm; variant = $Variant; head = $Head
        input = $InputRes; target = $TargetRes; seed = $Seed
        freeze = $Freeze; aug = $Aug; backbone_lr_factor = $BackboneLrFactor
        batch = $sched.batch; accum = $sched.accum
    }
}

# ---- Arm V: freeze on/off across three seeds, everything else held fixed ----
if ($Arms -match "V") {
    foreach ($seed in @(42, 7, 123)) {
        foreach ($freeze in @(0, 3)) {
            Add-Run -Arm "V" -Variant "node21_talebi_zeroinit" -Head "centernet_det" `
                    -InputRes 1024 -TargetRes 224 -Seed $seed -Freeze $freeze
        }
    }
}

# ---- Arm I: downsampler x head at target 512, where FCOS has 32x the supervision ----
if ($Arms -match "I") {
    foreach ($seed in @(42, 7)) {
        foreach ($variant in @("bilinear", "node21_talebi_zeroinit")) {
            foreach ($head in @("centernet_det", "fcos")) {
                Add-Run -Arm "I" -Variant $variant -Head $head `
                        -InputRes 1024 -TargetRes 512 -Seed $seed -Freeze 0
            }
        }
    }
}

# ---- Arm T: attention-subtraction variant against its own zeroinit baseline ----
if ($Arms -match "T") {
    foreach ($seed in @(42, 7)) {
        foreach ($variant in @("node21_talebi_zeroinit", "node21_talebi_zeroinit_attn")) {
            Add-Run -Arm "T" -Variant $variant -Head "centernet_det" `
                    -InputRes 1024 -TargetRes 384 -Seed $seed -Freeze 0
        }
    }
}

# ---- Arm G: aug v4 with no pos-frac, against the v3 baseline at the same config ----
if ($Arms -match "G") {
    Add-Run -Arm "G" -Variant "node21_talebi_zeroinit" -Head "centernet_det" `
            -InputRes 1024 -TargetRes 224 -Seed 42 -Freeze 0 -Aug "v4"
}

# ---- Arm L: does the pretrained backbone want an even lower learning rate? ----
if ($Arms -match "L") {
    foreach ($factor in @(0.01, 0.03)) {
        Add-Run -Arm "L" -Variant "node21_talebi_zeroinit" -Head "centernet_det" `
                -InputRes 1024 -TargetRes 224 -Seed 42 -Freeze 0 -BackboneLrFactor $factor
    }
}

$manifestObj = [ordered]@{
    run_group     = $RunGroup
    started       = $STAMP
    arms          = $Arms
    evaluation    = "box IoU (0.25 primary, 0.5 reported), per-image sampling"
    comparable_to = "v6_box_detection run groups only"
    select_on     = $SelectOn
    patience      = $Patience
    lr            = "2e-4"
    datasets      = "node21"
    backbone      = "convnext_fpn"
    n_runs        = $RUNS.Count
    runs          = $RUNS
}
$manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $MANIFEST -Encoding utf8

Write-Host ""
Write-Host "v7 recipe + FCOS -- $($RUNS.Count) runs | arms=$Arms" -ForegroundColor Cyan
Write-Host "  Arm V reads as SPREAD across seeds, not mean. Arm I reads as the INTERACTION." -ForegroundColor Yellow
Write-Host "  manifest: $MANIFEST" -ForegroundColor DarkGray
Write-Host ""

# PowerShell's "$x" interpolation is invariant-culture, but the -f operator is not:
# under en-ZA it renders 0.1 as "0,1". The value PASSED to train.py is correct either
# way; this only stops the log from reading like a broken argument.
foreach ($cfg in $RUNS) {
    Write-Host ("  [{0}] {1,-24} {2,-14} {3,4}->{4,-4} s{5,-4} freeze={6} aug={7} bblr={8} b={9}/a={10}" -f `
        $cfg.arm, $cfg.variant, $cfg.head, $cfg.input, $cfg.target, $cfg.seed, `
        $cfg.freeze, $cfg.aug, $cfg.backbone_lr_factor.ToString([System.Globalization.CultureInfo]::InvariantCulture), $cfg.batch, $cfg.accum) -ForegroundColor DarkGray
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

    # train.py's directory name does NOT encode freeze / aug / backbone-lr, so a plain
    # name match would wrongly skip a differently-configured run. Confirm against the
    # provenance fields recorded inside metrics.json before skipping.
    $leaf = "$($Config.variant)_$($Config.head)_focal_v5_s$($Config.seed)$resTag"

    $candidates = Get-ChildItem -Path "outputs" -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq $leaf -and
            $_.Parent.Name -like "*_$RunGroup" -and
            (Test-Path (Join-Path $_.FullName "metrics.json"))
        }

    foreach ($candidate in $candidates) {
        $m = Get-Content (Join-Path $candidate.FullName "metrics.json") -Raw | ConvertFrom-Json

        $freezeMatches = ($m.freeze_backbone_epochs -eq $Config.freeze)
        $augMatches    = ($m.aug_profile -eq $Config.aug)
        $lrMatches     = ([math]::Abs($m.backbone_lr_factor - $Config.backbone_lr_factor) -lt 1e-9)

        if ($freezeMatches -and $augMatches -and $lrMatches) { return $candidate }
    }

    return $null
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
    $label = "[$($cfg.arm)] $($cfg.variant)/$($cfg.head) $($cfg.input)->$($cfg.target) s$($cfg.seed) freeze=$($cfg.freeze) ($completed/$($RUNS.Count))"

    if ($Resume) {
        $existing = Get-ExistingRun -Config $cfg

        if ($existing) {
            Write-Host "SKIP $label -- already complete at $($existing.FullName)" -ForegroundColor DarkYellow
            "SKIP $label" | Add-Content $LOG
            continue
        }
    }

    Invoke-Run -Label $label -TrainArgs @(
        "--backbone",                "convnext_fpn",
        "--variant",                 $cfg.variant,
        "--head",                    $cfg.head,
        "--seed",                    "$($cfg.seed)",
        "--datasets",                "node21",
        "--input-size",              "$($cfg.input)",
        "--target-size",             "$($cfg.target)",
        "--batch-size",              "$($cfg.batch)",
        "--grad-accum",              "$($cfg.accum)",
        "--lr",                      "2e-4",
        "--backbone-lr-factor",      "$($cfg.backbone_lr_factor)",
        "--freeze-backbone-epochs",  "$($cfg.freeze)",
        "--epochs",                  "50",
        "--patience",                "$Patience",
        "--select-on",               $SelectOn,
        "--aug-profile",             $cfg.aug,
        "--val-frac",                "0.20",
        "--run-group",               $RunGroup,
        "--augment",
        "--froc"
    )

    $elapsed = (Get-Date) - $RUN_START
    Write-Host ("[timing] {0}/{1} done, cumulative {2:hh\:mm}" -f $completed, $RUNS.Count, $elapsed) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "v7 complete. Log: $LOG" -ForegroundColor Green
Write-Host "Analyse with:" -ForegroundColor Yellow
Write-Host "  python scripts/eval/analyse_v7.py" -ForegroundColor DarkGray
