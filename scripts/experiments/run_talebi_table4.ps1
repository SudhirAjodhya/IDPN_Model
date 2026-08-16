# run_talebi_table4.ps1
# Full Talebi Table 4 sweep on ImageNette.
# Order: smoke -> for each backbone: all bilinear configs -> all proposed configs
#        (bilinears first per backbone so proposed_* can find their init ckpt).
# Approx ~11h for R50 + MobileNet-v2, 2 seeds, 40 epochs on RTX 5070 Ti.

$ErrorActionPreference = "Stop"
Set-Location -Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "../..")

$BACKBONES = @("resnet50", "mobilenet_v2")
$BILINEAR_CONFIGS = @("bilinear_224", "bilinear_256", "bilinear_320", "bilinear_368", "bilinear_448")
$PROPOSED_CONFIGS = @("proposed_224", "proposed_256", "proposed_320", "proposed_368", "proposed_448")
$SEEDS  = @(0, 1)
$EPOCHS = 40

$LOG_ROOT = Join-Path "outputs" "talebi_table4"
if (-not (Test-Path $LOG_ROOT)) { New-Item -ItemType Directory -Path $LOG_ROOT | Out-Null }
$STAMP = Get-Date -Format "yyyy-MM-dd_HHmm"
$LOG = Join-Path $LOG_ROOT "run_$STAMP.log"

function Run($label, $args_) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    "=== $label ===" | Add-Content $LOG
    # PS 5.1 wraps stderr lines from native exes as NativeCommandError under
    # ErrorActionPreference=Stop; the string-cast in ForEach-Object unwraps
    # them so triton/warning noise doesn't abort the sweep.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python scripts/train_talebi_table4.py @args_ 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $LOG -Append
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($exit -ne 0) { throw "$label failed (exit $exit)" }
}

Run "SMOKE TEST" @("--smoke")

foreach ($bb in $BACKBONES) {
    foreach ($seed in $SEEDS) {
        # Bilinears first per (backbone, seed) so proposed_* can find bilinear_224 ckpt
        foreach ($cfg in $BILINEAR_CONFIGS) {
            Run "$bb / $cfg / seed $seed" @(
                "--backbone", $bb, "--config", $cfg, "--seed", $seed, "--epochs", $EPOCHS
            )
        }
        foreach ($cfg in $PROPOSED_CONFIGS) {
            Run "$bb / $cfg / seed $seed" @(
                "--backbone", $bb, "--config", $cfg, "--seed", $seed, "--epochs", $EPOCHS
            )
        }
    }
}

Run "Summary" @("--summarize")

Write-Host "`nAll runs done. Log: $LOG" -ForegroundColor Green


