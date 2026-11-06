param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$ModelB32,
    [Parameter(Mandatory=$true)][string]$ModelB16
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Dataset = Join-Path $Root 'machine_readable\inputs\main_benchmark'
$Cache = Join-Path $Root 'machine_readable\feature_caches'
$Out = Join-Path $Root 'reproduced_outputs\same_support_calibration'
New-Item -ItemType Directory -Force $Out | Out-Null

& $Python (Join-Path $PSScriptRoot '54_teacher_targeted_validation.py') `
    --dataset $Dataset --output-dir $Out --random-draws 50 --bootstrap-replicates 50 `
    --model-b32 $ModelB32 --model-b16 $ModelB16 `
    --features-b32 (Join-Path $Cache 'visual_audit_crop4_features.npz') `
    --features-b16 (Join-Path $Cache 'visual_audit_crop4_features_b16.npz')
if ($LASTEXITCODE -ne 0) { throw "Matched support/validation calibration replay failed with exit code $LASTEXITCODE." }
& $Python (Join-Path $PSScriptRoot '64_same_support_calibration_summary.py') --input-dir $Out
if ($LASTEXITCODE -ne 0) { throw "Matched calibration summary failed with exit code $LASTEXITCODE." }
if (-not (Test-Path (Join-Path $Out 'teacher_bootstrap_all_k_predictions.csv'))) { throw 'Expected all-k bootstrap predictions were not produced.' }
Write-Host "Completed matched support/validation calibration replay: $Out"
