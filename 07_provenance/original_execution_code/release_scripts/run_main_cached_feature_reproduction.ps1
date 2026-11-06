param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$ModelB32,
    [Parameter(Mandatory=$true)][string]$ModelB16
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Dataset = Join-Path $Root 'machine_readable\inputs\main_benchmark'
$CacheDir = Join-Path $Root 'machine_readable\feature_caches'
$Out = Join-Path $Root 'reproduced_outputs\experiment_b_cached_feature_replay'

& $Python (Join-Path $PSScriptRoot 'verify_release.py')
if ($LASTEXITCODE -ne 0) { throw "Release-integrity verification failed with exit code $LASTEXITCODE." }
New-Item -ItemType Directory -Force $Out | Out-Null

# Replays the canonical aggregation diagnostic from frozen image features.
# It does not require original image files, but does require local CLIP checkpoints.
& $Python (Join-Path $PSScriptRoot '41_novelty_rebuild_experiment_b.py') `
    --dataset $Dataset --base-features (Join-Path $CacheDir 'visual_audit_crop4_features.npz') `
    --model $ModelB32 --backbone 'ViT-B-32' --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw "ViT-B-32 main cached-feature replay failed with exit code $LASTEXITCODE." }
& $Python (Join-Path $PSScriptRoot '41_novelty_rebuild_experiment_b.py') `
    --dataset $Dataset --base-features (Join-Path $CacheDir 'visual_audit_crop4_features_b16.npz') `
    --model $ModelB16 --backbone 'ViT-B-16' --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw "ViT-B-16 main cached-feature replay failed with exit code $LASTEXITCODE." }

Write-Host "Completed cached-feature replay: $Out"
