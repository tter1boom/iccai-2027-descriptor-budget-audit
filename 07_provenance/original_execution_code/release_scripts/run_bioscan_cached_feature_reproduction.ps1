param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$ModelB32,
    [Parameter(Mandatory=$true)][string]$ModelB16
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Dataset = Join-Path $Root 'machine_readable\inputs\bioscan_boundary'
$CacheDir = Join-Path $Root 'machine_readable\bioscan_boundary\feature_caches'
$Catalogue = Join-Path $Root 'machine_readable\additional_analyses\class_knowledge_untruncated_posthoc_v1.json'
$Out = Join-Path $Root 'reproduced_outputs\experiment_c_bioscan_untruncated_cached_feature_replay'
New-Item -ItemType Directory -Force $Out | Out-Null

& $Python (Join-Path $PSScriptRoot '44_novelty_rebuild_experiment_c.py') `
    --dataset $Dataset --catalogue $Catalogue --features (Join-Path $CacheDir 'bioscan_subset_clip_features_b32.npz') `
    --model $ModelB32 --backbone 'ViT-B-32' --output-dir $Out --strict-tokens
if ($LASTEXITCODE -ne 0) { throw "ViT-B-32 BIOSCAN untruncated replay failed with exit code $LASTEXITCODE." }
& $Python (Join-Path $PSScriptRoot '44_novelty_rebuild_experiment_c.py') `
    --dataset $Dataset --catalogue $Catalogue --features (Join-Path $CacheDir 'bioscan_subset_clip_features_b16.npz') `
    --model $ModelB16 --backbone 'ViT-B-16' --output-dir $Out --strict-tokens
if ($LASTEXITCODE -ne 0) { throw "ViT-B-16 BIOSCAN untruncated replay failed with exit code $LASTEXITCODE." }

Write-Host "Completed strict-token BIOSCAN cached-feature replay: $Out"
