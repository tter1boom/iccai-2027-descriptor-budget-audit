param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$ModelB32,
    [Parameter(Mandatory=$true)][string]$ModelB16
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Dataset = Join-Path $Root 'machine_readable\inputs\main_benchmark'
$CacheDir = Join-Path $Root 'machine_readable\feature_caches'
$Resource = Join-Path $Root 'machine_readable\source_only_resource\source_only_descriptor_resource_v1.json'
$Out = Join-Path $Root 'reproduced_outputs\source_only_cached_feature_replay'
New-Item -ItemType Directory -Force $Out | Out-Null

& $Python (Join-Path $PSScriptRoot '56_source_only_budget_curves.py') `
    --dataset $Dataset --resource $Resource --features (Join-Path $CacheDir 'visual_audit_crop4_features.npz') `
    --model $ModelB32 --backbone 'ViT-B-32' --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw "ViT-B-32 source-only replay failed with exit code $LASTEXITCODE." }
& $Python (Join-Path $PSScriptRoot '56_source_only_budget_curves.py') `
    --dataset $Dataset --resource $Resource --features (Join-Path $CacheDir 'visual_audit_crop4_features_b16.npz') `
    --model $ModelB16 --backbone 'ViT-B-16' --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw "ViT-B-16 source-only replay failed with exit code $LASTEXITCODE." }

Write-Host "Completed source-only cached-feature replay: $Out"
