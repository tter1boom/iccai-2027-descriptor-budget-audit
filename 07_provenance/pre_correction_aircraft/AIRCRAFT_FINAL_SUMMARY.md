# FGVC-Aircraft external-confirmation round: final frozen summary

## Status and scope

This document archives the pre-specified external-confirmation round. Its
resource (`resources/aircraft_descriptors_clean.json`), five split manifests
(20261003--20261007), checkpoints, prompt templates, and original result files
are frozen. No descriptor, seed, or result was selected or changed after these
analyses. The Stage-2 materials in `stage2_posthoc/` are separate, explicitly
post-hoc mechanism diagnostics and are not part of this confirmation round.

The experiment used the fixed FGVC-Aircraft S/V/Q manifests, 100 classes, four
source-derived descriptors per class, two frozen CLIP backbones (ViT-B-16 and
ViT-B-32), and two descriptor rankers (positive mean, PM; contrastive margin,
CM). Reported curves are mean test macro accuracy across five resamples. At
each seed, validation selected the smallest k attaining the highest validation
macro accuracy.

## Complete test budget curves

| Backbone | Scorer | k=0 | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B-16 | PM | 24.45 | 26.25 | 26.39 | 24.70 | 25.27 |
| ViT-B-16 | CM | 24.45 | 25.62 | 25.70 | 24.24 | 25.27 |
| ViT-B-32 | PM | 19.79 | 19.99 | 19.95 | 18.73 | 19.89 |
| ViT-B-32 | CM | 19.79 | 19.73 | 19.68 | 18.56 | 19.89 |

Values are percentages. Thus, the Aircraft response was non-monotonic in all
four backbone/ranker conditions, but its shape differed by backbone: B/16
peaked at compact budgets (k=1--2), whereas B/32 showed a shallow curve with a
marked trough at k=3 and no consistent compact-budget gain.

## Validation-selected operating rule versus fixed budgets

| Backbone | Scorer | selected-k frequency over 5 seeds | selected test macro accuracy | fixed k=1 | fixed k=4 |
|---|---|---|---:|---:|---:|
| ViT-B-16 | PM | k=0: 1, k=1: 4 | 25.86 | 26.25 | 25.27 |
| ViT-B-16 | CM | k=0: 1, k=1: 4 | 25.35 | 25.62 | 25.27 |
| ViT-B-32 | PM | k=1: 1, k=2: 2, k=4: 2 | 19.96 | 19.99 | 19.89 |
| ViT-B-32 | CM | k=1: 2, k=2: 1, k=4: 2 | 19.79 | 19.73 | 19.89 |

The selector was therefore an adaptive safeguard, not a uniformly better
replacement for a fixed budget. In particular, B/16 selected k=0 at one seed
for both rankers, and its mean selected result was below fixed k=1.

## Paired bootstrap: correction to the earlier statement

The 20,000-replicate class-stratified paired bootstrap was computed from paired
test predictions. Its effect column is paired top-1 accuracy (not the macro
accuracy used for the curves), and CIs are reported in percentage points (pp).
The statement that *all* CIs crossed zero was incorrect. Five stored contrast
rows excluded zero, although two are the same B/16 PM k=1-versus-k=4 comparison
reported under both a selected-versus-k=4 and a fixed-k=1-versus-k=4 label
because the selected budget was k=1.

| Seed | Backbone | Scorer | Contrast | Difference (pp) | 95% CI (pp) |
|---:|---|---|---|---:|---:|
| 20261004 | ViT-B-16 | PM | k=1 minus k=4 | +1.17 | [+0.06, +2.28] |
| 20261004 | ViT-B-16 | PM | selected k=0 minus k=1 | -2.01 | [-3.09, -0.83] |
| 20261004 | ViT-B-16 | CM | selected k=0 minus k=1 | -1.35 | [-2.42, -0.23] |
| 20261007 | ViT-B-16 | PM | selected k=1 minus k=4 | +1.38 | [+0.33, +2.43] |
| 20261007 | ViT-B-16 | PM | k=1 minus k=4 | +1.38 | [+0.33, +2.43] |

Across seeds, the Aircraft experiment did not show a stable advantage of
validation-selected budgets over fixed k=1. Most paired contrasts were small
and seed-dependent; several individual B/16 contrasts excluded zero, including
one seed in which validation selected k=0 and significantly underperformed
fixed k=1. The repeated 20261007 row does not constitute independent evidence.

For orientation only, seed-averaged selected-minus-k=1 paired effects were
-0.40 pp (B/16 PM), -0.27 pp (B/16 CM), -0.04 pp (B/32 PM), and +0.05 pp
(B/32 CM). These averages do not convert the seed-level analysis into a new
independent confirmation test.

## Frozen-resource and representation diagnostics

| Backbone | mean descriptor redundancy | descriptor-to-base overlap | mean descriptor tokens |
|---|---:|---:|---:|
| ViT-B-16 | 0.874 | 0.732 | 7.22 |
| ViT-B-32 | 0.900 | 0.774 | 7.22 |

Global prototype drift and mean correct-class decision margin also varied with
budget and backbone. For B/16 PM, drift was 0.000, 0.074, 0.072, 0.075, and
0.061 for k=0--4, while margins were -0.0089, -0.0081, -0.0079, -0.0097, and
-0.0086. For B/32 PM, drift was 0.000, 0.046, 0.044, 0.049, and 0.056, while
margins were -0.0122, -0.0113, -0.0114, -0.0126, and -0.0106. CM produced the
same broad pattern. These diagnostics establish representation movement, not a
causal explanation for the performance curves.

## Bounded conclusion

The frozen Aircraft round supports the existence of a non-monotonic
descriptor-budget response under a new domain and resource, while showing that
the response shape and the behavior of the validation rule depend on backbone
and scoring condition. It does **not** show that validation-selected budgets
stably outperform fixed k=1, and it does not identify redundancy, wording, or
length as the cause of the B/16 versus B/32 difference. Those latter questions
are reserved for the labelled post-hoc Stage-2 diagnostics.

## Archived source files

- Curves: `results/budget_curves.csv`
- Selected budgets: `results/selected_budgets.csv`
- Paired bootstrap: `results/paired_bootstrap.csv`
- Resource properties: `results/resource_properties.csv`
- Prototype drift and margins: `results/representation_diagnostics.csv`
- Frozen hashes and split manifests: `protocol/PROTOCOL_FROZEN.json` and
  `protocol/split_seed_*.json`
