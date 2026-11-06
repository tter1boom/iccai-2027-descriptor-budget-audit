# ICCAI 2027 Reproducibility Package

This public repository is the single entry point for the paper's protocol, source-linked Aircraft resource, per-image outputs, cached features, analysis records, and reproduction code. Repository URL: https://github.com/tter1boom/iccai-2027-descriptor-budget-audit. It is public and account-attributed, not an anonymous deposit. The package does not redistribute the authorized 793-image development set, FGVC-Aircraft source images, or local CLIP checkpoints.

## Directory guide

- `01_protocol/`: experiment, dataset, model, prompt, token, split, and analysis-plan records.
- `02_resources/`: frozen Aircraft descriptors, item-level audit, revision/freeze evidence, and evidence map.
- `03_code/`: current package-relative scoring, ranking, prompting, experiment, and statistical utilities.
- `04_data/`: per-image predictions, validation-selection records, cached image features, and non-image historical development evidence.
- `05_results/`: paper-facing tables, figure source data, and detailed archived outputs.
- `06_reproduction/`: release checks, prediction-level recomputation, output audits, and cached-feature replay entry points.
- `07_provenance/`: selected historical execution code and records, pre-correction Aircraft evidence, and provenance index. These are provenance records, not the recommended active commands; a small number of old auxiliary paths are redacted in the public copy.
- `08_integrity/`: file hashes, verification summary, and release manifest.

## Commands

Run from this directory with the environment in `environment.yml` activated.

| Command | Inputs | What it verifies or produces | Checkpoint required? |
|---|---|---|---|
| `python 06_reproduction/verify_release.py` | Package files; sibling submission documents are optional | Checks package layout, one README, active-path hygiene, and package hashes; also checks sibling submission-file hashes when present | No |
| `python 06_reproduction/verify_analysis_tables.py` | Archived predictions, analysis tables, identity audits, and replay outputs | Checks S5-S7 source records, analysis grids, counts, p-values, and table consistency | No |
| `python 06_reproduction/recompute_from_predictions.py` | `04_data/aircraft_prediction_records/` | Recomputes Aircraft accuracy curves from per-image predictions | No |
| `python 06_reproduction/rerun_resampling.py` | Archived exact exceedance counts and paired effects | Recomputes raw p-values and Holm adjustment; it does not regenerate the 100,000 null draws | No |
| `python 06_reproduction/replay_from_features.py --models-root <checkpoint-root>` | Packaged feature cache, saved settings, local frozen text checkpoints | Replays random ranking, class-name-only, normalized-mean, and three-interface identity analyses | Yes |
| `python 03_code/experiments/fixed_catalogue_control.py --models-root <checkpoint-root>` | Packaged feature cache, split manifests, local checkpoint root | Replays fixed-catalogue ordering control | Yes |
| `python 06_reproduction/generate_supplementary_audits.py --models-root <checkpoint-root>` | Archived class-wise/p-value records and local tokenizers/checkpoints | Regenerates class-wise, exact p-value, token, and checkpoint audit tables | Yes |
| `python 06_reproduction/generate_paper_tables.py` | Archived CSV results | Prints row-count inventory only; it is not a table-generation pipeline | No |

## Review evidence map

- Supplementary Table S5: `05_results/interface_k1_identity_audit.csv` (30 pairwise checks: two backbones ? five support resamples ? three interface pairs).
- Supplementary Table S6: `02_resources/resource_revision_timeline.csv`, `02_resources/resource_file_map.csv`, `01_protocol/pre_run_checks.json`, and the corrected-run record identified in `07_provenance/PROVENANCE_INDEX.csv`.
- Supplementary Table S7: `05_results/aircraft_per_class_summary.csv`; the 100-class detail is `05_results/aircraft_per_class_gains_losses.csv`.
- Per-image predictions: `04_data/aircraft_prediction_records/`; the fixed official test split is reused across the five support resamples.
- Random ranking and normalized-mean outputs: `05_results/replayed_additional_controls/`, with fixed seeds listed in `07_provenance/additional_controls_run_manifest.json`.

## Reproduction boundary

Prediction/statistical checks and curve recomputation use archived per-image predictions and analysis inputs; they do not require source images. Cached-feature replay uses the packaged feature cache and local fingerprinted text-encoder checkpoints, but no image files. End-to-end image inference requires the authorized source datasets and frozen checkpoints, which are not redistributed.

The official Aircraft test split was accessed in an earlier analysis before resource correction. The reported result is a protocol-gated corrected rerun, not an untouched holdout. The adjacent-budget analysis was specified after earlier test access; multiplicity adjustment does not remove adaptation associated with that history.
