"""Verify archived analysis grids, exact p-values, replayed controls, and audit tables."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "05_results"

curves = pd.read_csv(RESULTS / "aircraft_budget_curves.csv")
selections = pd.read_csv(ROOT / "04_data/validation_selection_records/all_selections.csv")
identity = pd.read_csv(RESULTS / "interface_k1_identity_audit.csv")
assert len(curves) == 200 and set(curves.k) == set(range(5))
assert len(selections) == 24000
assert selections.support_seed.nunique() == 20
assert selections.validation_permutation.nunique() == 50
assert selections.n_validation_per_class.nunique() == 6

# Three pairwise comparisons in each of five support resamples for each backbone.
pair_counts = identity.groupby(["backbone", "reference_interface", "comparison_interface"]).support_resample_seed.nunique()
assert len(identity) == 30 and len(pair_counts) == 6 and pair_counts.eq(5).all()
assert identity.backbone.nunique() == 2
assert identity.logits_within_tolerance.astype(str).str.lower().eq("true").all()
assert identity.all_image_predictions_identical.astype(str).str.lower().eq("true").all()
assert (identity.max_abs_logit_difference <= 1e-5).all()
assert (identity.prediction_disagreements == 0).all()

# Exact integer exceedance counts, add-one Monte Carlo p, and Holm family.
counts = pd.read_csv(RESULTS / "adjacent_inference.csv")
audit = pd.read_csv(RESULTS / "adjacent_signflip_p_value_audit.csv")
manifest = json.loads((RESULTS / "analysis_outputs/adjacent_budget_inference/analysis_manifest.json").read_text(encoding="utf-8"))
assert len(counts) == len(audit) == 16
expected_raw = (counts.exceedance_count.to_numpy(dtype=np.int64) + 1) / (counts.n_permutations.to_numpy(dtype=np.int64) + 1)
order = np.argsort(expected_raw, kind="stable")
expected_holm = np.empty_like(expected_raw)
running = 0.0
for rank, index in enumerate(order):
    running = max(running, min(1.0, (len(expected_raw) - rank) * expected_raw[index]))
    expected_holm[index] = running
assert np.allclose(expected_raw, counts.raw_p.to_numpy(dtype=float), rtol=0, atol=1e-15)
assert np.allclose(expected_holm, counts.holm_p.to_numpy(dtype=float), rtol=0, atol=1e-14)
assert audit.random_seed.nunique() == 1 and int(audit.random_seed.iloc[0]) == int(manifest["random_seed"])
assert np.array_equal(audit.exceedance_count.to_numpy(), counts.exceedance_count.to_numpy())
assert np.allclose(audit.raw_p, expected_raw, rtol=0, atol=1e-15)
assert np.allclose(audit.holm_p, expected_holm, rtol=0, atol=1e-14)

# Reconcile the former float32 p-value with the current exact integer null without rerunning a model.
reconciliation = json.loads((ROOT / "07_provenance/legacy_signflip_pvalue_reconciliation.json").read_text(encoding="utf-8"))
legacy_null = np.load(ROOT / "07_provenance/adjacent_signflip_null_float32_legacy.npz")
exact_null = np.load(RESULTS / "analysis_outputs/adjacent_budget_inference/adjacent_signflip_null_float64.npz")
contrast = "ViT-B-16|positive_mean|1->2"
column = list(exact_null["labels"]).index(contrast)
observed = int(exact_null["observed_numerators_int64"][column])
integer_null = exact_null["null_numerators_int64"][:, column]
denominator = int(exact_null["statistic_denominator"][0])
recovered_legacy_numerators = np.rint(legacy_null["null_statistics"].astype(np.float64) * denominator).astype(np.int64)
assert np.array_equal(legacy_null["random_seed"], exact_null["random_seed"])
assert np.array_equal(legacy_null["labels"], exact_null["labels"])
assert np.array_equal(recovered_legacy_numerators, exact_null["null_numerators_int64"])
strict = int((np.abs(integer_null) > abs(observed)).sum())
ties = int((np.abs(integer_null) == abs(observed)).sum())
exceedances = int((np.abs(integer_null) >= abs(observed)).sum())
legacy_float32_count = int((np.abs(legacy_null["null_statistics"][:, column].astype(np.float64)) >= abs(observed) / denominator).sum())
assert reconciliation["legacy_implied_exceedance_count"] == 59417
assert reconciliation["current_random_seed"] == int(exact_null["random_seed"][0])
assert (strict, ties, exceedances, legacy_float32_count) == (59372, 53, 59425, 59372)
assert reconciliation["legacy_float32_direct_recount_exceedance_count"] == legacy_float32_count
assert reconciliation["legacy_implied_boundary_ties_counted"] == 45
assert reconciliation["boundary_ties_omitted_by_legacy_comparison"] == 8
assert exceedances == int(audit.loc[audit.contrast_id.eq(contrast), "exceedance_count"].iloc[0])
assert abs((exceedances + 1) / 100001 - float(audit.loc[audit.contrast_id.eq(contrast), "raw_p"].iloc[0])) < 1e-15
assert exceedances - reconciliation["legacy_implied_exceedance_count"] == 8

# Cached-feature replay: fixed random seeds and archived curves must agree exactly.
replay = RESULTS / "replayed_additional_controls"
run = json.loads((replay / "run_manifest.json").read_text(encoding="utf-8"))
seeds = run["random_ranking"]["random_seeds"]
assert seeds == list(range(260100, 260150))
assert run["image_pixels_read"] is False and run["image_encoder_rerun"] is False
for archived, recomputed in [
    (RESULTS / "random_ranking_control.csv", replay / "random_ranking_summary.csv"),
    (RESULTS / "normalized_ensemble.csv", replay / "normalized_mean_curves.csv"),
    (RESULTS / "aircraft_classwise_deltas.csv", replay / "aircraft_classwise_deltas.csv"),
    (RESULTS / "validation_fullsample_paired_by_support.csv", replay / "validation_fullsample_paired_by_support.csv"),
]:
    left, right = pd.read_csv(archived), pd.read_csv(recomputed)
    assert left.columns.tolist() == right.columns.tolist() and left.shape == right.shape
    assert left.equals(right), f"Replayed output differs from archived output: {archived.name}"
random_curves = pd.read_csv(replay / "random_ranking_budget_curves.csv")
assert len(random_curves) == 500 and random_curves.random_ranking_seed.nunique() == 50
assert random_curves.groupby(["backbone", "random_ranking_seed"]).k.nunique().eq(5).all()

# The 100-class audit is a projection of the 500 B/16 PM class-by-resample rows.
per_class = pd.read_csv(RESULTS / "aircraft_per_class_gains_losses.csv")
class_summary = pd.read_csv(RESULTS / "aircraft_per_class_summary.csv")
assert len(per_class) == 100 and per_class.class_id.nunique() == 100
assert per_class.support_resamples.eq(5).all() and len(class_summary) == 9
assert int(per_class.resample_cells_with_any_gain.sum()) == 330
assert int(per_class.resample_cells_with_any_loss.sum()) == 264

# Exhaustive token audit by k, context-limit handling, and checkpoint fingerprints.
tokens = pd.read_csv(ROOT / "01_protocol/prompt_token_audit_by_budget.csv")
assert len(tokens) == 10 and tokens.groupby("backbone").k.apply(lambda x: set(x) == set(range(5))).all()
assert tokens.unique_class_specific_prompts.to_list() == [100, 400, 600, 400, 100] * 2
assert tokens.over_limit_prompts.eq(0).all() and tokens.context_limit.eq(77).all()
assert tokens.max_tokens.max() == 61
checkpoint = json.loads((ROOT / "01_protocol/checkpoint_tokenizer_manifest.json").read_text(encoding="utf-8"))
assert checkpoint["runtime_observed_during_audit_generation"]["transformers"] == "5.16.1"
assert set(checkpoint["models"]) == {"ViT-B-16", "ViT-B-32"}

print("PASS: Aircraft curve and 20 x 50 x 6 validation grids")
print("PASS: 30 k=1 checks = 2 backbones x 5 resamples x 3 pairs; five blocks per backbone/pair")
print("PASS: all 16 integer-count raw p-values and Holm adjustments; seed recorded")
print("PASS: legacy B/16 PM 1->2 p-value reconciled against the same float32-quantized null sequence; exact count includes 53 ties")
print("PASS: 50 random rankings, normalized-mean, classwise and 20 paired validation records exactly match replay")
print("PASS: 100 class summaries and per-budget token/checkpoint audit")

