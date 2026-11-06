# -*- coding: utf-8 -*-
"""Experiment E: paired statistical inference for the declared main-task contrasts.

Eight planned contrasts use the same 300 held-out test images per condition:
four k=0 baseline versus validation-selected k* contrasts and four
validation-selected k* versus k=4 contrasts. Accuracy differences are estimated
with 10,000 class-stratified paired bootstrap replicates. Prediction changes are
tested with an exact two-sided McNemar test; Holm correction covers all eight
planned tests. No test image is used to select k.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260928


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, position in enumerate(order):
        running = max(running, (count - rank) * p_values[position])
        adjusted[position] = min(1.0, running)
    return adjusted


def stratified_bootstrap_delta(truth, candidate_correct, comparator_correct, rng):
    """Class-stratified resampling preserves the 30 images/class test design."""
    values = []
    for class_index in np.unique(truth):
        positions = np.flatnonzero(truth == class_index)
        sample = rng.choice(positions, size=(BOOTSTRAP_REPLICATES, len(positions)), replace=True)
        values.append((candidate_correct[sample] - comparator_correct[sample]).sum(axis=1))
    delta = np.sum(values, axis=0) / len(truth)
    return float(np.quantile(delta, .025)), float(np.quantile(delta, .975))


def paired_row(frame, backbone, selector, comparison, comparator_k, candidate_k, rng):
    subset = frame[(frame.domain == "main") & (frame.backbone == backbone) & (frame.selector == selector)]
    comparator = subset[subset.k == comparator_k].sort_values("stem")
    candidate = subset[subset.k == candidate_k].sort_values("stem")
    assert len(comparator) == len(candidate) == 300
    assert comparator.stem.tolist() == candidate.stem.tolist()
    assert comparator.true_index.tolist() == candidate.true_index.tolist()
    truth = comparator.true_index.to_numpy()
    baseline_correct = (comparator.predicted_index.to_numpy() == truth).astype(float)
    method_correct = (candidate.predicted_index.to_numpy() == truth).astype(float)
    b = int(((baseline_correct == 1) & (method_correct == 0)).sum())
    c = int(((baseline_correct == 0) & (method_correct == 1)).sum())
    discordant = b + c
    raw_p = float(binomtest(min(b, c), n=discordant, p=.5, alternative="two-sided").pvalue) if discordant else 1.0
    low, high = stratified_bootstrap_delta(truth, method_correct, baseline_correct, rng)
    return {
        "backbone": backbone,
        "selector": selector,
        "comparison": comparison,
        "comparator_k": int(comparator_k),
        "candidate_k": int(candidate_k),
        "n_test_images": len(truth),
        "comparator_accuracy": float(baseline_correct.mean()),
        "candidate_accuracy": float(method_correct.mean()),
        "accuracy_difference_candidate_minus_comparator": float((method_correct - baseline_correct).mean()),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "mcnemar_comparator_correct_candidate_wrong": b,
        "mcnemar_comparator_wrong_candidate_correct": c,
        "mcnemar_discordant": discordant,
        "mcnemar_exact_p_raw": raw_p,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_scheme": "class_stratified_paired_image_resampling",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="experiments/results/experiment_47_novelty_rebuild_D/experiment_d_per_image_metrics.csv")
    parser.add_argument("--validation-selected", default="experiments/results/experiment_41_novelty_rebuild_B/experiment_b_validation_selected.csv")
    parser.add_argument("--output-dir", default="experiments/results/experiment_50_novelty_rebuild_E")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    selected = pd.read_csv(args.validation_selected)
    selected = selected[selected.composition == "recompose"].copy()
    assert len(selected) == 4
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for row in selected.itertuples(index=False):
        rows.append(paired_row(predictions, row.backbone, row.selector, "baseline_k0_vs_validation_selected", 0, row.val_selected_k, rng))
        rows.append(paired_row(predictions, row.backbone, row.selector, "validation_selected_vs_full_k4", row.val_selected_k, 4, rng))
    results = pd.DataFrame(rows)
    results["mcnemar_exact_p_holm_8"] = holm_adjust(results.mcnemar_exact_p_raw.to_numpy())
    results["holm_family"] = "8 declared paired main-benchmark contrasts"
    results.to_csv(out / "experiment_e_paired_statistics.csv", index=False)
    results[results.comparison == "baseline_k0_vs_validation_selected"].to_csv(out / "experiment_e_baseline_vs_calibrated.csv", index=False)
    results[results.comparison == "validation_selected_vs_full_k4"].to_csv(out / "experiment_e_compact_vs_k4.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
