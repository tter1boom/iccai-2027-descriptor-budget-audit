# -*- coding: utf-8 -*-
"""Dependence-robust paired inference using visual near-duplicate components.

The fixed evaluation split contains a small number of multi-image visual
similarity components.  This script replaces image-level bootstrap inference
with class-stratified component resampling and adds component-level sign-flip
tests.  Components are dependence units, not specimen identifiers.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_macro_cluster_bootstrap import macro_cluster_bootstrap_ci


N_BOOT = 20_000
N_SIGNFLIP = 100_000
SEED = 20260930


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[position])
        adjusted[position] = min(1.0, running)
    return adjusted


def component_map(groups_path, test_stems, labels):
    groups_path = Path(groups_path)
    if groups_path.suffix.lower() == ".csv":
        groups = pd.read_csv(groups_path)
        required = {"stem", "component_id", "true_index"}
        if not required.issubset(groups.columns):
            raise ValueError(f"component map must contain {sorted(required)}")
        groups = groups[groups.stem.isin(test_stems)].copy()
        observed = groups.set_index("stem").true_index.to_dict()
        if any(observed.get(stem) != label for stem, label in labels.items()):
            raise ValueError("component map labels do not match frozen test predictions")
        if set(groups.stem) != set(test_stems):
            raise ValueError("component map does not cover every test image exactly")
        return groups
    payload = json.loads(groups_path.read_text(encoding="utf-8"))
    mapping = {}
    for group in payload["groups"]:
        in_test = [stem for stem in group["stems"] if stem in test_stems]
        for stem in in_test:
            mapping[stem] = group["group_id"]
    for stem in test_stems:
        mapping.setdefault(stem, f"singleton_{stem}")
    groups = pd.DataFrame({"stem": sorted(test_stems), "component_id": [mapping[stem] for stem in sorted(test_stems)]})
    groups["true_index"] = groups.stem.map(labels)
    return groups


def cluster_bootstrap(component_frame, rng):
    """Equal-class image-delta bootstrap, with components as within-class units."""
    return macro_cluster_bootstrap_ci(
        component_frame.delta.to_numpy(),
        component_frame.true_index.to_numpy(),
        component_frame.component_id.to_numpy(),
        N_BOOT,
        rng,
    )


def signflip(component_frame, rng):
    """Monte-Carlo sign-flip of component-level paired accuracy differences."""
    values = component_frame.groupby("component_id", sort=True).delta.sum().to_numpy(float)
    observed = abs(values.sum())
    exceed = 0
    remaining = N_SIGNFLIP
    while remaining:
        take = min(10_000, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(take, len(values)))
        exceed += int((np.abs(signs @ values) >= observed).sum())
        remaining -= take
    return (exceed + 1.0) / (N_SIGNFLIP + 1.0), len(values)


def pair(predictions, component_lookup, backbone, selector, comparison, comparator_k, candidate_k, rng):
    frame = predictions[(predictions.backbone == backbone) & (predictions.selector == selector)]
    if "domain" in frame.columns:
        frame = frame[frame.domain.eq("main")]
    if "split" in frame.columns:
        frame = frame[frame.split.eq("test")]
    if "composition" in frame.columns:
        frame = frame[frame.composition.eq("recompose")]
    baseline = frame[frame.k == comparator_k].sort_values("stem")
    candidate = frame[frame.k == candidate_k].sort_values("stem")
    assert baseline.stem.tolist() == candidate.stem.tolist() and len(baseline) == 300
    merged = pd.DataFrame({"stem": baseline.stem.to_numpy(), "true_index": baseline.true_index.to_numpy(),
                           "comparator_correct": (baseline.predicted_index.to_numpy() == baseline.true_index.to_numpy()).astype(float),
                           "candidate_correct": (candidate.predicted_index.to_numpy() == candidate.true_index.to_numpy()).astype(float)})
    merged = merged.merge(component_lookup, on=["stem", "true_index"], validate="one_to_one")
    merged["delta"] = merged.candidate_correct - merged.comparator_correct
    low, high = cluster_bootstrap(merged, rng)
    sign_p, n_components = signflip(merged, rng)
    size = merged.groupby("component_id").size()
    return {"backbone": backbone, "selector": selector, "comparison": comparison, "comparator_k": comparator_k,
            "candidate_k": candidate_k, "n_test_images": len(merged), "n_visual_components": n_components,
            "n_multi_image_components": int((size > 1).sum()), "max_component_size": int(size.max()),
            "comparator_accuracy": float(merged.comparator_correct.mean()), "candidate_accuracy": float(merged.candidate_correct.mean()),
            "accuracy_difference_candidate_minus_comparator": float(merged.delta.mean()),
            "group_clustered_bootstrap_ci95_low": low, "group_clustered_bootstrap_ci95_high": high,
            "component_signflip_p_raw": sign_p, "bootstrap_replicates": N_BOOT, "signflip_replicates": N_SIGNFLIP,
            "seed": SEED, "bootstrap_scheme": "class_stratified_visual_component_resampling",
            "signflip_scheme": "visual_component_paired_signflip"}


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    package_data = root / "machine_readable"
    parser.add_argument("--predictions", default=str(package_data / "validation_and_predictions/experiment_b_predictions.csv"))
    parser.add_argument("--base-predictions", default=str(package_data / "validation_and_predictions/main_base_k0_per_image_predictions.csv"))
    parser.add_argument("--validation-selected", default=str(package_data / "validation_and_predictions/experiment_b_validation_selected.csv"))
    parser.add_argument("--groups", default=str(package_data / "group_clustered_inference/test_visual_component_map.csv"))
    parser.add_argument("--output-dir", default=str(package_data / "group_clustered_inference"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    expected = [out / "test_visual_component_map.csv", out / "test_visual_component_sizes.csv", out / "group_clustered_paired_statistics.csv", out / "GROUP_CLUSTERED_STATISTICS.md"]
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Output exists; pass --overwrite or choose another --output-dir")
    predictions = pd.read_csv(args.predictions)
    base_path = Path(args.base_predictions) if args.base_predictions else None
    if base_path is not None:
        if not base_path.is_file():
            raise FileNotFoundError(f"k=0 per-image predictions are missing: {base_path}")
        baseline = pd.read_csv(base_path)
        if not {"backbone", "selector", "k", "stem", "true_index", "predicted_index"}.issubset(baseline.columns):
            raise ValueError("k=0 prediction file has an incompatible schema")
        baseline = baseline[baseline.k.eq(0)].copy()
        if "composition" not in baseline.columns:
            baseline["composition"] = "recompose"
        if "split" not in baseline.columns:
            baseline["split"] = "test"
        predictions = pd.concat([predictions, baseline], ignore_index=True, sort=False)
    selected = pd.read_csv(args.validation_selected)
    selected = selected[selected.composition.eq("recompose")].copy()
    test = predictions.copy()
    if "domain" in test.columns:
        test = test[test.domain.eq("main")]
    if "split" in test.columns:
        test = test[test.split.eq("test")]
    if "composition" in test.columns:
        test = test[test.composition.eq("recompose")]
    labels = test[test.k.eq(0)].drop_duplicates("stem").set_index("stem").true_index.to_dict()
    lookup = component_map(args.groups, set(labels), labels)
    lookup.to_csv(out / "test_visual_component_map.csv", index=False)
    component_sizes = lookup.groupby(["true_index", "component_id"], sort=True).size().reset_index(name="n_images")
    component_sizes.to_csv(out / "test_visual_component_sizes.csv", index=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for setting in selected.itertuples(index=False):
        rows.append(pair(predictions, lookup, setting.backbone, setting.selector, "baseline_k0_vs_validation_selected", 0, int(setting.val_selected_k), rng))
        rows.append(pair(predictions, lookup, setting.backbone, setting.selector, "validation_selected_vs_full_k4", int(setting.val_selected_k), 4, rng))
    results = pd.DataFrame(rows)
    results["component_signflip_p_holm_8"] = holm(results.component_signflip_p_raw.to_numpy())
    results["holm_family"] = "8 declared paired main-benchmark contrasts"
    results.to_csv(out / "group_clustered_paired_statistics.csv", index=False)
    summary = ["# Group-clustered paired statistics", "", "The fixed evaluation split contains 300 images represented by visual-similarity components. Components are dependence-robust resampling units, not specimen IDs.", "", "```csv", results.to_csv(index=False).strip(), "```"]
    (out / "GROUP_CLUSTERED_STATISTICS.md").write_text("\n".join(summary), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
