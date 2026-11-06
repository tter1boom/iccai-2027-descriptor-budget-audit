# -*- coding: utf-8 -*-
"""Exploratory visual-component intervals for the post hoc source-only resource.

These intervals are deliberately separate from the declared original-resource
tests.  They quantify paired selected-budget versus k=4 differences, but do not
transfer p-values or confirmatory claims from the historical resource.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_macro_cluster_bootstrap import macro_cluster_bootstrap_ci


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "machine_readable" / "source_only_exploratory_intervals"
SOURCE = ROOT / "machine_readable" / "source_only_resource"
GROUPS = ROOT / "machine_readable" / "group_clustered_inference" / "test_visual_component_map.csv"
N_BOOT = 20_000
SEED = 20260930


def test_component_map(test_stems, labels):
    if GROUPS.suffix.lower() == ".csv":
        lookup = pd.read_csv(GROUPS)
        lookup = lookup[lookup.stem.isin(test_stems)].copy()
        if set(lookup.stem) != set(test_stems):
            raise RuntimeError("Frozen component map does not cover source-only test images")
        observed = lookup.set_index("stem").true_index.to_dict()
        if any(observed.get(stem) != label for stem, label in labels.items()):
            raise RuntimeError("Frozen component-map labels disagree with source-only predictions")
        return lookup[["stem", "component_id", "true_index"]].sort_values("stem").reset_index(drop=True)
    payload = json.loads(GROUPS.read_text(encoding="utf-8"))
    mapping = {}
    for group in payload["groups"]:
        for stem in group["stems"]:
            if stem in test_stems:
                mapping[stem] = group["group_id"]
    return pd.DataFrame({"stem": sorted(test_stems),
                         "component_id": [mapping.get(stem, f"singleton_{stem}") for stem in sorted(test_stems)],
                         "true_index": [labels[stem] for stem in sorted(test_stems)]})


def component_ci(frame, rng):
    return macro_cluster_bootstrap_ci(
        frame.delta.to_numpy(),
        frame.true_index.to_numpy(),
        frame.component_id.to_numpy(),
        N_BOOT,
        rng,
    )


def compare(predictions, lookup, backbone, selector, comparator_k, candidate_k, rng):
    data = predictions[(predictions.backbone == backbone) & (predictions.selector == selector) & predictions.split.eq("test")]
    comparator = data[data.k.eq(comparator_k)].sort_values("stem")
    candidate = data[data.k.eq(candidate_k)].sort_values("stem")
    assert comparator.stem.tolist() == candidate.stem.tolist() and len(comparator) == 300
    paired = pd.DataFrame({"stem": comparator.stem.to_numpy(), "true_index": comparator.true_index.to_numpy(),
                           "comparator_correct": comparator.correct.astype(float).to_numpy(),
                           "candidate_correct": candidate.correct.astype(float).to_numpy()})
    paired = paired.merge(lookup, on=["stem", "true_index"], validate="one_to_one")
    paired["delta"] = paired.candidate_correct - paired.comparator_correct
    low, high = component_ci(paired, rng)
    return {"analysis_status": "exploratory_post_hoc_no_confirmatory_p_value",
            "backbone": backbone, "selector": selector, "comparator_k": comparator_k, "candidate_k": candidate_k,
            "comparison": f"k={candidate_k}_minus_k={comparator_k}", "n_test_images": len(paired),
            "n_visual_components": int(paired.component_id.nunique()),
            "comparator_accuracy": float(paired.comparator_correct.mean()),
            "candidate_accuracy": float(paired.candidate_correct.mean()),
            "accuracy_difference_candidate_minus_comparator": float(paired.delta.mean()),
            "component_clustered_bootstrap_ci95_low": low, "component_clustered_bootstrap_ci95_high": high,
            "bootstrap_replicates": N_BOOT, "seed": SEED,
            "bootstrap_scheme": "class_stratified_visual_component_resampling"}


def main():
    global N_BOOT, SEED, SOURCE, GROUPS, OUT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--component-map", type=Path, default=GROUPS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    N_BOOT, SEED, SOURCE, GROUPS, OUT = args.bootstrap_replicates, args.seed, args.source_dir, args.component_map, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / "source_only_test_visual_component_map.csv", OUT / "source_only_exploratory_paired_intervals.csv", OUT / "SOURCE_ONLY_EXPLORATORY_INTERVALS.md"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Output exists; pass --overwrite or choose another --output-dir")
    predictions = pd.concat([pd.read_csv(SOURCE / "source_only_ViT-B-32_per_image_predictions.csv"),
                             pd.read_csv(SOURCE / "source_only_ViT-B-16_per_image_predictions.csv")], ignore_index=True)
    selected = pd.concat([pd.read_csv(SOURCE / "source_only_ViT-B-32_validation_selection.csv"),
                          pd.read_csv(SOURCE / "source_only_ViT-B-16_validation_selection.csv")], ignore_index=True)
    test0 = predictions[(predictions.split == "test") & (predictions.k == 0)].drop_duplicates("stem")
    lookup = test_component_map(set(test0.stem), test0.set_index("stem").true_index.to_dict())
    lookup.to_csv(OUT / "source_only_test_visual_component_map.csv", index=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for setting in selected.itertuples(index=False):
        k_star = int(setting.selected_k)
        rows.append(compare(predictions, lookup, setting.backbone, setting.selector, 4, k_star, rng))
        rows.append(compare(predictions, lookup, setting.backbone, setting.selector, 0, k_star, rng))
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "source_only_exploratory_paired_intervals.csv", index=False)
    lines = ["# Source-only exploratory paired intervals", "", "These intervals concern a post hoc, image-blind resource reconstruction. They are descriptive sensitivity intervals only; no hypothesis-test p-values are reported and no historical-resource inference is transferred to this resource.", "", "Bootstrap replicates resample visual components with replacement within each class. Sampled component image deltas are pooled within class, divided by the sampled image count, and macro-averaged equally across classes. This is the image-level macro-accuracy estimand.", "", "```csv", results.to_csv(index=False).strip(), "```"]
    (OUT / "SOURCE_ONLY_EXPLORATORY_INTERVALS.md").write_text("\n".join(lines), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
