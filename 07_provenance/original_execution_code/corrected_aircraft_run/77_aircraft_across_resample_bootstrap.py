"""Across-resample paired bootstrap for the clean FGVC-Aircraft run.

The estimand is the equally weighted mean macro-accuracy contrast across the
five frozen support/validation resamples.  Each bootstrap replicate resamples
test images within class once and applies that same resample to all five
support/validation realizations, preserving the shared-test-set dependence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONTRASTS = (
    ("selected_minus_k0", "selected", 0),
    ("selected_minus_k1", "selected", 1),
    ("selected_minus_k4", "selected", 4),
    ("k1_minus_k4", 1, 4),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("external_aircraft_corrected_clean"))
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    selected = pd.read_csv(args.run_dir / "selected_budgets.csv")
    test = predictions[predictions["split"] == "test"].copy()
    outputs = args.run_dir / "bootstrap_outputs_across_resamples"
    outputs.mkdir(exist_ok=False)
    rows: list[dict] = []

    for backbone in sorted(test["backbone"].unique()):
        for scorer in sorted(test["scorer"].unique()):
            block = test[(test["backbone"] == backbone) & (test["scorer"] == scorer)]
            seeds = sorted(block["seed"].unique())
            selection = selected[(selected["backbone"] == backbone) & (selected["scorer"] == scorer)].set_index("seed")
            reference = block[(block["seed"] == seeds[0]) & (block["k"] == 0)].sort_values("image_id")
            image_ids = reference["image_id"].to_numpy()
            truth = reference["true_index"].to_numpy()
            classes = np.unique(truth)

            for contrast, left_spec, right_spec in CONTRASTS:
                deltas = []
                seed_effects = []
                for seed in seeds:
                    left_k = int(selection.loc[seed, "selected_k"]) if left_spec == "selected" else int(left_spec)
                    right_k = int(right_spec)
                    left = block[(block["seed"] == seed) & (block["k"] == left_k)].sort_values("image_id")
                    right = block[(block["seed"] == seed) & (block["k"] == right_k)].sort_values("image_id")
                    if not (np.array_equal(left["image_id"].to_numpy(), image_ids)
                            and np.array_equal(right["image_id"].to_numpy(), image_ids)):
                        raise RuntimeError(f"ID alignment failed: {backbone}/{scorer}/{seed}/{contrast}")
                    delta = ((left["predicted_index"].to_numpy() == truth).astype(np.float32)
                             - (right["predicted_index"].to_numpy() == truth).astype(np.float32))
                    deltas.append(delta)
                    seed_effects.append(float(np.mean([delta[truth == c].mean() for c in classes])))

                delta_matrix = np.stack(deltas, axis=0)
                rng = np.random.default_rng(args.seed + sum(map(ord, backbone + scorer + contrast)))
                draw_blocks = []
                for start in range(0, args.replicates, 500):
                    count = min(500, args.replicates - start)
                    block_draws = np.zeros(count, dtype=np.float32)
                    for class_id in classes:
                        positions = np.flatnonzero(truth == class_id)
                        sampled = positions[rng.integers(0, len(positions), size=(count, len(positions)))]
                        block_draws += delta_matrix[:, sampled].mean(axis=(0, 2)) / len(classes)
                    draw_blocks.append(block_draws)
                draws = np.concatenate(draw_blocks)

                observed = float(np.mean(seed_effects))
                np.savez_compressed(
                    outputs / f"{backbone}_{scorer}_{contrast}.npz",
                    bootstrap_macro_accuracy_difference=draws,
                    seed_macro_accuracy_difference=np.asarray(seed_effects),
                    seeds=np.asarray(seeds),
                )
                rows.append({
                    "backbone": backbone,
                    "scorer": scorer,
                    "contrast": contrast,
                    "estimand": "equal-weight mean macro-accuracy difference across five frozen resamples",
                    "mean_effect": observed,
                    "seed_sd": float(np.std(seed_effects, ddof=1)),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_replicates": args.replicates,
                    "bootstrap_unit": "test images resampled within class; same draw applied across resamples",
                })

    pd.DataFrame(rows).to_csv(args.run_dir / "paired_bootstrap_across_resamples.csv", index=False)
    print(args.run_dir / "paired_bootstrap_across_resamples.csv")


if __name__ == "__main__":
    main()
