"""Experiment 1: adjacent-budget inference for the clean Aircraft run.

This script uses only the saved per-image predictions from the corrected clean
FGVC-Aircraft experiment. It does not encode images, rank descriptors, select
budgets, or access validation predictions.

Frozen analysis definition
--------------------------
* Conditions: ViT-B/16 and ViT-B/32 x positive mean and contrastive margin.
* Transitions: 0->1, 1->2, 2->3, and 3->4 (16 contrasts in one family).
* Estimand: equal-weight mean, over the five frozen support/validation
  resamples, of the test macro-accuracy difference A_s(k+1) - A_s(k).
* Bootstrap: 20,000 class-stratified paired replicates. Within every class,
  test images are sampled with replacement. A draw is shared by both budgets,
  all five resamples, and all 16 contrasts. Percentile 95% intervals are used.
* Randomization test: 100,000 Monte Carlo two-sided sign flips. The five
  resample-specific correctness differences for one test image form one block
  and receive one common random sign. Raw p values use the add-one correction.
* Multiplicity: Holm correction across all 16 adjacent-budget contrasts.

The default output directory must not already contain result files unless
--overwrite is explicitly supplied. Use --preflight to validate inputs without
computing or revealing numerical results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FROZEN_SEEDS = (20261003, 20261004, 20261005, 20261006, 20261007)
FROZEN_BACKBONES = ("ViT-B-16", "ViT-B-32")
FROZEN_SCORERS = ("positive_mean", "contrastive_margin")
FROZEN_BUDGETS = (0, 1, 2, 3, 4)
TRANSITIONS = ((0, 1), (1, 2), (2, 3), (3, 4))
EXPECTED_TEST_IMAGES = 3333
EXPECTED_CLASSES = 100
REQUIRED_COLUMNS = {
    "seed",
    "backbone",
    "scorer",
    "split",
    "k",
    "image_id",
    "true_index",
    "predicted_index",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(master_seed: int, label: str) -> int:
    payload = f"{master_seed}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def holm_adjust(raw_p: np.ndarray) -> np.ndarray:
    """Holm adjusted p values, monotone in ordered raw p values."""
    raw_p = np.asarray(raw_p, dtype=np.float64)
    if raw_p.ndim != 1 or np.any(~np.isfinite(raw_p)):
        raise ValueError("Holm correction requires a finite one-dimensional array")
    m = raw_p.size
    order = np.argsort(raw_p, kind="stable")
    ordered = raw_p[order]
    adjusted_ordered = np.maximum.accumulate((m - np.arange(m)) * ordered)
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_ordered
    return adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen adjacent-budget inference on clean Aircraft predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("external_aircraft_corrected_clean/predictions.csv"),
        help="Corrected-clean per-image prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision_experiments_v13/exp1_aircraft_adjacent"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--signflip-replicates", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=20261004)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Monte Carlo rows processed together; lower this if RAM is limited.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate frozen inputs and exit without calculating results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this script's result files only.",
    )
    return parser.parse_args()


def validate_and_load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path.resolve()}")
    frame = pd.read_csv(path, dtype={"image_id": str})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    test = frame.loc[frame["split"].eq("test"), sorted(REQUIRED_COLUMNS)].copy()
    if test.empty:
        raise ValueError("No split='test' rows found")

    expected_conditions = {
        (backbone, scorer)
        for backbone in FROZEN_BACKBONES
        for scorer in FROZEN_SCORERS
    }
    observed_conditions = set(zip(test["backbone"], test["scorer"]))
    if observed_conditions != expected_conditions:
        raise ValueError(
            f"Condition mismatch: expected {sorted(expected_conditions)}, "
            f"observed {sorted(observed_conditions)}"
        )
    if set(test["seed"].astype(int)) != set(FROZEN_SEEDS):
        raise ValueError("The five frozen resample seeds do not match the protocol")
    if set(test["k"].astype(int)) != set(FROZEN_BUDGETS):
        raise ValueError("Observed budgets do not equal k=0,...,4")

    key = ["backbone", "scorer", "seed", "k", "image_id"]
    if test.duplicated(key).any():
        examples = test.loc[test.duplicated(key, keep=False), key].head().to_dict("records")
        raise ValueError(f"Duplicate test prediction keys found, e.g. {examples}")

    group_sizes = test.groupby(["backbone", "scorer", "seed", "k"]).size()
    if not group_sizes.eq(EXPECTED_TEST_IMAGES).all():
        raise ValueError(f"Every condition/seed/k must contain {EXPECTED_TEST_IMAGES} test rows")
    return test


def build_effect_tensor(
    test: pd.DataFrame,
) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray, list[str]]:
    """Return [contrast, seed, image] correctness differences and metadata."""
    reference = (
        test.loc[
            (test["backbone"] == FROZEN_BACKBONES[0])
            & (test["scorer"] == FROZEN_SCORERS[0])
            & (test["seed"] == FROZEN_SEEDS[0])
            & (test["k"] == 0)
        ]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    image_ids = reference["image_id"].to_numpy()
    truth = reference["true_index"].to_numpy(dtype=np.int64)
    classes = np.unique(truth)
    if image_ids.size != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} shared test images")
    if classes.size != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} test classes")

    metadata: list[dict] = []
    arrays: list[np.ndarray] = []
    for backbone in FROZEN_BACKBONES:
        for scorer in FROZEN_SCORERS:
            condition = test.loc[
                (test["backbone"] == backbone) & (test["scorer"] == scorer)
            ]
            for left_k, right_k in TRANSITIONS:
                seed_deltas: list[np.ndarray] = []
                for seed in FROZEN_SEEDS:
                    pair: dict[int, pd.DataFrame] = {}
                    for budget in (left_k, right_k):
                        rows = (
                            condition.loc[
                                (condition["seed"] == seed) & (condition["k"] == budget)
                            ]
                            .sort_values("image_id")
                            .reset_index(drop=True)
                        )
                        if not np.array_equal(rows["image_id"].to_numpy(), image_ids):
                            raise RuntimeError(
                                f"Image alignment failed for {backbone}/{scorer}/"
                                f"seed={seed}/k={budget}"
                            )
                        if not np.array_equal(rows["true_index"].to_numpy(), truth):
                            raise RuntimeError(
                                f"Truth alignment failed for {backbone}/{scorer}/"
                                f"seed={seed}/k={budget}"
                            )
                        pair[budget] = rows
                    left_correct = (
                        pair[left_k]["predicted_index"].to_numpy(dtype=np.int64) == truth
                    ).astype(np.int8)
                    right_correct = (
                        pair[right_k]["predicted_index"].to_numpy(dtype=np.int64) == truth
                    ).astype(np.int8)
                    seed_deltas.append(right_correct - left_correct)
                arrays.append(np.stack(seed_deltas, axis=0))
                metadata.append(
                    {
                        "backbone": backbone,
                        "ranker": scorer,
                        "transition": f"{left_k}->{right_k}",
                        "k_from": left_k,
                        "k_to": right_k,
                    }
                )
    return np.stack(arrays, axis=0), metadata, image_ids, truth, [str(c) for c in classes]


def macro_effects_by_seed(effect_tensor: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Macro-accuracy differences with shape [contrast, seed]."""
    classes = np.unique(truth)
    values = np.zeros((effect_tensor.shape[0], effect_tensor.shape[1]), dtype=np.float64)
    for class_id in classes:
        mask = truth == class_id
        class_means = effect_tensor[:, :, mask].mean(axis=2)
        values += class_means / classes.size
    return values


def paired_class_bootstrap(
    effect_tensor: np.ndarray,
    truth: np.ndarray,
    replicates: int,
    random_seed: int,
    chunk_size: int,
) -> np.ndarray:
    """Shared class-stratified test draw for all contrasts and resamples."""
    rng = np.random.default_rng(stable_seed(random_seed, "bootstrap"))
    classes = np.unique(truth)
    # Linearity makes averaging over the five resamples before sampling exactly
    # equivalent to sampling each resample and then taking their equal-weight mean.
    mean_over_resamples = effect_tensor.mean(axis=1, dtype=np.float64)
    output = np.empty((replicates, effect_tensor.shape[0]), dtype=np.float32)
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        draws = np.zeros((count, effect_tensor.shape[0]), dtype=np.float64)
        for class_id in classes:
            positions = np.flatnonzero(truth == class_id)
            sampled = positions[
                rng.integers(0, positions.size, size=(count, positions.size))
            ]
            draws += mean_over_resamples[:, sampled].mean(axis=2).T / classes.size
        output[start : start + count] = draws.astype(np.float32)
    return output


def image_block_signflip(
    effect_tensor: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    replicates: int,
    random_seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Monte Carlo sign-flip test with one sign per test image.

    The five-resample difference vector stays intact because its mean is taken
    only after assigning a common sign to the corresponding image block.
    """
    rng = np.random.default_rng(stable_seed(random_seed, "signflip"))
    classes = np.unique(truth)
    mean_over_resamples = effect_tensor.mean(axis=1, dtype=np.float64)
    weights = np.empty(truth.size, dtype=np.float64)
    for class_id in classes:
        mask = truth == class_id
        weights[mask] = 1.0 / (classes.size * int(mask.sum()))
    contributions = (mean_over_resamples * weights[None, :]).T

    null = np.empty((replicates, effect_tensor.shape[0]), dtype=np.float32)
    exceedances = np.zeros(effect_tensor.shape[0], dtype=np.int64)
    threshold = np.abs(observed)
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        signs = rng.integers(0, 2, size=(count, truth.size), dtype=np.int8)
        signs = signs.astype(np.float32) * 2.0 - 1.0
        statistics = signs @ contributions
        null[start : start + count] = statistics.astype(np.float32)
        exceedances += np.sum(np.abs(statistics) >= threshold[None, :], axis=0)
    raw_p = (exceedances + 1.0) / (replicates + 1.0)
    return raw_p, null


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "effects": output_dir / "adjacent_effects.csv",
        "bootstrap": output_dir / "adjacent_bootstrap.csv",
        "signflip": output_dir / "adjacent_signflip.csv",
        "results": output_dir / "adjacent_results.csv",
        "bootstrap_draws": output_dir / "adjacent_bootstrap_draws.npz",
        "signflip_null": output_dir / "adjacent_signflip_null.npz",
        "inputs": output_dir / "bootstrap_inputs.npz",
        "figure": output_dir / "figure_adjacent_ci.pdf",
        "manifest": output_dir / "run_manifest.json",
        "log": output_dir / "run.log",
    }


def ensure_output_safety(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(existing)
        raise FileExistsError(
            "Refusing to overwrite existing experiment outputs. Use a new "
            f"--output-dir or pass --overwrite:\n  {joined}"
        )


def make_figure(results: pd.DataFrame, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["0->1", "1->2", "2->3", "3->4"]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True, sharey=True)
    for axis, backbone, ranker in zip(
        axes.flat,
        ("ViT-B-16", "ViT-B-16", "ViT-B-32", "ViT-B-32"),
        ("positive_mean", "contrastive_margin", "positive_mean", "contrastive_margin"),
    ):
        subset = results.loc[
            (results["backbone"] == backbone) & (results["ranker"] == ranker)
        ].set_index("transition").loc[labels]
        y = subset["delta_pp"].to_numpy()
        low = subset["ci95_low_pp"].to_numpy()
        high = subset["ci95_high_pp"].to_numpy()
        supported = subset["supported_decline"].to_numpy(dtype=bool)
        colors = np.where(supported, "#B2182B", "#2166AC")
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
        axis.errorbar(
            np.arange(4), y, yerr=np.vstack((y - low, high - y)),
            fmt="none", ecolor="#555555", elinewidth=1.1, capsize=3, zorder=2,
        )
        axis.scatter(np.arange(4), y, c=colors, s=30, zorder=3)
        axis.set_title(f"{backbone.replace('ViT-', '')} | {ranker.replace('_', ' ')}", fontsize=9)
        axis.set_xticks(np.arange(4), labels)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0, 0].set_ylabel("Adjacent change (pp)")
    axes[1, 0].set_ylabel("Adjacent change (pp)")
    axes[1, 0].set_xlabel("Descriptor-budget transition")
    axes[1, 1].set_xlabel("Descriptor-budget transition")
    fig.suptitle("Aircraft adjacent-budget effects (95% paired bootstrap CI)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0 or args.signflip_replicates <= 0:
        raise ValueError("Replication counts must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    test = validate_and_load(args.predictions)
    effect_tensor, metadata, image_ids, truth, classes = build_effect_tensor(test)
    if effect_tensor.shape != (16, 5, EXPECTED_TEST_IMAGES):
        raise RuntimeError(f"Unexpected effect tensor shape: {effect_tensor.shape}")

    print("PASS: corrected-clean Aircraft prediction preflight")
    print(f"  prediction_sha256={sha256(args.predictions)}")
    print(f"  effect_tensor_shape={effect_tensor.shape}")
    print(f"  classes={len(classes)}, shared_test_images={len(image_ids)}")
    print(f"  seeds={list(FROZEN_SEEDS)}")
    print("  contrast_family=16 adjacent-budget contrasts")
    if args.preflight:
        print("Preflight only: no numerical result was computed or written.")
        return

    paths = output_paths(args.output_dir)
    ensure_output_safety(paths, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_effects = macro_effects_by_seed(effect_tensor, truth)
    observed = seed_effects.mean(axis=1)
    bootstrap = paired_class_bootstrap(
        effect_tensor,
        truth,
        args.bootstrap_replicates,
        args.random_seed,
        args.chunk_size,
    )
    ci_low = np.quantile(bootstrap, 0.025, axis=0)
    ci_high = np.quantile(bootstrap, 0.975, axis=0)
    raw_p, signflip_null = image_block_signflip(
        effect_tensor,
        truth,
        observed,
        args.signflip_replicates,
        args.random_seed,
        args.chunk_size,
    )
    holm_p = holm_adjust(raw_p)

    effect_rows: list[dict] = []
    for contrast_index, meta in enumerate(metadata):
        for seed_index, seed in enumerate(FROZEN_SEEDS):
            effect_rows.append(
                {
                    **meta,
                    "seed": seed,
                    "delta_macro_accuracy": seed_effects[contrast_index, seed_index],
                    "delta_pp": 100.0 * seed_effects[contrast_index, seed_index],
                }
            )
    pd.DataFrame(effect_rows).to_csv(paths["effects"], index=False)

    bootstrap_rows = []
    signflip_rows = []
    result_rows = []
    for index, meta in enumerate(metadata):
        bootstrap_rows.append(
            {
                **meta,
                "delta_macro_accuracy": observed[index],
                "delta_pp": 100.0 * observed[index],
                "ci95_low": ci_low[index],
                "ci95_high": ci_high[index],
                "ci95_low_pp": 100.0 * ci_low[index],
                "ci95_high_pp": 100.0 * ci_high[index],
                "bootstrap_replicates": args.bootstrap_replicates,
                "interval": "percentile_95",
            }
        )
        signflip_rows.append(
            {
                **meta,
                "delta_macro_accuracy": observed[index],
                "raw_p_two_sided": raw_p[index],
                "holm_p_16": holm_p[index],
                "signflip_replicates": args.signflip_replicates,
                "p_value_correction": "Monte Carlo add-one",
            }
        )
        result_rows.append(
            {
                **meta,
                "delta_macro_accuracy": observed[index],
                "delta_pp": 100.0 * observed[index],
                "ci95_low": ci_low[index],
                "ci95_high": ci_high[index],
                "ci95_low_pp": 100.0 * ci_low[index],
                "ci95_high_pp": 100.0 * ci_high[index],
                "raw_p_two_sided": raw_p[index],
                "holm_p_16": holm_p[index],
                "ci_excludes_zero": bool(ci_low[index] > 0 or ci_high[index] < 0),
                "supported_decline": bool(
                    observed[index] < 0 and ci_high[index] < 0 and holm_p[index] < 0.05
                ),
            }
        )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    signflip_frame = pd.DataFrame(signflip_rows)
    results = pd.DataFrame(result_rows)
    bootstrap_frame.to_csv(paths["bootstrap"], index=False)
    signflip_frame.to_csv(paths["signflip"], index=False)
    results.to_csv(paths["results"], index=False)

    labels = np.asarray(
        [f"{m['backbone']}|{m['ranker']}|{m['transition']}" for m in metadata]
    )
    np.savez_compressed(
        paths["bootstrap_draws"],
        labels=labels,
        draws=bootstrap,
        random_seed=np.asarray([stable_seed(args.random_seed, "bootstrap")], dtype=np.uint64),
    )
    np.savez_compressed(
        paths["signflip_null"],
        labels=labels,
        null_statistics=signflip_null,
        random_seed=np.asarray([stable_seed(args.random_seed, "signflip")], dtype=np.uint64),
    )
    np.savez_compressed(
        paths["inputs"],
        labels=labels,
        seeds=np.asarray(FROZEN_SEEDS),
        image_ids=image_ids,
        true_index=truth,
        correctness_difference=effect_tensor,
    )
    make_figure(results, paths["figure"])

    manifest = {
        "analysis": "Aircraft adjacent-budget paired inference",
        "status": "completed",
        "input_predictions": str(args.predictions.resolve()),
        "input_predictions_sha256": sha256(args.predictions),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "frozen_seeds": list(FROZEN_SEEDS),
        "backbones": list(FROZEN_BACKBONES),
        "rankers": list(FROZEN_SCORERS),
        "budgets": list(FROZEN_BUDGETS),
        "transitions": [f"{a}->{b}" for a, b in TRANSITIONS],
        "contrast_family_size": 16,
        "bootstrap_replicates": args.bootstrap_replicates,
        "signflip_replicates": args.signflip_replicates,
        "master_random_seed": args.random_seed,
        "bootstrap_seed": stable_seed(args.random_seed, "bootstrap"),
        "signflip_seed": stable_seed(args.random_seed, "signflip"),
        "holm_family": "all 16 adjacent-budget contrasts",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "result_files": {key: path.name for key, path in paths.items() if key != "manifest"},
        "declaration": (
            "This analysis was specified in response to reviewer comments before "
            "its numerical results were inspected. No result-dependent threshold, "
            "seed, contrast family, or reporting rule was changed after execution."
        ),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["log"].write_text(
        "\n".join(
            [
                "PASS: Experiment 1 completed.",
                f"input={args.predictions.resolve()}",
                f"input_sha256={manifest['input_predictions_sha256']}",
                f"script_sha256={manifest['script_sha256']}",
                f"bootstrap_replicates={args.bootstrap_replicates}",
                f"signflip_replicates={args.signflip_replicates}",
                "holm_family=all 16 adjacent-budget contrasts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PASS: results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
