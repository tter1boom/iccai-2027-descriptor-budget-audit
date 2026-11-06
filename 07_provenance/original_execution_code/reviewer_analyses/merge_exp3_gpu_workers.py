"""Verify and merge the forty frozen Experiment 3 GPU shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aircraft_cached_core import BACKBONES, SCORERS
from run_exp3_validation_size import SUPPORT_SEEDS, PERMUTATION_SEEDS, VALIDATION_SIZES


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selected, ranking, curves, prompts = [], [], [], []
    records = []
    for backbone in BACKBONES:
        for seed in SUPPORT_SEEDS:
            index = (seed - SUPPORT_SEEDS[0]) % 4
            shard = (args.worker_root / f"worker_{index}" / "shards"
                     / f"{backbone}_seed{seed}")
            marker_path = shard / "complete.json"
            if not marker_path.is_file():
                raise FileNotFoundError(f"Incomplete GPU shard: {shard}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (marker["backbone"], marker["support_seed"], marker["worker_index"]) != (backbone, seed, index):
                raise RuntimeError(f"Shard identity mismatch: {shard}")
            for filename, expected in marker["files"].items():
                path = shard / filename
                if not path.is_file() or digest(path) != expected:
                    raise RuntimeError(f"Shard file/hash mismatch: {path}")
            with np.load(shard / "predictions_positive_mean.npz") as pm, np.load(
                    shard / "predictions_contrastive_margin.npz") as cm:
                for key in ("validation_image_id", "test_image_id", "validation_true_index", "test_true_index"):
                    if not np.array_equal(pm[key], cm[key]):
                        raise RuntimeError(f"PM/CM prediction alignment failed: {shard}/{key}")
                if len(pm["test_image_id"]) != 3333 or pm["test_predicted_index"].shape != (5, 3333):
                    raise RuntimeError(f"Unexpected test prediction dimensions: {shard}")
            selected.append(pd.read_csv(shard / "all_selections.csv"))
            ranking.append(pd.read_csv(shard / "rankings.csv"))
            curves.append(pd.read_csv(shard / "test_curves.csv"))
            prompts.append(pd.read_csv(shard / "exact_prompts.csv"))
            records.append(dict(backbone=backbone, support_seed=seed,
                                worker_index=index, marker_sha256=digest(marker_path)))

    selections = pd.concat(selected, ignore_index=True)
    rankings = pd.concat(ranking, ignore_index=True)
    test_curves = pd.concat(curves, ignore_index=True)
    exact_prompts = pd.concat(prompts, ignore_index=True)
    expected_counts = {"selections": 24000, "rankings": 32000,
                       "test_curves": 400, "exact_prompts": 40000}
    observed_counts = {"selections": len(selections), "rankings": len(rankings),
                       "test_curves": len(test_curves), "exact_prompts": len(exact_prompts)}
    if observed_counts != expected_counts:
        raise RuntimeError(f"Merged row-count mismatch: {observed_counts}")
    unique_keys = [
        (selections, ["backbone", "ranker", "support_seed", "validation_permutation", "n_validation_per_class"]),
        (rankings, ["backbone", "ranker", "support_seed", "class_id", "rank"]),
        (test_curves, ["backbone", "ranker", "support_seed", "k"]),
        (exact_prompts, ["backbone", "ranker", "support_seed", "k", "class_id"]),
    ]
    if any(frame.duplicated(keys).any() for frame, keys in unique_keys):
        raise RuntimeError("Duplicate condition key in merged shards")
    if set(selections["support_seed"]) != set(SUPPORT_SEEDS):
        raise RuntimeError("Support seed coverage mismatch")
    if set(selections["validation_permutation_seed"]) != set(PERMUTATION_SEEDS):
        raise RuntimeError("Validation permutation seed coverage mismatch")
    if set(selections["n_validation_per_class"].astype(str)) != set(map(str, VALIDATION_SIZES)):
        raise RuntimeError("Validation size coverage mismatch")
    if set(zip(selections["backbone"], selections["ranker"])) != {
            (backbone, scorer) for backbone in BACKBONES for scorer in SCORERS}:
        raise RuntimeError("Backbone/ranker coverage mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all_selections.csv": selections,
        "rankings.csv": rankings,
        "test_curves.csv": test_curves,
        "exact_prompts.csv": exact_prompts,
    }
    for name, frame in outputs.items():
        path = args.output_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite merged output: {path}")
        frame.to_csv(path, index=False)
    manifest = {
        "status": "PASS", "shards": len(records), "counts": observed_counts,
        "shard_markers": records,
        "output_sha256": {name: digest(args.output_dir / name) for name in outputs},
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(records)} GPU shards merged into {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
