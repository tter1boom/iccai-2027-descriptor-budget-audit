"""Run one disjoint quarter of frozen Aircraft Experiment 3 on one GPU.

Each support seed is an independent resumable shard. The CLIP text encoder runs
on the assigned CUDA device; all image features come from the frozen cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aircraft_cached_core import (
    BACKBONES, SCORERS, ROOT, build_recomposed_prompts, descriptor_features,
    encode_text, load_context, macro_accuracy, official_ids, predict,
    rank_descriptors,
)
from run_exp3_validation_size import SUPPORT_SEEDS, PERMUTATION_SEEDS, VALIDATION_SIZES, class_sample


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def expected_files(shard: Path) -> list[Path]:
    return [
        shard / "all_selections.csv", shard / "rankings.csv",
        shard / "test_curves.csv", shard / "exact_prompts.csv",
        shard / "predictions_positive_mean.npz",
        shard / "predictions_contrastive_margin.npz",
    ]


def completed(shard: Path, worker_index: int) -> bool:
    marker = shard / "complete.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    return (record.get("worker_index") == worker_index
            and record.get("worker_script_sha256") == digest(Path(__file__))
            and record.get("core_script_sha256") == digest(Path(__file__).with_name("aircraft_cached_core.py"))
            and all(path.is_file() and record["files"].get(path.name) == digest(path)
                    for path in expected_files(shard)))


def run_seed(context: dict, backbone: str, seed: int, attrs: list[torch.Tensor],
             train_ids: list[str], train_y: np.ndarray, val_ids: list[str],
             val_y: np.ndarray, val_features: torch.Tensor, test_ids: list[str],
             test_y: np.ndarray, test_features: torch.Tensor, shard: Path,
             worker_index: int) -> None:
    support_ids = class_sample(train_ids, train_y, 5, np.random.default_rng(seed))
    if len(support_ids) != 500 or len(set(support_ids)) != 500:
        raise RuntimeError(f"Invalid five-shot support sample for seed {seed}")

    selections: list[dict] = []
    rankings: list[dict] = []
    curves: list[dict] = []
    prompts_audit: list[dict] = []
    by_class = {c: np.flatnonzero(val_y == c) for c in range(100)}
    if any(len(idx) < 20 for idx in by_class.values()):
        raise RuntimeError("Official Aircraft validation split has a class with fewer than 20 images")

    for scorer in SCORERS:
        orders = rank_descriptors(context, support_ids, scorer, attrs)
        for class_id, order in enumerate(orders):
            for rank, descriptor_index in enumerate(order, 1):
                rankings.append(dict(backbone=backbone, ranker=scorer, support_seed=seed,
                                     class_id=class_id, rank=rank,
                                     descriptor_id=descriptor_index + 1))

        val_pred: dict[int, np.ndarray] = {}
        test_pred: dict[int, np.ndarray] = {}
        test_acc: dict[int, float] = {}
        for k in range(5):
            prompts = build_recomposed_prompts(context["classes"], orders, k)
            text_features = encode_text(context, prompts)
            val_pred[k], _ = predict(val_features, text_features)
            test_pred[k], _ = predict(test_features, text_features)
            test_acc[k] = macro_accuracy(test_y, test_pred[k])
            curves.append(dict(backbone=backbone, ranker=scorer, support_seed=seed,
                               k=k, test_macro_accuracy=test_acc[k]))
            for class_id, prompt in enumerate(prompts):
                prompts_audit.append(dict(backbone=backbone, ranker=scorer,
                                          support_seed=seed, k=k, class_id=class_id,
                                          final_prompt=prompt,
                                          prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest()))

        np.savez_compressed(
            shard / f"predictions_{scorer}.npz",
            validation_image_id=np.asarray(val_ids), validation_true_index=val_y,
            validation_predicted_index=np.stack([val_pred[k] for k in range(5)]),
            test_image_id=np.asarray(test_ids), test_true_index=test_y,
            test_predicted_index=np.stack([test_pred[k] for k in range(5)]),
            budgets=np.arange(5),
        )

        oracle_acc = max(test_acc.values())
        oracle_k = min(k for k, value in test_acc.items() if value == oracle_acc)
        second = sorted(test_acc.values(), reverse=True)[1]
        for permutation_index, permutation_seed in enumerate(PERMUTATION_SEEDS):
            rng = np.random.default_rng(seed * 1000003 + permutation_seed)
            permutations = {c: rng.permutation(idx) for c, idx in by_class.items()}
            for nv in VALIDATION_SIZES:
                indices = np.concatenate([
                    permutations[c] if nv == "all" else permutations[c][:int(nv)]
                    for c in range(100)
                ])
                scores = {k: macro_accuracy(val_y[indices], p[indices])
                          for k, p in val_pred.items()}
                best_validation = max(scores.values())
                selected_k = min(k for k, value in scores.items()
                                 if value == best_validation)
                selected_acc = test_acc[selected_k]
                regret = oracle_acc - selected_acc
                selections.append(dict(
                    backbone=backbone, ranker=scorer, support_seed=seed,
                    validation_permutation=permutation_index,
                    validation_permutation_seed=permutation_seed,
                    n_validation_per_class=str(nv), selected_k=selected_k,
                    selected_validation_macro_accuracy=best_validation,
                    selected_test_macro_accuracy=selected_acc,
                    fixed_k1_test_macro_accuracy=test_acc[1],
                    fixed_k4_test_macro_accuracy=test_acc[4], oracle_k=oracle_k,
                    oracle_test_macro_accuracy=oracle_acc, oracle_regret=regret,
                    oracle_gap_best_minus_second=oracle_acc - second,
                    exact_oracle_k=selected_k == oracle_k,
                    regret_gt_0_5pp=regret > .005,
                    regret_gt_1pp=regret > .01,
                ))

    pd.DataFrame(selections).to_csv(shard / "all_selections.csv", index=False)
    pd.DataFrame(rankings).to_csv(shard / "rankings.csv", index=False)
    pd.DataFrame(curves).to_csv(shard / "test_curves.csv", index=False)
    pd.DataFrame(prompts_audit).to_csv(shard / "exact_prompts.csv", index=False)
    marker = {
        "backbone": backbone, "support_seed": seed,
        "worker_index": worker_index, "device": context["device"],
        "worker_script_sha256": digest(Path(__file__)),
        "core_script_sha256": digest(Path(__file__).with_name("aircraft_cached_core.py")),
        "files": {path.name: digest(path) for path in expected_files(shard)},
        "counts": {"selection_rows": len(selections), "ranking_rows": len(rankings),
                   "curve_rows": len(curves), "prompt_rows": len(prompts_audit)},
    }
    if marker["counts"] != {"selection_rows": 600, "ranking_rows": 800,
                            "curve_rows": 10, "prompt_rows": 1000}:
        raise RuntimeError(f"Unexpected shard counts: {marker['counts']}")
    (shard / "complete.json").write_text(json.dumps(marker, indent=2) + "\n",
                                         encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers != 4 or args.worker_index not in range(4):
        raise ValueError("Frozen server configuration requires four workers indexed 0..3")
    seeds = SUPPORT_SEEDS[args.worker_index::args.workers]
    if len(seeds) != 5:
        raise RuntimeError("Support seeds were not partitioned into five per worker")
    if args.preflight:
        print(f"PASS worker={args.worker_index} device={args.device} seeds={list(seeds)}")
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU worker requested CUDA, but this Python environment has no CUDA")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "PROGRESS.txt"
    done = 0
    started = time.time()
    for backbone in BACKBONES:
        pending = [seed for seed in seeds if not completed(
            output / "shards" / f"{backbone}_seed{seed}", args.worker_index)]
        done += len(seeds) - len(pending)
        if not pending:
            continue
        context = load_context(backbone, device=args.device)
        attrs = descriptor_features(context)
        train_ids, val_ids, test_ids = (official_ids(split)
                                        for split in ("train", "val", "test"))
        positions = context["positions"]
        train_y = np.asarray([context["labels"][positions[x]] for x in train_ids])
        val_y = np.asarray([context["labels"][positions[x]] for x in val_ids])
        test_y = np.asarray([context["labels"][positions[x]] for x in test_ids])
        val_features = context["features"][torch.tensor([positions[x] for x in val_ids])]
        test_features = context["features"][torch.tensor([positions[x] for x in test_ids])]
        for seed in pending:
            shard = output / "shards" / f"{backbone}_seed{seed}"
            shard.mkdir(parents=True, exist_ok=True)
            run_seed(context, backbone, seed, attrs, train_ids, train_y, val_ids,
                     val_y, val_features, test_ids, test_y, test_features,
                     shard, args.worker_index)
            done += 1
            line = (f"worker={args.worker_index} completed={done}/10 "
                    f"last={backbone}_seed{seed} elapsed_seconds={time.time()-started:.1f}")
            progress.write_text(line + "\n", encoding="utf-8")
            print(line, flush=True)
        del context
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    progress.write_text(f"worker={args.worker_index} status=PASS completed=10/10\n",
                        encoding="utf-8")
    print(f"PASS worker {args.worker_index}", flush=True)


if __name__ == "__main__":
    main()
