"""Post hoc base-matched extension to Aircraft Experiment 5.

Uses frozen image features, the existing matched-analysis rankings and local
frozen CLIP text encoders. It performs no image encoding, training or test-image
loading. It contrasts one recomposed text prototype with an unweighted mean of
scores from separate prompts that each contain the same base description and
one selected descriptor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aircraft_cached_core import (
    BACKBONES,
    ROOT,
    encode_text,
    load_context,
    macro_accuracy,
    official_ids,
)

SEEDS = (20261003, 20261004, 20261005, 20261006, 20261007)
K_VALUES = (1, 2, 3, 4)
OUT_DEFAULT = ROOT / "revision_experiments" / "analysis_package_v13" / "RESULTS" / "experiment_5_base_matched_addendum"
RANKINGS_DEFAULT = ROOT / "revision_experiments" / "analysis_package_v13" / "RESULTS" / "experiment_5_matched_prior" / "rankings.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, default=RANKINGS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def ranking_orders(frame: pd.DataFrame, backbone: str, seed: int, n_classes: int) -> list[list[int]]:
    rows = frame[(frame.analysis == "matched") & frame.backbone.eq(backbone) & frame.seed.eq(seed)]
    result: list[list[int]] = []
    for class_id in range(n_classes):
        group = rows[rows.class_id.eq(class_id)].sort_values("rank")
        if len(group) != 4 or group["rank"].tolist() != [1, 2, 3, 4]:
            raise RuntimeError(f"Expected four frozen descriptor ranks for {backbone}, seed={seed}, class={class_id}")
        ids = group.descriptor_id.astype(int).tolist()
        if sorted(ids) != [1, 2, 3, 4]:
            raise RuntimeError(f"Unexpected descriptor IDs for {backbone}, seed={seed}, class={class_id}: {ids}")
        result.append([value - 1 for value in ids])
    if rows.class_id.nunique() != n_classes:
        raise RuntimeError("Frozen ranking file does not cover every class")
    return result


def make_prompts(classes: list[dict]) -> list[list[str]]:
    prompts = []
    for record in classes:
        base = record["base_description"].strip().rstrip(".!? ")
        prompts.append([f"{base}. Diagnostic attribute: {descriptor}." for descriptor in record["descriptors"]])
    return prompts


def parse_model_map(root: Path) -> dict:
    candidates = [
        root / "external_aircraft_corrected_clean" / "model_fingerprints.json",
        root / "external_aircraft_clean" / "protocol" / "model_fingerprints_release_mapping.json",
        root / "external_aircraft" / "model_fingerprints.json",
    ]
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "models" in payload:
                payload = payload["models"]
            return payload
    raise FileNotFoundError("Frozen checkpoint mapping was not found under the resolved VLM root")


def main() -> None:
    a = args()
    ranking_path = a.rankings.resolve()
    resource_path = ROOT / "external_aircraft_clean" / "resources" / "aircraft_descriptors_clean.json"
    if not resource_path.is_file():
        resource_path = ROOT / "external_aircraft" / "resources" / "aircraft_descriptors_clean.json"
    cache_paths = [ROOT / "external_aircraft_clean" / "feature_cache" / f"image_features_{b}.npz" for b in BACKBONES]
    if any(not path.is_file() for path in [ranking_path, resource_path, *cache_paths]):
        raise FileNotFoundError("Frozen resource, ranking or feature-cache input is missing")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty addendum output: {a.output_dir}")

    resource_hash = sha256(resource_path)
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    classes = resource["classes"]
    rankings = pd.read_csv(ranking_path)
    if rankings.analysis.eq("matched").sum() != len(BACKBONES) * len(SEEDS) * len(classes) * 4:
        raise RuntimeError("Frozen matched ranking rows do not match 2 backbones x 5 seeds x classes x 4 descriptors")
    for backbone in BACKBONES:
        for seed in SEEDS:
            ranking_orders(rankings, backbone, seed, len(classes))

    model_map = parse_model_map(ROOT)
    expected_models = {}
    for backbone in BACKBONES:
        info = model_map[backbone]
        model_path = Path(info["path"])
        if not model_path.is_absolute() or not model_path.exists():
            options = [ROOT / model_path, *(parent / model_path for parent in ROOT.parents)]
            model_path = next((path for path in options if path.exists()), options[0])
        if not model_path.is_dir():
            raise FileNotFoundError(f"Frozen local checkpoint not found: {model_path}")
        expected_files = info.get("files", {})
        for name, expected in expected_files.items():
            actual_path = model_path / name
            if not actual_path.is_file() or sha256(actual_path) != expected:
                raise RuntimeError(f"Checkpoint fingerprint mismatch: {backbone}/{name}")
        expected_models[backbone] = {"path": str(model_path), "files": expected_files}

    if a.preflight:
        print("PASS: addendum inputs, 20,000 frozen ranking rows, clean-resource hash and local checkpoint fingerprints verified")
        return

    curve_rows: list[dict] = []
    prediction_rows: list[dict] = []
    prompt_rows: list[dict] = []
    for backbone in BACKBONES:
        context = load_context(backbone, device="cpu")
        test_ids = official_ids("test")
        positions = torch.tensor([context["positions"][image_id] for image_id in test_ids], dtype=torch.long)
        test_features = context["features"][positions]
        truth = context["labels"][positions.numpy()]
        prompt_bank = make_prompts(context["classes"])
        encoded = [encode_text(context, by_class) for by_class in prompt_bank]

        for class_id, record in enumerate(context["classes"]):
            for descriptor_id, prompt in enumerate(prompt_bank[class_id], start=1):
                prompt_rows.append({
                    "backbone": backbone,
                    "class_id": class_id,
                    "class_name": record["name_en"],
                    "descriptor_id": descriptor_id,
                    "prompt": prompt,
                    "prompt_template": "{base_description}. Diagnostic attribute: {descriptor}.",
                })

        for seed in SEEDS:
            orders = ranking_orders(rankings, backbone, seed, len(context["classes"]))
            for k in K_VALUES:
                class_scores = []
                for class_id, order in enumerate(orders):
                    selected = encoded[class_id][order[:k]]
                    class_scores.append((test_features @ selected.T).mean(dim=1).numpy())
                scores = np.stack(class_scores, axis=1)
                predicted = scores.argmax(axis=1)
                metric = macro_accuracy(truth, predicted, len(context["classes"]))
                curve_rows.append({
                    "domain": "aircraft",
                    "backbone": backbone,
                    "seed": seed,
                    "ranker": "positive_mean",
                    "k": k,
                    "interface": "base_matched_per_descriptor_ensemble",
                    "test_macro_accuracy": metric,
                    "device": "cpu",
                    "analysis_status": "post_hoc_addendum_not_in_original_five_analysis_protocol",
                })
                prediction_rows.extend({
                    "analysis_status": "post_hoc_addendum_not_in_original_five_analysis_protocol",
                    "backbone": backbone,
                    "seed": seed,
                    "k": k,
                    "interface": "base_matched_per_descriptor_ensemble",
                    "image_id": image_id,
                    "true_index": int(target),
                    "predicted_index": int(prediction),
                } for image_id, target, prediction in zip(test_ids, truth, predicted))

    a.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(a.output_dir / "base_matched_interface_curves.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(a.output_dir / "base_matched_predictions.csv", index=False)
    pd.DataFrame(prompt_rows).to_csv(a.output_dir / "base_matched_exact_prompts.csv", index=False)
    manifest = {
        "analysis_status": "post_hoc_addendum_not_in_original_five_analysis_protocol",
        "scientific_question": "With the same class base description, ranked descriptors, support split, backbone, k, frozen test set and image features, does one joint text encoding differ from mean scores over separate base-matched descriptor prompts?",
        "prompt_template": "{base_description}. Diagnostic attribute: {descriptor}.",
        "aggregation": "unweighted mean of cosine-similarity scores from the k selected per-descriptor text embeddings, followed by argmax across classes",
        "rankings_source": str(ranking_path),
        "rankings_sha256": sha256(ranking_path),
        "resource_path": str(resource_path),
        "resource_sha256": resource_hash,
        "feature_cache_sha256": {b: sha256(p) for b, p in zip(BACKBONES, cache_paths)},
        "checkpoint_fingerprints": expected_models,
        "support_seeds": list(SEEDS),
        "k_values": list(K_VALUES),
        "n_test_images": len(test_ids),
        "device": "cpu",
        "image_model_inference": "none; uses frozen cached image features and local text encoding only",
        "test_images_opened": False,
    }
    (a.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: base-matched post hoc curves written to {a.output_dir.resolve()}")


if __name__ == "__main__":
    main()
