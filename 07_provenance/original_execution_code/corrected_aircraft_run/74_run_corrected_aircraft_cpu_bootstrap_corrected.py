"""Run the protocol-frozen, double-audited FGVC-Aircraft experiment on CPU.

This runner is deliberately self-contained: it first writes a machine-readable
pre-run report and refuses to load a test image unless every required check
passes.  It never writes or normalizes descriptor text; the applied JSON is
consumed byte-for-byte from ``external_aircraft/resources``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261003, 20261004, 20261005, 20261006, 20261007)
BACKBONES = ("ViT-B-16", "ViT-B-32")
SCORERS = ("positive_mean", "contrastive_margin")
EXPECTED_RESOURCE_SHA = "67876cc858a9b6a1785b7eaa229a54dd40ad3394de7a7c989ad0d77b803808fc"
EXPECTED_AUDIT_SHA = "bdb085a4bbfaa1ec8a732e2ec0930388cfe07bdee5e726391ef0221ff18433b5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_annotations(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        image_id, sep, class_name = line.partition(" ")
        if not sep or not image_id or not class_name.strip():
            raise ValueError(f"Malformed annotation: {path}:{line_no}")
        rows.append((image_id, class_name.strip()))
    return rows


def build_prompt(record: dict, selected: set[int]) -> str:
    """Use exact base text at k=0 and catalogue order for every selected set."""
    base = record["base_description"]
    if not selected:
        return base
    values = [value for index, value in enumerate(record["descriptors"]) if index in selected]
    return base + ". Diagnostic attributes: " + "; ".join(values) + "."


def token_lengths(tokenizer: CLIPTokenizer, prompts: list[str]) -> list[int]:
    return [len(ids) for ids in tokenizer(prompts, add_special_tokens=True, truncation=False)["input_ids"]]


def load_and_check_catalogue(resource: Path, audit_csv: Path, names: list[str]) -> tuple[dict, list[dict], list[str]]:
    errors: list[str] = []
    if sha256(resource) != EXPECTED_RESOURCE_SHA:
        errors.append("applied descriptor JSON SHA-256 does not match authority")
    if sha256(audit_csv) != EXPECTED_AUDIT_SHA:
        errors.append("applied descriptor audit CSV SHA-256 does not match authority")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    ids = [record.get("class_id") for record in classes]
    if len(classes) != 100 or ids != list(range(100)):
        errors.append("class IDs are not exactly 0..99 once each")
    if [record.get("name_en") for record in classes] != names:
        errors.append("class names/order do not match official dataset audit")
    if any(record.get("review_status") != "verified_double_checked" for record in classes):
        errors.append("one or more class review statuses are not verified_double_checked")
    seen_rows = []
    for record in classes:
        descriptors = record.get("descriptors")
        if not isinstance(descriptors, list) or len(descriptors) != 4:
            errors.append(f"{record.get('name_en')}: does not have exactly four descriptors")
            continue
        if len(set(descriptors)) != 4:
            errors.append(f"{record['name_en']}: within-class duplicate descriptor")
        normalized_name = record["name_en"].lower()
        for slot, descriptor in enumerate(descriptors, 1):
            seen_rows.append((record["class_id"], slot))
            if normalized_name in descriptor.lower():
                errors.append(f"{record['name_en']}/d{slot}: class-name copied into descriptor")
    if len(seen_rows) != 400 or len(set(seen_rows)) != 400:
        errors.append("catalogue does not contain 400 unique descriptor slots")
    with audit_csv.open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if len(audit_rows) != 400:
        errors.append("audit CSV does not contain 400 rows")
    else:
        expected_slots = {(str(class_id), f"d{slot}") for class_id in range(100) for slot in range(1, 5)}
        got_slots = {(row.get("class_id", ""), row.get("descriptor_id", "")) for row in audit_rows}
        if got_slots != expected_slots:
            errors.append("audit CSV class/descriptor slots are incomplete or duplicated")
        pending_words = ("pending", "draft", "review")
        for row in audit_rows:
            statuses = (row.get("pass1_status", ""), row.get("pass2_status", ""))
            if any(any(word in value.lower() for word in pending_words) for value in statuses):
                errors.append("audit CSV contains pending/draft/review status")
                break
        expected_text = {(str(record["class_id"]), f"d{slot}"): descriptor
                         for record in classes for slot, descriptor in enumerate(record["descriptors"], 1)}
        for row in audit_rows:
            key = (row.get("class_id", ""), row.get("descriptor_id", ""))
            if expected_text.get(key) != row.get("descriptor_text") or row.get("final_text") != row.get("descriptor_text"):
                errors.append("audit CSV text is not field-by-field identical to applied JSON")
                break
    return payload, classes, errors


def split_check(protocol_dir: Path, dataset_dir: Path, names: list[str]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    annotations = {split: read_annotations(dataset_dir / "data" / f"images_variant_{split}.txt")
                   for split in ("train", "val", "test")}
    official = {split: {image_id for image_id, _ in rows} for split, rows in annotations.items()}
    if official["train"] & official["val"] or official["train"] & official["test"] or official["val"] & official["test"]:
        errors.append("official split IDs overlap")
    manifests = {}
    for seed in SEEDS:
        path = protocol_dir / f"split_seed_{seed}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests[seed] = manifest
        support = [image_id for name in names for image_id in manifest["support"].get(name, [])]
        validation = [image_id for name in names for image_id in manifest["validation"].get(name, [])]
        if len(support) != 500 or len(validation) != 500:
            errors.append(f"seed {seed}: not exactly 5 support + 5 validation IDs per class")
        if len(set(support)) != 500 or len(set(validation)) != 500:
            errors.append(f"seed {seed}: duplicate support/validation ID")
        if not set(support) <= official["train"]:
            errors.append(f"seed {seed}: support contains non-train ID")
        if not set(validation) <= official["val"]:
            errors.append(f"seed {seed}: validation contains non-val ID")
        if (set(support) | set(validation)) & official["test"]:
            errors.append(f"seed {seed}: official test ID occurs in support or validation")
    return {"annotations": annotations, "official_ids": official, "manifests": manifests}, errors


def model_and_token_check(classes: list[dict], config_path: Path, output: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    fingerprints = json.loads(config_path.read_text(encoding="utf-8"))
    token_report = {}
    for backbone in BACKBONES:
        info = fingerprints.get(backbone)
        if not info:
            errors.append(f"missing fingerprint for {backbone}")
            continue
        model_path = Path(info["path"])
        if info.get("device_policy") != "cpu":
            errors.append(f"{backbone}: non-CPU model policy")
        for file_name, expected in info.get("files", {}).items():
            path = model_path / file_name
            if not path.is_file() or sha256(path) != expected:
                errors.append(f"{backbone}: checkpoint/preprocessor fingerprint mismatch for {file_name}")
        required_tokenizer = (model_path / "vocab.json", model_path / "merges.txt")
        if not all(path.is_file() for path in required_tokenizer):
            errors.append(f"{backbone}: local CLIP tokenizer files missing; fallback is forbidden")
            continue
        try:
            tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
            model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
            limit = model_config["text_config"]["max_position_embeddings"]
        except Exception as exc:  # preflight must record, not hide, a local-loader failure
            errors.append(f"{backbone}: local tokenizer/config load failed: {type(exc).__name__}: {exc}")
            continue
        by_k = {}
        for k in range(5):
            prompts = [build_prompt(record, set(range(k))) for record in classes]
            lengths = token_lengths(tokenizer, prompts)
            by_k[str(k)] = {"max_tokens": max(lengths), "min_tokens": min(lengths), "limit": limit}
            if max(lengths) > limit:
                errors.append(f"{backbone}: k={k} prompt token limit exceeded")
        full_prompts = [build_prompt(record, {0, 1, 2, 3}) for record in classes]
        # Full membership is all catalogue slots under both scorers; compare serializations directly.
        if full_prompts != [build_prompt(record, {0, 1, 2, 3}) for record in classes]:
            errors.append(f"{backbone}: k=4 scorer serialization mismatch")
        token_report[backbone] = {"tokenizer_class": tokenizer.__class__.__name__, "local_path": str(model_path), "k": by_k,
                                  "k4_identical_across_scorers": True}
    output.write_text(json.dumps(token_report, indent=2) + "\n", encoding="utf-8")
    return fingerprints, errors


def preflight(resource: Path, audit_csv: Path, protocol_dir: Path, config_path: Path, dataset_dir: Path, output: Path) -> tuple[dict, list[dict], dict]:
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((protocol_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    names = audit["classes_lexicographic"]
    payload, classes, errors = load_and_check_catalogue(resource, audit_csv, names)
    split_data, split_errors = split_check(protocol_dir, dataset_dir, names)
    token_file = output / "token_audit.json"
    fingerprints, model_errors = model_and_token_check(classes, config_path, token_file)
    errors.extend(split_errors)
    errors.extend(model_errors)
    report = {
        "run_id": output.name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resource": str(resource), "resource_sha256": sha256(resource),
        "audit_csv": str(audit_csv), "audit_sha256": sha256(audit_csv),
        "class_count": len(classes), "descriptor_count": sum(len(record.get("descriptors", [])) for record in classes),
        "dataset_audit_sha256": sha256(protocol_dir / "dataset_audit.json"),
        "split_manifest_sha256": {str(seed): sha256(protocol_dir / f"split_seed_{seed}.json") for seed in SEEDS},
        "model_fingerprints_sha256": sha256(config_path), "status": "PASS" if not errors else "FAIL", "errors": errors,
        "cpu_only": True, "python": sys.version, "platform": platform.platform(),
        "no_test_access_before_status": "PASS",
    }
    (output / "pre_run_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("PRE-RUN QA FAILED: " + " | ".join(errors))
    frozen = {
        "protocol_version": "fgvc_aircraft_corrected_clean_double_audited_v1",
        "resource_sha256": report["resource_sha256"], "audit_sha256": report["audit_sha256"],
        "dataset_audit_sha256": report["dataset_audit_sha256"], "split_manifest_sha256": report["split_manifest_sha256"],
        "model_fingerprints_sha256": report["model_fingerprints_sha256"], "test_access": "unlocked_after_pre_run_checks_PASS",
        "catalogue_order": "d1,d2,d3,d4", "device": "cpu", "metric": "macro/class-balanced accuracy",
    }
    (output / "PROTOCOL_CORRECTED_CLEAN_FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return payload, classes, {**split_data, "names": names, "fingerprints": fingerprints}


def encode_text(model: CLIPModel, tokenizer: CLIPTokenizer, prompts: list[str]) -> torch.Tensor:
    limit = model.config.text_config.max_position_embeddings
    if max(token_lengths(tokenizer, prompts)) > limit:
        raise RuntimeError("Prompt token limit exceeded after preflight")
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    with torch.no_grad():
        text = model.text_model(**batch)
        positions = batch["attention_mask"].sum(1) - 1
        features = model.text_projection(text.last_hidden_state[torch.arange(len(prompts)), positions])
    return F.normalize(features, dim=-1).cpu()


def image_features(model: CLIPModel, processor: CLIPProcessor, images: list[tuple[str, str, str]], cache: Path, log) -> tuple[torch.Tensor, list[str]]:
    rows, ids = [], [image_id for image_id, _, _ in images]
    for start in range(0, len(images), 16):
        batch = []
        for image_id, _, _ in images[start:start + 16]:
            with Image.open(ROOT / "external_datasets/fgvc-aircraft-2013b/data/images" / f"{image_id}.jpg") as image:
                image = image.convert("RGB")
                if image.height <= 20:
                    raise RuntimeError(f"banner crop impossible: {image_id}")
                batch.append(image.crop((0, 0, image.width, image.height - 20)))
        encoded = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**encoded)
            if not torch.is_tensor(features):
                # Transformers 5 returns a BaseModelOutputWithPooling here on
                # this local checkpoint, whereas earlier installed versions
                # returned the projected tensor directly.
                features = features.pooler_output if getattr(features, "pooler_output", None) is not None else features.last_hidden_state[:, 0]
            rows.append(F.normalize(features, dim=-1).cpu())
        if start % 160 == 0 or start + 16 >= len(images):
            log(f"{cache.stem}: encoded {min(start + 16, len(images))}/{len(images)} images")
    values = torch.cat(rows)
    np.savez_compressed(cache, features=values.numpy(), image_ids=np.asarray(ids))
    return values, ids


def rank_attributes(image: torch.Tensor, labels: torch.Tensor, attributes: list[torch.Tensor], scorer: str) -> list[list[int]]:
    orders = []
    for class_id, feature_set in enumerate(attributes):
        own, other = image[labels == class_id], image[labels != class_id]
        scores = []
        for descriptor_id, text_feature in enumerate(feature_set):
            positive, negative = float((own @ text_feature).mean()), float((other @ text_feature).mean())
            score = positive if scorer == "positive_mean" else positive - negative
            scores.append((score, descriptor_id, positive, negative))
        scores.sort(key=lambda item: (-item[0], item[1]))
        orders.append([item[1] for item in scores])
    return orders


def macro_accuracy(true: np.ndarray, predicted: np.ndarray, class_count: int) -> float:
    return float(recall_score(true, predicted, labels=np.arange(class_count), average="macro", zero_division=0))


def run(args, payload: dict, classes: list[dict], data: dict, output: Path, log) -> None:
    names, fingerprints = data["names"], data["fingerprints"]
    annotations = data["annotations"]
    all_images = [(image_id, class_name, split) for split in ("train", "val", "test") for image_id, class_name in annotations[split]]
    image_ids = [row[0] for row in all_images]
    labels = np.asarray([names.index(row[1]) for row in all_images], dtype=np.int64)
    positions = {image_id: index for index, image_id in enumerate(image_ids)}
    test_positions = torch.tensor([index for index, row in enumerate(all_images) if row[2] == "test"])
    curves, selected_rows, predictions, rankings, confusion_rows = [], [], [], [], []
    logits_dir, cache_dir = output / "logits", output / "feature_cache"
    logits_dir.mkdir(), cache_dir.mkdir()
    for backbone in BACKBONES:
        info = fingerprints[backbone]
        model_path = Path(info["path"])
        log(f"loading {backbone} from exact local checkpoint {model_path}")
        model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval()
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("CPU-only protocol violated")
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
        image, cached_ids = image_features(model, processor, all_images, cache_dir / f"image_features_{backbone}.npz", log)
        if cached_ids != image_ids:
            raise RuntimeError("newly created feature-cache IDs are not aligned")
        attributes = [encode_text(model, tokenizer, ["a photo of an aircraft showing " + descriptor + "." for descriptor in record["descriptors"]]) for record in classes]
        for seed in SEEDS:
            manifest = data["manifests"][seed]
            support_ids = [image_id for name in names for image_id in manifest["support"][name]]
            validation_ids = [image_id for name in names for image_id in manifest["validation"][name]]
            support_positions = torch.tensor([positions[image_id] for image_id in support_ids])
            validation_positions = torch.tensor([positions[image_id] for image_id in validation_ids])
            for scorer in SCORERS:
                order = rank_attributes(image[support_positions], torch.tensor(labels[support_positions]), attributes, scorer)
                rankings.extend({"seed": seed, "backbone": backbone, "scorer": scorer, "class_id": class_id,
                                 "class_name": names[class_id], "rank": rank + 1, "descriptor_id": f"d{descriptor_id + 1}",
                                 "descriptor_text": classes[class_id]["descriptors"][descriptor_id]}
                                for class_id, class_order in enumerate(order) for rank, descriptor_id in enumerate(class_order))
                validation_scores = {}
                for k in range(5):
                    prompts = [build_prompt(record, set(order[class_id][:k])) for class_id, record in enumerate(classes)]
                    text = encode_text(model, tokenizer, prompts)
                    for split, split_positions in (("validation", validation_positions), ("test", test_positions)):
                        score_matrix = (image[split_positions] @ text.T).numpy()
                        truth = labels[split_positions.numpy()]
                        predicted = score_matrix.argmax(1)
                        macro = macro_accuracy(truth, predicted, len(names))
                        ordinary = float(accuracy_score(truth, predicted))
                        curves.append({"seed": seed, "backbone": backbone, "scorer": scorer, "split": split, "k": k,
                                       "macro_accuracy": macro, "top1_accuracy_secondary": ordinary, "n": len(truth)})
                        if split == "validation":
                            validation_scores[k] = macro
                        if split == "test":
                            np.savez_compressed(logits_dir / f"logits_{backbone}_seed{seed}_{scorer}_k{k}.npz",
                                                image_id=np.asarray([image_ids[index] for index in split_positions.numpy()]),
                                                true_index=truth, logits=score_matrix.astype(np.float32))
                            confusion_rows.append({"seed": seed, "backbone": backbone, "scorer": scorer, "k": k,
                                                   "matrix": confusion_matrix(truth, predicted, labels=np.arange(len(names))).tolist()})
                        predictions.extend({"seed": seed, "backbone": backbone, "scorer": scorer, "split": split, "k": k,
                                            "image_id": image_ids[int(pos)], "true_index": int(true_id), "predicted_index": int(predicted_id)}
                                           for pos, true_id, predicted_id in zip(split_positions.numpy(), truth, predicted))
                selected_k = min(k for k, score in validation_scores.items() if score == max(validation_scores.values()))
                selected_rows.append({"seed": seed, "backbone": backbone, "scorer": scorer, "selected_k": selected_k,
                                      "validation_macro_accuracy": validation_scores[selected_k]})
                log(f"finished {backbone} seed={seed} scorer={scorer}; selected k={selected_k}")
    pd.DataFrame(curves).to_csv(output / "budget_curves.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output / "selected_budgets.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(rankings).to_csv(output / "ranked_descriptors.csv", index=False)
    (output / "confusion_matrices.json").write_text(json.dumps(confusion_rows) + "\n", encoding="utf-8")
    log("evaluation complete")


def bootstrap(output: Path, reps: int, log) -> None:
    predictions = pd.read_csv(output / "predictions.csv")
    selected = pd.read_csv(output / "selected_budgets.csv")
    test = predictions[predictions.split == "test"].sort_values("image_id")
    results, inputs = [], []
    for row in selected.itertuples(index=False):
        block = test[(test.seed == row.seed) & (test.backbone == row.backbone) & (test.scorer == row.scorer)]
        for contrast, left_k, right_k in (("selected_minus_k4", int(row.selected_k), 4),
                                          ("k1_minus_k4", 1, 4),
                                          ("selected_minus_k1", int(row.selected_k), 1)):
            left = block[block.k == left_k].sort_values("image_id")
            right = block[block.k == right_k].sort_values("image_id")
            if not np.array_equal(left.image_id.to_numpy(), right.image_id.to_numpy()):
                raise RuntimeError("paired bootstrap ID alignment failed")
            true = left.true_index.to_numpy()
            delta = (left.predicted_index.to_numpy() == true).astype(np.float32) - (right.predicted_index.to_numpy() == true).astype(np.float32)
            classes = np.unique(true)
            rng = np.random.default_rng(20261003 + int(row.seed) + left_k * 13 + right_k)
            values = []
            for start in range(0, reps, 500):
                n = min(500, reps - start)
                total = np.zeros(n, dtype=np.float32)
                for class_id in classes:
                    group = delta[true == class_id]
                    total += group[rng.integers(0, len(group), size=(n, len(group)))].mean(1) / len(classes)
                values.append(total)
            draws = np.concatenate(values)
            observed_macro = float(np.mean([delta[true == class_id].mean() for class_id in classes]))
            stem = f"{row.backbone}_seed{row.seed}_{row.scorer}_{contrast}"
            np.savez_compressed(output / "bootstrap_inputs" / f"{stem}.npz", image_id=left.image_id.to_numpy(), true_index=true, paired_delta=delta)
            np.savez_compressed(output / "bootstrap_outputs" / f"{stem}.npz", bootstrap_macro_accuracy_difference=draws)
            results.append({"seed": row.seed, "backbone": row.backbone, "scorer": row.scorer, "contrast": contrast,
                            "left_k": left_k, "right_k": right_k, "macro_accuracy_difference": observed_macro,
                            "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
                            "bootstrap_replicates": reps})
    pd.DataFrame(results).to_csv(output / "paired_bootstrap.csv", index=False)
    log(f"paired bootstrap complete: {reps} class-stratified replicates per contrast")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="external_aircraft_corrected_clean")
    parser.add_argument("--replicates", type=int, default=20000)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing corrected-run directory: {output}")
    resource = ROOT / "external_aircraft/resources/aircraft_descriptors_clean.json"
    audit_csv = ROOT / "external_aircraft/resources/aircraft_descriptor_audit.csv"
    protocol_dir = ROOT / "external_aircraft/protocol"
    config_path = ROOT / "external_aircraft/config/model_fingerprints.json"
    dataset_dir = ROOT / "external_datasets/fgvc-aircraft-2013b"
    payload, classes, data = preflight(resource, audit_csv, protocol_dir, config_path, dataset_dir, output)
    (output / "execution_scripts").mkdir()
    shutil.copy2(Path(__file__), output / "execution_scripts" / Path(__file__).name)
    shutil.copy2(ROOT / "experiments/71_fgvc_paired_bootstrap.py", output / "execution_scripts/71_fgvc_paired_bootstrap_reference.py")
    (output / "bootstrap_inputs").mkdir()
    (output / "bootstrap_outputs").mkdir()
    log_path = output / "run.log"
    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    log("pre-run QA PASS; test evaluation unlocked")
    run(args, payload, classes, data, output, log)
    bootstrap(output, args.replicates, log)
    log("CORRECTED CLEAN AIRCRAFT RUN COMPLETE")


if __name__ == "__main__":
    main()
