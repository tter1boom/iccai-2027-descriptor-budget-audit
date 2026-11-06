"""Portable cached-feature utilities for the corrected Aircraft analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BACKBONES = ("ViT-B-16", "ViT-B-32")
SEEDS = (20261003, 20261004, 20261005, 20261006, 20261007)
RESOURCE = PACKAGE_ROOT / "02_resources" / "aircraft_descriptors_frozen.json"
RESOURCE_SHA256 = "67876cc858a9b6a1785b7eaa229a54dd40ad3394de7a7c989ad0d77b803808fc"
MODEL_LOCK = PACKAGE_ROOT / "01_protocol" / "model_lock.json"
LABELS = PACKAGE_ROOT / "01_protocol" / "aircraft_test_image_labels.csv"
FEATURE_CACHE = PACKAGE_ROOT / "04_data" / "feature_cache"
RANKINGS = PACKAGE_ROOT / "04_data" / "matched_rankings.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_context(backbone: str, models_root: Path, device: str = "cpu") -> dict:
    if backbone not in BACKBONES:
        raise ValueError(backbone)
    if sha256(RESOURCE) != RESOURCE_SHA256:
        raise RuntimeError("Aircraft descriptor-resource SHA-256 mismatch")
    resource = json.loads(RESOURCE.read_text(encoding="utf-8"))
    classes = resource["classes"]
    cache_path = FEATURE_CACHE / f"image_features_{backbone}.npz"
    cache = np.load(cache_path)
    ids = cache["image_ids"].astype(str).tolist()
    features = F.normalize(torch.from_numpy(cache["features"].astype(np.float32)), dim=-1)
    labels_frame = pd.read_csv(LABELS, dtype={"image_id": str})
    if len(labels_frame) != 3333 or labels_frame.image_id.nunique() != 3333:
        raise RuntimeError("The official Aircraft test-label map must contain 3,333 unique IDs")
    id_to_position = {image_id: i for i, image_id in enumerate(ids)}
    test_ids = labels_frame.image_id.astype(str).tolist()
    if any(image_id not in id_to_position for image_id in test_ids):
        raise RuntimeError("Test-label IDs do not align with the included frozen feature cache")
    labels = labels_frame.true_index.to_numpy(dtype=np.int64)
    lock = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))["models"][backbone]
    model_path = (models_root / lock["relative_path"]).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Frozen local checkpoint is unavailable: {lock['relative_path']}")
    for name, expected in lock["files"].items():
        file_path = model_path / name
        if not file_path.is_file() or sha256(file_path) != expected:
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {backbone}/{name}")
    model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
    positions = np.asarray([id_to_position[x] for x in test_ids], dtype=np.int64)
    return {
        "classes": classes,
        "features": features[torch.from_numpy(positions)],
        "image_ids": test_ids,
        "labels": labels,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "cache_sha256": sha256(cache_path),
    }


def encode_text(context: dict, prompts: list[str], batch_size: int = 128) -> torch.Tensor:
    model, tokenizer = context["model"], context["tokenizer"]
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
        if int(inputs["input_ids"].shape[1]) > model.config.text_config.max_position_embeddings:
            raise RuntimeError("Prompt exceeds the frozen CLIP text limit")
        inputs = {key: value.to(context["device"]) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = model.text_model(**inputs)
            last = inputs["attention_mask"].sum(1) - 1
            pooled = hidden.last_hidden_state[
                torch.arange(len(batch), device=context["device"]), last
            ]
            outputs.append(F.normalize(model.text_projection(pooled), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def recomposed_prompt(record: dict, selected: list[str]) -> str:
    if not selected:
        return record["base_description"]
    return record["base_description"] + ". Diagnostic attributes: " + "; ".join(selected) + "."


def macro_accuracy(truth: np.ndarray, predicted: np.ndarray, n_classes: int = 100) -> float:
    per_class = []
    for class_id in range(n_classes):
        mask = truth == class_id
        per_class.append(float(np.mean(predicted[mask] == truth[mask])) if mask.any() else 0.0)
    return float(np.mean(per_class))
