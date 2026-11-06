"""Shared cached-feature utilities for v13 Aircraft analyses."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from transformers import CLIPModel, CLIPTokenizer

def _locate_package_root() -> Path:
    configured = os.environ.get("VLM_ROOT")
    if configured:
        return Path(configured).resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "external_aircraft_clean").is_dir():
            return candidate
    raise RuntimeError("Could not locate v13 package root (external_aircraft_clean missing)")


ROOT = _locate_package_root()
WORKSPACE_ROOT = next(
    (candidate for candidate in (ROOT, *ROOT.parents)
     if (candidate / "external_datasets" / "fgvc-aircraft-2013b").is_dir()),
    ROOT,
)
BACKBONES = ("ViT-B-16", "ViT-B-32")
SCORERS = ("positive_mean", "contrastive_margin")
RUN = ROOT / "external_aircraft_clean"
RESOURCE = RUN / "resources/aircraft_descriptors_clean.json"
AUDIT = RUN / "resources/aircraft_descriptor_audit.csv"
DATASET = WORKSPACE_ROOT / "external_datasets/fgvc-aircraft-2013b"
PROTOCOL = RUN / "protocol"
MODEL_MAP = PROTOCOL / "model_fingerprints_release_mapping.json"
EXPECTED_RESOURCE_SHA = "67876cc858a9b6a1785b7eaa229a54dd40ad3394de7a7c989ad0d77b803808fc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def annotations(split: str) -> list[tuple[str, str]]:
    rows = []
    path = DATASET / "data" / f"images_variant_{split}.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        image_id, class_name = line.split(" ", 1)
        rows.append((image_id, class_name))
    return rows


def load_context(backbone: str, device: str = "cpu") -> dict:
    if backbone not in BACKBONES:
        raise ValueError(backbone)
    if sha256(RESOURCE) != EXPECTED_RESOURCE_SHA:
        raise RuntimeError("Aircraft resource hash mismatch")
    classes = json.loads(RESOURCE.read_text(encoding="utf-8"))["classes"]
    names = [r["name_en"] for r in classes]
    all_rows = [(i, n, s) for s in ("train", "val", "test") for i, n in annotations(s)]
    cache = np.load(RUN / "feature_cache" / f"image_features_{backbone}.npz")
    ids = cache["image_ids"].astype(str).tolist()
    expected_ids = [r[0] for r in all_rows]
    if ids != expected_ids:
        raise RuntimeError("Feature-cache image IDs are not aligned with official annotations")
    features = torch.from_numpy(cache["features"].astype(np.float32))
    features = F.normalize(features, dim=-1)
    labels = np.asarray([names.index(r[1]) for r in all_rows], dtype=np.int64)
    splits = np.asarray([r[2] for r in all_rows])
    positions = {image_id: i for i, image_id in enumerate(ids)}
    model_map = json.loads(MODEL_MAP.read_text(encoding="utf-8"))
    if "models" in model_map:
        model_map = model_map["models"]
    model_info = model_map[backbone]
    model_path = Path(model_info["path"])
    if not model_path.is_absolute():
        options = [WORKSPACE_ROOT / model_path, ROOT / model_path,
                   *(parent / model_path for parent in ROOT.parents)]
        model_path = next((path for path in options if path.is_dir()), options[0])
    if not model_path.is_dir():
        raise FileNotFoundError(f"Frozen local checkpoint not found: {model_path}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
    return dict(classes=classes, names=names, ids=ids, features=features, labels=labels,
                splits=splits, positions=positions, model=model, tokenizer=tokenizer,
                model_path=model_path, device=device)


def encode_text(context: dict, prompts: list[str], batch_size: int = 128) -> torch.Tensor:
    model, tokenizer = context["model"], context["tokenizer"]
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
        if int(inputs["input_ids"].shape[1]) > model.config.text_config.max_position_embeddings:
            raise RuntimeError("Prompt exceeds frozen CLIP token limit")
        inputs = {key: value.to(context["device"]) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = model.text_model(**inputs)
            last = inputs["attention_mask"].sum(1) - 1
            pooled = hidden.last_hidden_state[torch.arange(len(batch), device=context["device"]), last]
            outputs.append(F.normalize(model.text_projection(pooled), dim=-1).cpu())
    return torch.cat(outputs)


def descriptor_features(context: dict, anchored: bool = False) -> list[torch.Tensor]:
    out = []
    for record in context["classes"]:
        if anchored:
            prompts = [f"a {record['name_en']} with {d}" for d in record["descriptors"]]
        else:
            prompts = [f"a photo of an aircraft showing {d}." for d in record["descriptors"]]
        out.append(encode_text(context, prompts))
    return out


def rank_descriptors(context: dict, support_ids: list[str], scorer: str,
                     attributes: list[torch.Tensor]) -> list[list[int]]:
    pos = torch.tensor([context["positions"][x] for x in support_ids])
    image = context["features"][pos]
    labels = torch.from_numpy(context["labels"][pos.numpy()])
    orders = []
    for class_id, texts in enumerate(attributes):
        own, other = image[labels == class_id], image[labels != class_id]
        score = own @ texts.T
        values = score.mean(0)
        if scorer == "contrastive_margin":
            values = values - (other @ texts.T).mean(0)
        orders.append(sorted(range(4), key=lambda j: (-float(values[j]), j)))
    return orders


def build_recomposed_prompts(classes: list[dict], orders: list[list[int]], k: int) -> list[str]:
    prompts = []
    for record, order in zip(classes, orders):
        if k == 0:
            prompts.append(record["base_description"])
        else:
            chosen = set(order[:k])
            canonical = [d for j, d in enumerate(record["descriptors"]) if j in chosen]
            prompts.append(record["base_description"] + ". Diagnostic attributes: " + "; ".join(canonical) + ".")
    return prompts


def predict(features: torch.Tensor, text: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits = (features @ text.T).numpy()
    return logits.argmax(1), logits


def macro_accuracy(y: np.ndarray, p: np.ndarray, n_classes: int = 100) -> float:
    return float(recall_score(y, p, labels=np.arange(n_classes), average="macro", zero_division=0))


def official_ids(split: str) -> list[str]:
    return [i for i, _ in annotations(split)]


def manifest_support(seed: int) -> list[str]:
    m = json.loads((PROTOCOL / f"split_seed_{seed}.json").read_text(encoding="utf-8"))
    audit = json.loads((PROTOCOL / "dataset_audit.json").read_text(encoding="utf-8"))
    names = audit["classes_lexicographic"]
    return [i for name in names for i in m["support"][name]]


def save_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
