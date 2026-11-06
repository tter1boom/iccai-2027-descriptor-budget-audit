# -*- coding: utf-8 -*-
"""Experiment C: seed-wise BIOSCAN validation-calibrated evidence budgets.

Diagnostic protocol (defined for this analysis):
* fixed query set: 15 images/class;
* each seed partitions the original 15 support images/class into 10 ranking
  supports and 5 validation images;
* selectors: positive mean (primary) and contrastive margin (secondary);
* aggregation: canonical-order prompt recomposition only;
* eligible budgets: k=0,1,2,3,4; validation ties select the smaller k.

The script never consults query labels while ranking or selecting k.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPTokenizer
from transformers.utils.hub import cached_file


ATTRIBUTE_PROMPT_VERSION = "bioscan_attribute_prompt_v1"
RECOMPOSITION_TEMPLATE_VERSION = "bioscan_canonical_catalogue_order_v1"
ATTRIBUTE_PROMPT = "a close-up specimen image showing {attribute}."


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_model_hash(model_id):
    """Hash the actual config and weight file used by transformers."""
    parts = []
    for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
        try:
            path = cached_file(model_id, name, local_files_only=True)
        except (EnvironmentError, OSError):
            path = None
        if path and os.path.isfile(path):
            parts.append((name, sha256_file(path)))
    if not parts:
        candidate = Path(model_id)
        for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
            path = candidate / name
            if path.is_file():
                parts.append((name, sha256_file(path)))
    if not parts:
        raise FileNotFoundError(f"Could not resolve a local config/weight file for {model_id}.")
    combined = "\n".join(f"{name}:{digest}" for name, digest in sorted(parts))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest(), combined


def token_audit(tokenizer, prompts, limit, strict):
    """Return exact CLIP token lengths before padding; optionally prohibit overflow."""
    lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=True, padding=False, truncation=False)["input_ids"]]
    if strict and max(lengths, default=0) > limit:
        raise ValueError(f"Prompt exceeds the {limit}-position limit: {[(i, n) for i, n in enumerate(lengths) if n > limit][:5]}")
    return lengths


def encode(model, tokenizer, prompts, strict_tokens=False):
    limit = model.config.text_config.max_position_embeddings
    token_audit(tokenizer, prompts, limit, strict_tokens)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=not strict_tokens,
        max_length=limit,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.text_model(**inputs)
        eos = inputs["attention_mask"].sum(dim=1) - 1
        pooled = output.last_hidden_state[torch.arange(len(prompts), device=device), eos]
        return F.normalize(model.text_projection(pooled), dim=-1).detach().cpu()


def rank_attributes(image, support_idx, support_labels, attribute_text, selector):
    rankings, audit_rows = [], []
    support_image = image[support_idx]
    for class_index, features in enumerate(attribute_text):
        own = support_image[support_labels == class_index]
        other = support_image[support_labels != class_index]
        scored = []
        for attribute_index, feature in enumerate(features):
            positive = float((own @ feature).mean())
            negative = float((other @ feature).mean())
            score = positive if selector == "positive_mean" else positive - negative
            scored.append((score, attribute_index, positive, negative))
        scored.sort(key=lambda row: (-row[0], row[1]))
        rankings.append([attribute_index for _, attribute_index, _, _ in scored])
        for rank, (score, attribute_index, positive, negative) in enumerate(scored, 1):
            audit_rows.append({
                "selector": selector,
                "class_index": class_index,
                "attribute_index": attribute_index,
                "positive_mean": positive,
                "negative_mean": negative,
                "contrastive_margin": positive - negative,
                "ranking_score": score,
                "rank": rank,
            })
    return rankings, audit_rows


def canonical_selection(ranked, k):
    return [sorted(order[:k]) for order in ranked]


def compose_prompts(knowledge, attributes, selected):
    prompts = []
    for row, candidate_attributes, indices in zip(knowledge, attributes, selected):
        if not indices:
            prompts.append(row["caption_en"])
        else:
            joined = ", ".join(candidate_attributes[index] for index in indices)
            prompts.append(f"{row['caption_en']}. Diagnostic attributes: {joined}.")
    return prompts


def metrics(labels, logits):
    prediction = logits.argmax(axis=1)
    return (
        float(accuracy_score(labels, prediction)),
        float(f1_score(labels, prediction, average="macro")),
        prediction,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="external_datasets/BIOSCAN_subset")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--catalogue", default=None, help="Optional descriptor catalogue; defaults to dataset/class_knowledge.json.")
    parser.add_argument("--strict-tokens", action="store_true", help="Fail before encoding if a prompt exceeds CLIP's position limit.")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.dataset) / "metadata.jsonl"
    catalogue_path = Path(args.catalogue) if args.catalogue else Path(args.dataset) / "class_knowledge.json"
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    knowledge = sorted(json.loads(catalogue_path.read_text(encoding="utf-8"))["classes"], key=lambda row: row["class_id"])
    attributes = [row["key_points_en"] for row in knowledge]
    assert len(knowledge) == 10 and all(len(values) >= 4 for values in attributes)

    cache = np.load(args.features)
    stems = [str(stem) for stem in cache["stems"]]
    labels_np = cache["labels"].astype(np.int64)
    image = torch.from_numpy(cache["image_features"].astype(np.float32))
    metadata_by_stem = {str(row["stem"]): row for row in metadata}
    assert set(stems) == set(metadata_by_stem), "Feature cache and metadata use different image IDs."
    assert all(int(metadata_by_stem[stem]["class_id"]) == int(label) for stem, label in zip(stems, labels_np))
    index_by_stem = {stem: index for index, stem in enumerate(stems)}

    original_support = {class_index: [] for class_index in range(len(knowledge))}
    query_by_class = {class_index: [] for class_index in range(len(knowledge))}
    for stem, row in metadata_by_stem.items():
        target = original_support if row["split"] == "support" else query_by_class
        target[int(row["class_id"])].append(stem)
    assert all(len(values) == 15 for values in original_support.values())
    assert all(len(values) == 15 for values in query_by_class.values())
    query_stems = [stem for class_index in range(len(knowledge)) for stem in sorted(query_by_class[class_index])]
    query_idx = torch.tensor([index_by_stem[stem] for stem in query_stems])
    query_labels = labels_np[query_idx.numpy()]

    model = CLIPModel.from_pretrained(args.model, local_files_only=True).eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = CLIPTokenizer.from_pretrained(args.model, local_files_only=True)
    attribute_text = [encode(model, tokenizer, [ATTRIBUTE_PROMPT.format(attribute=value) for value in values], args.strict_tokens) for values in attributes]

    curve_rows, selection_rows, assignment_rows, prediction_rows, audit_rows, hash_rows, token_rows = [], [], [], [], [], [], []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        support_stems, validation_stems = [], []
        for class_index in range(len(knowledge)):
            values = sorted(original_support[class_index])
            rng = random.Random(seed * 1009 + class_index)
            selected_support = set(rng.sample(values, 10))
            support_stems.extend(sorted(selected_support))
            validation_stems.extend(sorted(set(values) - selected_support))
        assert len(support_stems) == 100 and len(validation_stems) == 50
        support_idx = torch.tensor([index_by_stem[stem] for stem in support_stems])
        support_labels = torch.tensor(labels_np[support_idx.numpy()])
        validation_idx = torch.tensor([index_by_stem[stem] for stem in validation_stems])
        validation_labels = labels_np[validation_idx.numpy()]
        for split_name, split_stems in (("support", support_stems), ("validation", validation_stems), ("query", query_stems)):
            for stem in split_stems:
                assignment_rows.append({"seed": seed, "split": split_name, "stem": stem, "class_index": int(metadata_by_stem[stem]["class_id"]), "species": metadata_by_stem[stem]["species"]})

        for selector in ("positive_mean", "contrastive_margin"):
            ranked, audit = rank_attributes(image, support_idx, support_labels, attribute_text, selector)
            audit_rows.extend([{**row, "seed": seed, "backbone": args.backbone} for row in audit])
            for k in range(5):
                selected = canonical_selection(ranked, k)
                prompts = compose_prompts(knowledge, attributes, selected)
                exact_lengths = token_audit(tokenizer, prompts, model.config.text_config.max_position_embeddings, args.strict_tokens)
                token_rows.extend({"seed": seed, "backbone": args.backbone, "selector": selector, "k": k,
                                   "class_index": class_index, "prompt": prompt, "token_length_before_padding": length,
                                   "position_limit": model.config.text_config.max_position_embeddings,
                                   "within_limit": length <= model.config.text_config.max_position_embeddings,
                                   "strict_untruncated": bool(args.strict_tokens)}
                                  for class_index, (prompt, length) in enumerate(zip(prompts, exact_lengths)))
                text = encode(model, tokenizer, prompts, args.strict_tokens)
                for class_index, (row, values, indices, prompt) in enumerate(zip(knowledge, attributes, selected, prompts)):
                    selection_rows.append({
                        "seed": seed, "backbone": args.backbone, "selector": selector, "k": k,
                        "class_index": class_index, "class_name": row["name_en"],
                        "selected_attribute_indices": ";".join(map(str, indices)),
                        "selected_attributes": " | ".join(values[index] for index in indices),
                    })
                    hash_rows.append({
                        "seed": seed, "backbone": args.backbone, "selector": selector, "k": k,
                        "class_index": class_index, "class_name": row["name_en"], "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "prompt_template_version": RECOMPOSITION_TEMPLATE_VERSION,
                    })
                for split_name, indices, truth, split_stems in (
                    ("validation", validation_idx, validation_labels, validation_stems),
                    ("query", query_idx, query_labels, query_stems),
                ):
                    logits = (image[indices] @ text.T).detach().numpy()
                    accuracy, macro_f1, prediction = metrics(truth, logits)
                    curve_rows.append({"seed": seed, "backbone": args.backbone, "selector": selector, "split": split_name, "k": k, "accuracy": accuracy, "macro_f1": macro_f1})
                    prediction_rows.extend({"seed": seed, "backbone": args.backbone, "selector": selector, "split": split_name, "k": k, "stem": stem, "true_index": int(true), "predicted_index": int(pred)} for stem, true, pred in zip(split_stems, truth, prediction))

    curves = pd.DataFrame(curve_rows)
    selected_rows = []
    for (seed, selector), group in curves[curves["split"] == "validation"].groupby(["seed", "selector"]):
        chosen = group.sort_values(["accuracy", "k"], ascending=[False, True]).iloc[0]
        query = curves[(curves.seed == seed) & (curves.selector == selector) & (curves["split"] == "query") & (curves.k == chosen.k)].iloc[0]
        selected_rows.append({"seed": int(seed), "backbone": args.backbone, "selector": selector, "selected_k": int(chosen.k), "validation_accuracy": chosen.accuracy, "query_accuracy": query.accuracy, "query_macro_f1": query.macro_f1})
    selected_frame = pd.DataFrame(selected_rows)
    aggregate = (selected_frame.groupby(["backbone", "selector", "selected_k"], as_index=False)
                 .agg(seed_count=("seed", "count"), mean_validation_accuracy=("validation_accuracy", "mean"), mean_query_accuracy=("query_accuracy", "mean"), std_query_accuracy=("query_accuracy", "std"), mean_query_macro_f1=("query_macro_f1", "mean"), std_query_macro_f1=("query_macro_f1", "std")))
    model_hash, model_hash_components = resolved_model_hash(args.model)
    metadata_frame = pd.DataFrame([{
        "backbone": args.backbone, "dataset_metadata_sha256": sha256_file(metadata_path),
        "catalogue_sha256": sha256_file(catalogue_path), "feature_cache": args.features,
        "feature_cache_sha256": sha256_file(args.features), "clip_checkpoint": args.model,
        "clip_checkpoint_sha256": model_hash, "clip_checkpoint_hash_components": model_hash_components,
        "attribute_prompt_template": ATTRIBUTE_PROMPT, "attribute_prompt_template_version": ATTRIBUTE_PROMPT_VERSION,
        "recomposition_template_version": RECOMPOSITION_TEMPLATE_VERSION,
        "seeds": args.seeds, "support_per_class": 10, "validation_per_class": 5, "query_per_class": 15,
        "eligible_budgets": "0,1,2,3,4", "validation_tie_break": "smallest_k",
        "token_handling": "strict_untruncated" if args.strict_tokens else "historical_truncation_enabled",
    }])
    prefix = out / f"experiment_c_{args.backbone}"
    curves.to_csv(f"{prefix}_curves.csv", index=False)
    selected_frame.to_csv(f"{prefix}_selected_k.csv", index=False)
    aggregate.to_csv(f"{prefix}_selected_k_aggregate.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(f"{prefix}_selected_attributes.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(f"{prefix}_split_assignments.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(f"{prefix}_ranking_audit.csv", index=False)
    pd.DataFrame(hash_rows).to_csv(f"{prefix}_prompt_hashes.csv", index=False)
    pd.DataFrame(token_rows).to_csv(f"{prefix}_token_audit.csv", index=False)
    metadata_frame.to_csv(f"{prefix}_metadata.csv", index=False)
    print(selected_frame.to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
