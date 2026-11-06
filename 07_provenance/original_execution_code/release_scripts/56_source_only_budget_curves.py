# -*- coding: utf-8 -*-
"""Image-blind source-only descriptor-resource budget curves.

This provenance sensitivity experiment loads a newly fixed resource constructed
from class identity and cited public source documents.  It never opens a current
benchmark image while constructing text, and it uses only cached image features
for the downstream, otherwise canonical, frozen-CLIP protocol.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPTokenizer


def encode(model, tokenizer, prompts):
    max_len = int(model.config.text_config.max_position_embeddings)
    raw = tokenizer(prompts, add_special_tokens=True, truncation=False)
    pre_lengths = [len(ids) for ids in raw["input_ids"]]
    if max(pre_lengths) > max_len:
        raise RuntimeError(f"Prompt would exceed CLIP's {max_len}-position limit: {max(pre_lengths)}")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
    executed_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.text_model(**encoded)
        last = encoded["attention_mask"].sum(dim=1) - 1
        pooled = output.last_hidden_state[torch.arange(len(prompts), device=device), last]
    return F.normalize(model.text_projection(pooled), dim=-1).detach().cpu(), pre_lengths, executed_lengths, max_len


def split_data(dataset, split_name, stem_to_index, metadata, class_ids):
    payload = json.loads((dataset / "splits" / f"{split_name}.json").read_text(encoding="utf-8"))
    stems = [Path(path).stem for path in payload["all_samples"]]
    index = torch.tensor([stem_to_index[stem] for stem in stems])
    labels = np.asarray([class_ids.index(metadata[stem]["class_id"]) for stem in stems])
    return stems, index, labels


def rank(image, support_index, support_labels, attr_features, selector):
    all_rankings, rows = [], []
    for class_index, features in enumerate(attr_features):
        own = image[support_index][support_labels == class_index]
        other = image[support_index][support_labels != class_index]
        scored = []
        for attribute_index, feature in enumerate(features):
            positive = float((own @ feature).mean())
            negative = float((other @ feature).mean())
            value = positive if selector == "positive_mean" else positive - negative
            scored.append((value, attribute_index, positive, negative))
        scored.sort(key=lambda row: (-row[0], row[1]))
        all_rankings.append([attribute_index for _, attribute_index, _, _ in scored])
        for position, (value, attribute_index, positive, negative) in enumerate(scored, start=1):
            rows.append({"selector": selector, "class_index": class_index, "attribute_index": attribute_index,
                         "rank": position, "ranking_score": value, "positive_mean": positive,
                         "negative_mean": negative, "contrastive_margin": positive - negative})
    return all_rankings, rows


def prompts_for_budget(resource, rankings, k):
    prompts, selected = [], []
    for item, ordering in zip(resource, rankings):
        indices = sorted(ordering[:k])
        selected.append(indices)
        if not indices:
            prompts.append(item["base_description"])
        else:
            descriptors = ", ".join(item["descriptors"][index] for index in indices)
            prompts.append(f"{item['base_description']}. Diagnostic attributes: {descriptors}.")
    return prompts, selected


def score(image, indices, labels, text):
    prediction = (image[indices] @ text.T).numpy().argmax(axis=1)
    return prediction, float(accuracy_score(labels, prediction)), float(f1_score(labels, prediction, average="macro"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pest_vlm_dataset")
    parser.add_argument("--resource", default="experiments/resources/source_only_descriptor_resource_v1.json")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset, out = Path(args.dataset), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resource_payload = json.loads(Path(args.resource).read_text(encoding="utf-8"))
    resource = sorted(resource_payload["classes"], key=lambda x: x["class_id"])
    if len(resource) != 10 or any(len(item["descriptors"]) != 4 for item in resource):
        raise ValueError("Source-only resource must contain ten classes and four descriptors per class.")

    cache = np.load(args.features)
    image = torch.from_numpy(cache["image_features"].astype(np.float32))
    stem_to_index = {str(stem): index for index, stem in enumerate(cache["stems"])}
    metadata_rows = [json.loads(line) for line in (dataset / "annotations" / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metadata = {row["stem"]: row for row in metadata_rows}
    class_ids = sorted({row["class_id"] for row in metadata_rows})
    split = {name: split_data(dataset, name, stem_to_index, metadata, class_ids) for name in ("train_pool", "val", "test")}

    model = CLIPModel.from_pretrained(args.model, local_files_only=True).eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = CLIPTokenizer.from_pretrained(args.model, local_files_only=True)
    attribute_prompts = [[resource_payload["scoring_template"].format(attribute=attr) for attr in item["descriptors"]] for item in resource]
    attr_features, token_rows = [], []
    for class_index, prompts in enumerate(attribute_prompts):
        encoded, pre_lengths, executed_lengths, max_len = encode(model, tokenizer, prompts)
        attr_features.append(encoded)
        for attribute_index, (prompt, pre_length, executed_length) in enumerate(zip(prompts, pre_lengths, executed_lengths)):
            token_rows.append({"backbone": args.backbone, "prompt_type": "attribute_scoring", "class_index": class_index,
                               "attribute_index": attribute_index, "k": None, "prompt": prompt,
                               "pre_truncation_input_length_including_special_tokens": pre_length,
                               "executed_input_length_including_special_tokens": int(executed_length),
                               "max_position_embeddings": max_len,
                               "execution_truncation_setting": False,
                               "truncated": False})

    support_stems, support_idx, support_labels_np = split["train_pool"]
    support_labels = torch.from_numpy(support_labels_np)
    curve_rows, ranking_rows, selected_rows, prediction_rows = [], [], [], []
    for selector in ("positive_mean", "contrastive_margin"):
        rankings, audit = rank(image, support_idx, support_labels, attr_features, selector)
        ranking_rows.extend([{**row, "backbone": args.backbone} for row in audit])
        for k in range(5):
            prompts, selected = prompts_for_budget(resource, rankings, k)
            text, pre_lengths, executed_lengths, max_len = encode(model, tokenizer, prompts)
            for class_index, (item, prompt, indices, pre_length, executed_length) in enumerate(zip(resource, prompts, selected, pre_lengths, executed_lengths)):
                token_rows.append({"backbone": args.backbone, "prompt_type": "recomposed", "class_index": class_index,
                                   "attribute_index": None, "k": k, "prompt": prompt,
                                   "pre_truncation_input_length_including_special_tokens": pre_length,
                                   "executed_input_length_including_special_tokens": int(executed_length),
                                   "max_position_embeddings": max_len,
                                   "execution_truncation_setting": False,
                                   "truncated": False})
                selected_rows.append({"backbone": args.backbone, "selector": selector, "k": k, "class_index": class_index,
                                      "class_name": item["name_en"], "selected_descriptor_indices": ";".join(map(str, indices)),
                                      "selected_descriptors": " | ".join(item["descriptors"][i] for i in indices),
                                      "final_prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
            for split_name in ("val", "test"):
                stems, indices, labels = split[split_name]
                predicted, accuracy, macro_f1 = score(image, indices, labels, text)
                curve_rows.append({"backbone": args.backbone, "selector": selector, "split": split_name, "k": k,
                                   "accuracy": accuracy, "macro_f1": macro_f1, "n_images": len(labels)})
                for stem, true_index, predicted_index in zip(stems, labels, predicted):
                    prediction_rows.append({"backbone": args.backbone, "selector": selector, "split": split_name, "k": k,
                                            "stem": stem, "true_index": int(true_index), "predicted_index": int(predicted_index),
                                            "correct": bool(true_index == predicted_index)})

    curves = pd.DataFrame(curve_rows)
    selections = []
    for (backbone, selector), group in curves[curves.split.eq("val")].groupby(["backbone", "selector"], sort=True):
        chosen = group.sort_values(["accuracy", "k"], ascending=[False, True]).iloc[0]
        test = curves[(curves.backbone == backbone) & (curves.selector == selector) & curves.split.eq("test") & (curves.k == chosen.k)].iloc[0]
        selections.append({"backbone": backbone, "selector": selector, "selected_k": int(chosen.k),
                           "validation_accuracy": chosen.accuracy, "fixed_evaluation_accuracy": test.accuracy,
                           "tie_rule": "smaller_k_on_equal_validation_accuracy"})
    pd.DataFrame(curves).to_csv(out / f"source_only_{args.backbone}_budget_curves.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(out / f"source_only_{args.backbone}_ranking_audit.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / f"source_only_{args.backbone}_selected_descriptors.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / f"source_only_{args.backbone}_per_image_predictions.csv", index=False)
    pd.DataFrame(token_rows).to_csv(out / f"source_only_{args.backbone}_prompt_token_audit.csv", index=False)
    pd.DataFrame(selections).to_csv(out / f"source_only_{args.backbone}_validation_selection.csv", index=False)
    metadata_out = {"resource_id": resource_payload["resource_id"], "construction_scope": resource_payload["construction_scope"],
                    "feature_cache": str(Path(args.features)), "model": args.model, "backbone": args.backbone,
                    "n_support": len(support_stems), "n_validation": len(split["val"][0]), "n_fixed_evaluation": len(split["test"][0]),
                    "token_limit": int(model.config.text_config.max_position_embeddings)}
    (out / f"source_only_{args.backbone}_run_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    print(pd.DataFrame(selections).to_string(index=False))


if __name__ == "__main__":
    main()
