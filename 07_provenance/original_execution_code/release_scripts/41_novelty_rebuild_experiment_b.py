# -*- coding: utf-8 -*-
"""Experiment B: scorer × evidence-composition comparison under A-clean protocol.

Ranking determines the selected set only. Selected attributes are always
serialized in frozen catalogue order for prompt recomposition. No parameter is
tuned: base-plus-vote uses lambda=0.5 as a fixed diagnostic baseline.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPTokenizer


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.text_config.max_position_embeddings)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        out = model.text_model(**inputs)
        eos = inputs["attention_mask"].sum(dim=1) - 1
        pooled = out.last_hidden_state[torch.arange(len(prompts), device=device), eos]
        return F.normalize(model.text_projection(pooled), dim=-1).detach().cpu()


def load_split(dataset, name, base_index, row_by_stem, class_ids):
    split = json.load(open(os.path.join(dataset, "splits", f"{name}.json"), encoding="utf-8"))
    stems = [os.path.splitext(os.path.basename(path))[0] for path in split["all_samples"]]
    return torch.tensor([base_index[stem] for stem in stems]), np.asarray([class_ids.index(row_by_stem[stem]["class_id"]) for stem in stems]), stems


def rankings(image, support_idx, support_labels, attribute_text, selector):
    all_rankings, audit = [], []
    for class_index, features in enumerate(attribute_text):
        own, other = image[support_idx][support_labels == class_index], image[support_idx][support_labels != class_index]
        rows = []
        for attr_index, feature in enumerate(features):
            positive, negative = float((own @ feature).mean()), float((other @ feature).mean())
            score = positive if selector == "positive_mean" else positive - negative
            rows.append((score, attr_index, positive, negative))
        rows.sort(key=lambda row: row[0], reverse=True)
        all_rankings.append([attr_index for _, attr_index, _, _ in rows])
        for rank, (score, attr_index, positive, negative) in enumerate(rows, 1):
            audit.append({"selector": selector, "class_index": class_index, "attribute_index": attr_index, "positive_mean": positive, "negative_mean": negative, "contrastive_margin": positive-negative, "ranking_score": score, "rank": rank})
    return all_rankings, audit


def canonical_selected(rankings, k):
    return [sorted(ranking[:k]) for ranking in rankings]


def compose_prompts(knowledge, attributes, selected):
    prompts = []
    for item, attrs, indices in zip(knowledge, attributes, selected):
        joined = ", ".join(attrs[index] for index in indices)
        prompts.append(f"{item['caption_en']}. Diagnostic attributes: {joined}.")
    return prompts


def evaluate_logits(labels, logits):
    prediction = logits.argmax(axis=1)
    return float(accuracy_score(labels, prediction)), float(f1_score(labels, prediction, average="macro")), prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pest_vlm_dataset")
    parser.add_argument("--base-features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base = np.load(args.base_features)
    image = torch.from_numpy(base["image_features"].astype(np.float32))
    base_index = {str(stem): i for i, stem in enumerate(base["stems"])}
    metadata = [json.loads(line) for line in open(os.path.join(args.dataset, "annotations", "metadata.jsonl"), encoding="utf-8") if line.strip()]
    row_by_stem = {row["stem"]: row for row in metadata}
    class_ids = sorted({row["class_id"] for row in metadata})
    catalogue_path = os.path.join(args.dataset, "annotations", "class_knowledge.json")
    knowledge = sorted(json.load(open(catalogue_path, encoding="utf-8"))["classes"], key=lambda row: row["class_id"])
    attributes = [row["key_points_en"] for row in knowledge]
    assert len(knowledge) == 10 and all(len(attrs) == 4 for attrs in attributes), "Experiment B requires the frozen 10×4 catalogue."
    support_idx, support_labels_np, _ = load_split(args.dataset, "train_pool", base_index, row_by_stem, class_ids)
    support_labels = torch.from_numpy(support_labels_np)
    splits = {name: load_split(args.dataset, name, base_index, row_by_stem, class_ids) for name in ("val", "test")}

    model = CLIPModel.from_pretrained(args.model, local_files_only=True).eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = CLIPTokenizer.from_pretrained(args.model, local_files_only=True)
    attr_prompts = [[f"a close-up of an insect showing {attribute}." for attribute in attrs] for attrs in attributes]
    attr_text = [encode(model, tokenizer, prompts) for prompts in attr_prompts]
    base_text = encode(model, tokenizer, [row["caption_en"] for row in knowledge])

    result_rows, selection_rows, hash_rows, prediction_rows, audit_rows = [], [], [], [], []
    for selector in ("positive_mean", "contrastive_margin"):
        ranked, audit = rankings(image, support_idx, support_labels, attr_text, selector)
        audit_rows.extend([{**row, "backbone": args.backbone} for row in audit])
        # k=0 is base-description-only classification.  It is included in the
        # same validation sweep so that all compositions expose the declared
        # 0--4 budget grid rather than silently excluding abstention.
        for k in range(0, 5):
            selected = canonical_selected(ranked, k)
            prompts = [item["caption_en"] for item in knowledge] if k == 0 else compose_prompts(knowledge, attributes, selected)
            recomposed_text = base_text if k == 0 else encode(model, tokenizer, prompts)
            for class_index, (item, attrs, indices, prompt) in enumerate(zip(knowledge, attributes, selected, prompts)):
                selected_text = [attrs[index] for index in indices]
                selection_rows.append({"backbone": args.backbone, "selector": selector, "k": k, "class_index": class_index, "class_name": item["name_en"], "selected_attribute_indices": ";".join(map(str, indices)), "selected_attributes": " | ".join(selected_text)})
                hash_rows.append({"backbone": args.backbone, "selector": selector, "composition": "recompose", "k": k, "class_index": class_index, "class_name": item["name_en"], "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "prompt_template_version": "canonical_catalogue_order_v1"})
                for attr_index in indices:
                    ap = attr_prompts[class_index][attr_index]
                    hash_rows.append({"backbone": args.backbone, "selector": selector, "composition": "attribute_vote", "k": k, "class_index": class_index, "class_name": item["name_en"], "prompt": ap, "prompt_sha256": hashlib.sha256(ap.encode("utf-8")).hexdigest(), "prompt_template_version": "attribute_prompt_v1"})
            for composition in ("recompose", "attribute_vote", "base_plus_vote"):
                for split_name, (indices, labels, stems) in splits.items():
                    if composition == "recompose" or k == 0:
                        logits = (image[indices] @ recomposed_text.T).detach().numpy()
                    else:
                        scores = []
                        for class_index, attribute_indices in enumerate(selected):
                            vote = (image[indices] @ attr_text[class_index][attribute_indices].T).mean(dim=1)
                            if composition == "base_plus_vote":
                                base_score = image[indices] @ base_text[class_index]
                                vote = .5 * base_score + .5 * vote
                            scores.append(vote.detach().numpy())
                        logits = np.stack(scores, axis=1)
                    accuracy, macro_f1, prediction = evaluate_logits(labels, logits)
                    result_rows.append({"backbone": args.backbone, "selector": selector, "composition": composition, "split": split_name, "k": k, "accuracy": accuracy, "macro_f1": macro_f1, "lambda_base": .5 if composition == "base_plus_vote" else np.nan})
                    prediction_rows.extend({"backbone": args.backbone, "selector": selector, "composition": composition, "split": split_name, "k": k, "stem": stem, "true_index": int(true), "predicted_index": int(pred)} for stem, true, pred in zip(stems, labels, prediction))

    results = pd.DataFrame(result_rows)
    validation_selected = []
    for (selector, composition), group in results[results.split == "val"].groupby(["selector", "composition"]):
        chosen = group.sort_values(["accuracy", "k"], ascending=[False, True]).iloc[0]
        test = results[(results.selector == selector) & (results.composition == composition) & (results.split == "test") & (results.k == chosen.k)].iloc[0]
        validation_selected.append({"backbone": args.backbone, "selector": selector, "composition": composition, "val_selected_k": int(chosen.k), "validation_accuracy": chosen.accuracy, "test_accuracy": test.accuracy, "test_macro_f1": test.macro_f1, "lambda_base": .5 if composition == "base_plus_vote" else np.nan})
    metadata_rows = [
        {"backbone": args.backbone, "catalogue_sha256": sha256(catalogue_path), "train_pool_sha256": sha256(os.path.join(args.dataset, "splits", "train_pool.json")), "val_sha256": sha256(os.path.join(args.dataset, "splits", "val.json")), "test_sha256": sha256(os.path.join(args.dataset, "splits", "test.json")), "crop_feature_cache": args.base_features, "crop_feature_cache_sha256": sha256(args.base_features), "clip_checkpoint": args.model, "attribute_prompt_template_version": "attribute_prompt_v1", "recomposition_template_version": "canonical_catalogue_order_v1", "eligible_budgets": "0,1,2,3,4", "k0_definition": "base_description_only", "base_plus_vote_lambda": .5}
    ]
    prefix = os.path.join(args.output_dir, f"experiment_b_{args.backbone}")
    results.to_csv(f"{prefix}_budget_curves.csv", index=False)
    pd.DataFrame(validation_selected).to_csv(f"{prefix}_validation_selected.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(f"{prefix}_selected_attributes.csv", index=False)
    pd.DataFrame(hash_rows).to_csv(f"{prefix}_prompt_hashes.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(f"{prefix}_ranking_audit.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(f"{prefix}_metadata.csv", index=False)
    print(pd.DataFrame(validation_selected).to_string(index=False))


if __name__ == "__main__":
    main()
