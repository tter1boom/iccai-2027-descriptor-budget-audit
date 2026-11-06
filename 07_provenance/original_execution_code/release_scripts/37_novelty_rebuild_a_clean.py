# -*- coding: utf-8 -*-
"""A-clean: canonical-order matched-pool evidence-budget comparison.

Ranking controls set membership only. Every selected set is serialized in the
fixed catalogue-index order, eliminating attribute-ordering as a prompt
composition confound. Random uses 50 class-wise permutation draws for k=1..3;
k=0 and k=4 are deterministic shared prompts by construction.
"""

import argparse
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPTokenizer


def encode(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.text_config.max_position_embeddings)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        out = model.text_model(**inputs)
        last = inputs["attention_mask"].sum(dim=1) - 1
        pooled = out.last_hidden_state[torch.arange(len(prompts), device=device), last]
    return F.normalize(model.text_projection(pooled), dim=-1).detach().cpu()


def load_split(dataset, split_name, base_index, row_by_stem, class_ids):
    split = json.load(open(os.path.join(dataset, "splits", f"{split_name}.json"), encoding="utf-8"))
    stems = [os.path.splitext(os.path.basename(path))[0] for path in split["all_samples"]]
    indices = torch.tensor([base_index[stem] for stem in stems])
    labels = np.asarray([class_ids.index(row_by_stem[stem]["class_id"]) for stem in stems])
    return indices, labels


def deterministic_rankings(image, support_idx, support_labels, attr_features, scorer):
    rankings, audit = [], []
    for class_index, features in enumerate(attr_features):
        own = image[support_idx][support_labels == class_index]
        other = image[support_idx][support_labels != class_index]
        rows = []
        for attr_index, feature in enumerate(features):
            positive = float((own @ feature).mean())
            negative = float((other @ feature).mean())
            score = positive if scorer == "positive_mean" else positive - negative
            rows.append((score, attr_index, positive, negative))
        rows.sort(key=lambda row: row[0], reverse=True)
        rankings.append([attr_index for _, attr_index, _, _ in rows])
        for rank, (score, attr_index, positive, negative) in enumerate(rows, start=1):
            audit.append({"selector": scorer, "class_index": class_index, "attribute_index": attr_index, "positive_mean": positive, "negative_mean": negative, "contrastive_margin": positive-negative, "ranking_score": score, "rank": rank})
    return rankings, audit


def random_rankings(class_count, attribute_count, draw, seed):
    generator = random.Random(seed + draw)
    rankings = []
    for _ in range(class_count):
        order = list(range(attribute_count))
        generator.shuffle(order)
        rankings.append(order)
    return rankings


def canonical_prompts(knowledge, attributes, rankings, k):
    prompts, selected = [], []
    for item, attrs, ranking in zip(knowledge, attributes, rankings):
        indices = sorted(ranking[:k])
        selected.append(indices)
        if k == 0:
            prompts.append(item["caption_en"])
        else:
            joined = ", ".join(attrs[index] for index in indices)
            prompts.append(f"{item['caption_en']}. Diagnostic attributes: {joined}.")
    return prompts, selected


def metrics(image, indices, labels, text):
    prediction = (image[indices] @ text.T).detach().numpy().argmax(axis=1)
    return float(accuracy_score(labels, prediction)), float(f1_score(labels, prediction, average="macro"))


def prompt_rows(selector, backbone, draw, k, knowledge, attributes, prompts, selected):
    rows = []
    for class_index, (item, attrs, prompt, indices) in enumerate(zip(knowledge, attributes, prompts, selected)):
        rows.append({
            "selector": selector, "backbone": backbone, "random_draw": draw, "k": k,
            "class_index": class_index, "class_name": item["name_en"],
            "selected_attribute_indices": ";".join(map(str, indices)),
            "selected_attributes": " | ".join(attrs[index] for index in indices),
            "final_prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
    return rows


def aggregate(rows):
    frame = pd.DataFrame(rows)
    output = []
    for (selector, backbone, split, k), group in frame.groupby(["selector", "backbone", "split", "k"], sort=True):
        accuracy = group.accuracy.to_numpy()
        f1 = group.macro_f1.to_numpy()
        output.append({
            "selector": selector, "backbone": backbone, "split": split, "k": k, "n_realisations": len(group),
            "accuracy_mean": accuracy.mean(), "accuracy_sd": accuracy.std(ddof=1) if len(accuracy) > 1 else 0.0,
            "accuracy_q025": np.quantile(accuracy, .025), "accuracy_q975": np.quantile(accuracy, .975),
            "macro_f1_mean": f1.mean(), "macro_f1_sd": f1.std(ddof=1) if len(f1) > 1 else 0.0,
        })
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pest_vlm_dataset")
    parser.add_argument("--base-features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-draws", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=4200)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base = np.load(args.base_features)
    image = torch.from_numpy(base["image_features"].astype(np.float32))
    base_index = {str(stem): i for i, stem in enumerate(base["stems"])}
    metadata = [json.loads(line) for line in open(os.path.join(args.dataset, "annotations", "metadata.jsonl"), encoding="utf-8") if line.strip()]
    row_by_stem = {row["stem"]: row for row in metadata}
    class_ids = sorted({row["class_id"] for row in metadata})
    knowledge = sorted(json.load(open(os.path.join(args.dataset, "annotations", "class_knowledge.json"), encoding="utf-8"))["classes"], key=lambda row: row["class_id"])
    attributes = [row["key_points_en"] for row in knowledge]
    assert len(knowledge) == 10 and all(len(attrs) == 4 for attrs in attributes), "A-clean requires the frozen 10×4 catalogue."
    support_idx, support_labels_np = load_split(args.dataset, "train_pool", base_index, row_by_stem, class_ids)
    support_labels = torch.from_numpy(support_labels_np)
    split_data = {name: load_split(args.dataset, name, base_index, row_by_stem, class_ids) for name in ("val", "test")}

    model = CLIPModel.from_pretrained(args.model, local_files_only=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model, local_files_only=True)
    attr_features = [encode(model, tokenizer, [f"a close-up of an insect showing {attribute}." for attribute in attrs]) for attrs in attributes]

    audit_rows, result_rows, prompt_audit = [], [], []
    all_prompts = {}
    deterministic = {}
    for selector in ("positive_mean", "contrastive_margin"):
        rankings, audit = deterministic_rankings(image, support_idx, support_labels, attr_features, selector)
        audit_rows.extend([{**row, "backbone": args.backbone} for row in audit])
        deterministic[selector] = rankings

    jobs = []
    for selector, rankings in deterministic.items():
        for k in range(5):
            jobs.append((selector, 0, rankings, k))
    for k in range(5):
        draws = [0] if k in (0, 4) else list(range(args.random_draws))
        for draw in draws:
            jobs.append(("random", draw, random_rankings(len(knowledge), 4, draw, args.random_seed), k))

    for selector, draw, rankings, k in jobs:
        prompts, selected = canonical_prompts(knowledge, attributes, rankings, k)
        hashes = tuple(hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts)
        all_prompts[(selector, draw, k)] = hashes
        prompt_audit.extend(prompt_rows(selector, args.backbone, draw, k, knowledge, attributes, prompts, selected))
        text = encode(model, tokenizer, prompts)
        for split, (indices, labels) in split_data.items():
            accuracy, macro_f1 = metrics(image, indices, labels, text)
            result_rows.append({"selector": selector, "backbone": args.backbone, "random_draw": draw, "split": split, "k": k, "accuracy": accuracy, "macro_f1": macro_f1})

    # Strict prompt invariance at k=0 and k=4. Random k=0/k=4 are deterministic jobs.
    for k in (0, 4):
        expected = all_prompts[("positive_mean", 0, k)]
        assert all_prompts[("contrastive_margin", 0, k)] == expected, f"Selector prompts differ at k={k}."
        assert all_prompts[("random", 0, k)] == expected, f"Random prompt differs at k={k}."
    results = pd.DataFrame(result_rows)
    grouped = aggregate(result_rows)
    # Results must also be invariant when prompts are invariant.
    for k in (0, 4):
        for split in ("val", "test"):
            values = results[(results.k == k) & (results.split == split)].accuracy.unique()
            assert len(values) == 1, f"Metrics differ despite identical prompts at k={k}, split={split}."

    results.to_csv(os.path.join(args.output_dir, f"a_clean_{args.backbone}_all_realisations.csv"), index=False)
    grouped.to_csv(os.path.join(args.output_dir, f"a_clean_{args.backbone}_aggregated_curves.csv"), index=False)
    grouped[grouped.split == "val"].to_csv(os.path.join(args.output_dir, f"a_clean_{args.backbone}_validation_curve.csv"), index=False)
    grouped[grouped.split == "test"].to_csv(os.path.join(args.output_dir, f"a_clean_{args.backbone}_test_curve.csv"), index=False)
    pd.DataFrame(audit_rows).to_csv(os.path.join(args.output_dir, f"a_clean_{args.backbone}_ranking_audit.csv"), index=False)
    pd.DataFrame(prompt_audit).to_csv(os.path.join(args.output_dir, f"a_clean_{args.backbone}_selected_attributes_and_prompts.csv"), index=False)
    sanity = pd.DataFrame([
        {"backbone": args.backbone, "check": "k0_prompt_identity_across_selectors", "status": "PASS"},
        {"backbone": args.backbone, "check": "k4_prompt_identity_across_selectors", "status": "PASS"},
        {"backbone": args.backbone, "check": "k0_metric_identity_across_selectors", "status": "PASS"},
        {"backbone": args.backbone, "check": "k4_metric_identity_across_selectors", "status": "PASS"},
    ])
    sanity.to_csv(os.path.join(args.output_dir, f"a_clean_{args.backbone}_sanity_checks.csv"), index=False)
    print(grouped[grouped.split == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
