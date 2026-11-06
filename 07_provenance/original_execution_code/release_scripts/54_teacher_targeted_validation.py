"""Teacher-directed validation controls for frozen-VLM evidence budgeting.

This script deliberately keeps the original test split fixed.  It records four
separate questions that were not answered by the original budget curves:

1. Are any emitted CLIP prompts truncated at the 77-position text limit?
2. Does the response persist after removing behavioural/contextual descriptors
   or after removing descriptor facts explicitly repeated by the base text?
3. Does support-conditioned ranking exceed *random ranking with the same
   validation budget-selection rule*, rather than only random top-1 prompts?
4. How often does a group-bootstrap of support and validation choose each k?

The group bootstrap uses established near-duplicate connected components.  It
is a dependence-robustness analysis, not a specimen-level resampling analysis.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from transformers import CLIPModel, CLIPTokenizer


RESOURCE_VERSION = "teacher_control_resource_v1"
SELECTORS = ("positive_mean", "contrastive_margin")
BUDGETS = tuple(range(5))

# These decisions are frozen before execution.  "morphology_only" removes
# descriptors that require time, behaviour or external substrate/context.
# "deduplicated" removes a descriptor only where its fact is already explicit
# in the original base description; a related but non-identical fact remains.
CONTROL_RULES = {
    0: {
        "morphology_keep": (0, 1, 2), "dedup_keep": (3,),
        "notes": {3: "Behavioural movement descriptor; excluded only from morphology-only resource."},
    },
    1: {
        "morphology_keep": (0, 1, 2), "dedup_keep": (2, 3),
        "notes": {3: "Behavioural movement descriptor; excluded only from morphology-only resource."},
    },
    2: {
        "morphology_keep": (0, 1, 2, 3), "dedup_keep": (3,),
        "notes": {0: "Explicitly repeats cylindrical brown body in base.", 1: "Explicitly repeats antennal club in base.", 2: "Explicitly repeats striated/pitted elytra in base."},
    },
    3: {
        "morphology_keep": (0, 1, 2, 3), "dedup_keep": (3,),
        "notes": {0: "Explicitly repeats oval/humped/red-brown body in base.", 1: "Explicitly repeats serrated antennae in base.", 2: "Explicitly repeats smooth wing covers in base."},
    },
    4: {
        "morphology_keep": (0, 1, 2, 3), "dedup_keep": (2, 3),
        "notes": {0: "Explicitly repeats glossy red-brown globular abdomen in base.", 1: "Explicitly repeats long spider-like legs in base."},
    },
    5: {
        "morphology_keep": (0, 1, 3), "dedup_keep": (2, 3),
        "notes": {0: "Explicitly repeats wingless, six-legged, short-antenna form in base.", 1: "Explicitly repeats forked jumping organ in base.", 2: "Behavioural descriptor; excluded only from morphology-only resource."},
    },
    6: {
        "morphology_keep": (0, 1, 2, 3), "dedup_keep": (3,),
        "notes": {0: "Explicitly repeats large head in base.", 1: "Explicitly repeats clear membranous wings in base.", 2: "Explicitly repeats roof-like resting wings in base.", 3: "Residual is the non-repeated phrase 'no cerci'."},
    },
    7: {
        "morphology_keep": (0, 1, 2), "dedup_keep": (2,),
        "notes": {0: "Explicitly repeats extremely long legs in base.", 1: "Explicitly repeats small body and long abdomen in base.", 3: "Web/vibration descriptor; excluded only from morphology-only resource."},
    },
    8: {
        "morphology_keep": (0, 1, 2, 3), "dedup_keep": (3,),
        "notes": {0: "Explicitly repeats slender form in base.", 1: "Explicitly repeats exposed head in base.", 2: "Explicitly repeats two-segmented club in base.", 3: "Morphology-only replacement removes 'bores hardwood'."},
    },
    9: {
        "morphology_keep": (0, 1, 2, 3), "dedup_keep": (1, 3),
        "notes": {0: "Explicitly repeats very tiny elongate brown body in base.", 2: "Explicitly repeats clubbed antennae in base."},
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TextCache:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer, self.cache = model, tokenizer, {}
        self.max_length = model.config.text_config.max_position_embeddings

    def encode(self, prompts):
        unknown = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self.cache))
        device = next(self.model.parameters()).device
        for start in range(0, len(unknown), 128):
            batch = unknown[start:start + 128]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                output = self.model.text_model(**inputs)
                eos = inputs["attention_mask"].sum(dim=1) - 1
                text = F.normalize(self.model.text_projection(output.last_hidden_state[torch.arange(len(batch), device=device), eos]), dim=-1).cpu()
            self.cache.update({prompt: vector for prompt, vector in zip(batch, text)})
        return torch.stack([self.cache[prompt] for prompt in prompts])

    def token_audit(self, prompts, metadata):
        rows = []
        for prompt, row in zip(prompts, metadata):
            raw = self.tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
            effective = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length)["input_ids"]
            rows.append({**row, "prompt": prompt, "raw_tokens": len(raw), "effective_tokens": len(effective), "max_positions": self.max_length, "truncated": len(raw) > self.max_length, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
        return rows


def load_data(dataset, feature_path):
    base = np.load(feature_path)
    image = torch.from_numpy(base["image_features"].astype(np.float32))
    index = {str(stem): position for position, stem in enumerate(base["stems"])}
    metadata = [json.loads(line) for line in open(dataset / "annotations" / "metadata.jsonl", encoding="utf-8") if line.strip()]
    by_stem = {row["stem"]: row for row in metadata}
    class_ids = sorted({row["class_id"] for row in metadata})
    knowledge = sorted(json.load(open(dataset / "annotations" / "class_knowledge.json", encoding="utf-8"))["classes"], key=lambda row: row["class_id"])

    def split(name):
        values = json.load(open(dataset / "splits" / f"{name}.json", encoding="utf-8"))["all_samples"]
        stems = [Path(value).stem for value in values]
        return stems, torch.tensor([index[stem] for stem in stems]), np.asarray([class_ids.index(by_stem[stem]["class_id"]) for stem in stems])

    return image, index, metadata, by_stem, class_ids, knowledge, {name: split(name) for name in ("train_pool", "val", "test")}


def build_resources(knowledge):
    resources = {name: [] for name in ("original", "morphology_only", "deduplicated")}
    manifest_rows = []
    for item in knowledge:
        class_id = item["class_id"]
        attrs = item["key_points_en"]
        rules = CONTROL_RULES[class_id]
        for index, original in enumerate(attrs):
            morph_text = original
            if class_id == 8 and index == 3:
                morph_text = "striated elytra"
            residual = original
            if class_id == 6 and index == 3:
                residual = "no cerci"
            keep_morph = index in rules["morphology_keep"]
            keep_dedup = index in rules["dedup_keep"]
            manifest_rows.append({
                "resource_version": RESOURCE_VERSION, "class_id": class_id, "class_name": item["name_en"], "attribute_index": index,
                "original_descriptor": original, "morphology_only_descriptor": morph_text if keep_morph else "",
                "deduplicated_descriptor": residual if keep_dedup else "", "in_morphology_only": keep_morph,
                "in_deduplicated": keep_dedup, "decision_note": rules["notes"].get(index, "No exceptional rule; descriptor retained in this resource."),
            })
        resources["original"].append([(i, value) for i, value in enumerate(attrs)])
        resources["morphology_only"].append([(i, morph_text if class_id == 8 and i == 3 else value) for i, value in enumerate(attrs) if i in rules["morphology_keep"]])
        resources["deduplicated"].append([(i, "no cerci" if class_id == 6 and i == 3 else value) for i, value in enumerate(attrs) if i in rules["dedup_keep"]])
    return resources, manifest_rows


def attribute_text(cache, resource):
    return [cache.encode([f"a close-up of an insect showing {text}." for _, text in entries]) for entries in resource]


def rank_attributes(image, support_indices, support_labels, texts, resource, selector):
    ranked, audit = [], []
    support = image[support_indices]
    for class_index, features in enumerate(texts):
        own, other = support[support_labels == class_index], support[support_labels != class_index]
        rows = []
        for local_index, feature in enumerate(features):
            positive, negative = float((own @ feature).mean()), float((other @ feature).mean())
            score = positive if selector == "positive_mean" else positive - negative
            rows.append((score, local_index, positive, negative))
        rows.sort(key=lambda value: (-value[0], value[1]))
        ranked.append([local for _, local, _, _ in rows])
        audit.extend({"selector": selector, "class_index": class_index, "local_index": local, "original_attribute_index": resource_index, "positive_mean": pos, "negative_mean": neg, "ranking_score": score, "rank": rank} for rank, (score, local, pos, neg) in enumerate(rows, 1) for resource_index, _ in [resource[class_index][local]])
    return ranked, audit


def compose(knowledge, resource, ranked, budget):
    prompts, chosen = [], []
    for item, entries, order in zip(knowledge, resource, ranked):
        local = sorted(order[:min(budget, len(order))], key=lambda position: entries[position][0])
        selected = [entries[position] for position in local]
        chosen.append(selected)
        prompts.append(item["caption_en"] if not selected else f"{item['caption_en']}. Diagnostic attributes: {', '.join(text for _, text in selected)}.")
    return prompts, chosen


def evaluate(image, indices, labels, text):
    prediction = (image[indices] @ text.T).numpy().argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "prediction": prediction,
    }


def choose_k(rows):
    best = max(row["validation_accuracy"] for row in rows)
    return min(row["k"] for row in rows if row["validation_accuracy"] == best)


def make_group_units(stems, by_stem, groups):
    group_by_stem = {}
    for group in groups:
        for stem in group["stems"]:
            group_by_stem[stem] = group["group_id"]
    units = {}
    for stem in stems:
        key = group_by_stem.get(stem, f"singleton::{stem}")
        units.setdefault(key, []).append(stem)
    for key, values in units.items():
        classes = {by_stem[stem]["class_id"] for stem in values}
        if len(classes) != 1:
            raise ValueError(f"Near-duplicate group {key} crosses classes.")
    return list(units.values())


def bootstrap_stems(units_by_class, rng):
    output = []
    for class_id in sorted(units_by_class):
        units = units_by_class[class_id]
        for _ in range(len(units)):
            output.extend(rng.choice(units))
    return output


def run_backbone(args, backbone, model_path, feature_path, outdir):
    dataset = Path(args.dataset)
    image, index, metadata, by_stem, class_ids, knowledge, splits = load_data(dataset, feature_path)
    resources, manifest = build_resources(knowledge)
    model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
    cache = TextCache(model, tokenizer)
    outputs = {"control": [], "tokens": [], "ranking": [], "random": [], "bootstrap": [], "bootstrap_predictions": [], "bootstrap_confusion": [], "bootstrap_all_k_predictions": []}
    fixed = {name: (stems, indices, labels) for name, (stems, indices, labels) in splits.items()}

    # Controls use all original support images and the original held-out V/Q split.
    original_ranked = {}
    for resource_name, resource in resources.items():
        texts = attribute_text(cache, resource)
        for selector in SELECTORS:
            ranked, audit = rank_attributes(image, fixed["train_pool"][1], torch.from_numpy(fixed["train_pool"][2]), texts, resource, selector)
            outputs["ranking"].extend([{**row, "backbone": backbone, "resource": resource_name} for row in audit])
            if resource_name == "original":
                original_ranked[selector] = ranked
            for budget in BUDGETS:
                prompts, chosen = compose(knowledge, resource, ranked, budget)
                metadata_rows = [{"backbone": backbone, "resource": resource_name, "selector": selector, "k": budget, "class_index": class_index, "class_name": item["name_en"], "effective_descriptor_count": len(selection), "selected_original_indices": ";".join(str(index) for index, _ in selection)} for class_index, (item, selection) in enumerate(zip(knowledge, chosen))]
                outputs["tokens"].extend(cache.token_audit(prompts, metadata_rows))
                text = cache.encode(prompts)
                for split_name in ("val", "test"):
                    result = evaluate(image, fixed[split_name][1], fixed[split_name][2], text)
                    outputs["control"].append({"backbone": backbone, "resource": resource_name, "selector": selector, "split": split_name, "k": budget, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "mean_effective_descriptors": float(np.mean([len(selection) for selection in chosen])), "min_effective_descriptors": min(len(selection) for selection in chosen), "max_effective_descriptors": max(len(selection) for selection in chosen)})

    # Fair random baseline: each random full ranking receives the identical V-based k selection rule.
    for selector in SELECTORS:
        deterministic_rows = [row for row in outputs["control"] if row["resource"] == "original" and row["selector"] == selector]
        deterministic_selection = [{"k": k, "validation_accuracy": next(row["accuracy"] for row in deterministic_rows if row["split"] == "val" and row["k"] == k)} for k in BUDGETS]
        selected = choose_k(deterministic_selection)
        deterministic_test = next(row["accuracy"] for row in deterministic_rows if row["split"] == "test" and row["k"] == selected)
        outputs["random"].append({"backbone": backbone, "selector": selector, "policy": "support_conditioned", "draw": -1, "selected_k": selected, "validation_accuracy": max(row["validation_accuracy"] for row in deterministic_selection), "test_accuracy": deterministic_test})
    resource = resources["original"]
    for draw in range(args.random_draws):
        rng = random.Random(args.random_seed + draw)
        ranked = []
        for entries in resource:
            order = list(range(len(entries))); rng.shuffle(order); ranked.append(order)
        curve = []
        for budget in BUDGETS:
            prompts, _ = compose(knowledge, resource, ranked, budget)
            text = cache.encode(prompts)
            val = evaluate(image, fixed["val"][1], fixed["val"][2], text)["accuracy"]
            test = evaluate(image, fixed["test"][1], fixed["test"][2], text)["accuracy"]
            curve.append({"k": budget, "validation_accuracy": val, "test_accuracy": test})
        selected = choose_k(curve)
        chosen = next(row for row in curve if row["k"] == selected)
        outputs["random"].append({"backbone": backbone, "selector": "random", "policy": "random_ranking_same_calibration", "draw": draw, "selected_k": selected, "validation_accuracy": chosen["validation_accuracy"], "test_accuracy": chosen["test_accuracy"]})

    # Near-duplicate group bootstrap.  The unit is a visual-similarity group, not a specimen.
    group_data = json.load(open(dataset / "annotations" / "leakage_groups.json", encoding="utf-8"))["groups"]
    split_units = {}
    for split_name in ("train_pool", "val"):
        units = make_group_units(fixed[split_name][0], by_stem, group_data)
        split_units[split_name] = {class_id: [unit for unit in units if by_stem[unit[0]]["class_id"] == class_id] for class_id in class_ids}
    for replicate in range(args.bootstrap_replicates):
        support_stems = bootstrap_stems(split_units["train_pool"], random.Random(args.bootstrap_seed + replicate * 2))
        validation_stems = bootstrap_stems(split_units["val"], random.Random(args.bootstrap_seed + replicate * 2 + 1))
        support_idx = torch.tensor([index[stem] for stem in support_stems])
        support_labels = torch.tensor([class_ids.index(by_stem[stem]["class_id"]) for stem in support_stems])
        val_idx = torch.tensor([index[stem] for stem in validation_stems])
        val_labels = np.asarray([class_ids.index(by_stem[stem]["class_id"]) for stem in validation_stems])
        texts = attribute_text(cache, resources["original"])
        for selector in SELECTORS:
            ranked, _ = rank_attributes(image, support_idx, support_labels, texts, resources["original"], selector)
            curve = []
            for budget in BUDGETS:
                prompts, _ = compose(knowledge, resources["original"], ranked, budget)
                text = cache.encode(prompts)
                validation = evaluate(image, val_idx, val_labels, text)
                query = evaluate(image, fixed["test"][1], fixed["test"][2], text)
                curve.append({"k": budget, "validation_accuracy": validation["accuracy"]})
                outputs["bootstrap_all_k_predictions"].extend({"backbone": backbone, "selector": selector, "replicate": replicate,
                    "k": budget, "stem": stem, "true_index": int(true), "predicted_index": int(pred)}
                    for stem, true, pred in zip(fixed["test"][0], fixed["test"][2], query["prediction"]))
            selected = choose_k(curve)
            selected_prediction = [r for r in outputs["bootstrap_all_k_predictions"] if r["backbone"] == backbone and r["selector"] == selector and r["replicate"] == replicate and r["k"] == selected]
            test_prediction = np.asarray([r["predicted_index"] for r in selected_prediction])
            test = {"accuracy": float((test_prediction == fixed["test"][2]).mean()), "macro_f1": float(f1_score(fixed["test"][2], test_prediction, average="macro")), "prediction": test_prediction}
            recalls = recall_score(fixed["test"][2], test["prediction"], labels=np.arange(len(class_ids)), average=None, zero_division=0)
            outputs["bootstrap"].append({"backbone": backbone, "selector": selector, "replicate": replicate, "selected_k": selected, "validation_accuracy": max(row["validation_accuracy"] for row in curve), "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"], "support_images_drawn": len(support_stems), "validation_images_drawn": len(validation_stems)})
            outputs["bootstrap_predictions"].extend({"backbone": backbone, "selector": selector, "replicate": replicate, "selected_k": selected, "stem": stem, "true_index": int(true), "predicted_index": int(pred), "class_recall": float(recalls[true])} for stem, true, pred in zip(fixed["test"][0], fixed["test"][2], test["prediction"]))
            matrix = confusion_matrix(fixed["test"][2], test["prediction"], labels=np.arange(len(class_ids)))
            outputs["bootstrap_confusion"].extend({"backbone": backbone, "selector": selector, "replicate": replicate, "selected_k": selected, "true_index": i, "predicted_index": j, "count": int(matrix[i, j])} for i in range(len(class_ids)) for j in range(len(class_ids)))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs, manifest, {"backbone": backbone, "model": model_path, "feature_cache": str(feature_path), "feature_cache_sha256": sha256(feature_path), "resource_version": RESOURCE_VERSION, "random_draws": args.random_draws, "random_seed": args.random_seed, "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.bootstrap_seed, "near_duplicate_groups": "connected components supplied by leakage_groups.json; not specimen IDs"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pest_vlm_dataset")
    parser.add_argument("--output-dir", default="experiments/results/experiment_54_teacher_targeted_validation")
    parser.add_argument("--random-draws", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260930)
    parser.add_argument("--bootstrap-replicates", type=int, default=50)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    parser.add_argument("--model-b32", default="openai/clip-vit-base-patch32")
    parser.add_argument("--model-b16", default="models/clip-vit-base-patch16")
    parser.add_argument("--features-b32", default="experiments/results/visual_audit_crop4_features.npz")
    parser.add_argument("--features-b16", default="experiments/results/visual_audit_crop4_features_b16.npz")
    args = parser.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    all_outputs = {key: [] for key in ("control", "tokens", "ranking", "random", "bootstrap", "bootstrap_predictions", "bootstrap_confusion", "bootstrap_all_k_predictions")}
    all_metadata, manifest = [], None
    jobs = (("ViT-B-32", args.model_b32, args.features_b32), ("ViT-B-16", args.model_b16, args.features_b16))
    for backbone, model_path, feature_path in jobs:
        output, current_manifest, metadata = run_backbone(args, backbone, model_path, feature_path, outdir)
        for key, rows in output.items(): all_outputs[key].extend(rows)
        all_metadata.append(metadata)
        manifest = current_manifest
    pd.DataFrame(manifest).to_csv(outdir / "candidate_resource_manifest_v1.csv", index=False, encoding="utf-8-sig")
    for name, rows in all_outputs.items():
        pd.DataFrame(rows).to_csv(outdir / f"teacher_{name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_metadata).to_csv(outdir / "teacher_run_metadata.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(all_outputs["control"]).query("split == 'test'").groupby(["backbone", "resource", "selector", "k"], as_index=False).accuracy.mean().to_string(index=False))


if __name__ == "__main__":
    main()
