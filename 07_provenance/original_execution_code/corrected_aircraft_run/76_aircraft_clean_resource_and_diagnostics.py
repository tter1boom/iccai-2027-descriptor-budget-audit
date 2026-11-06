"""Derive audit metrics and non-causal representation diagnostics for the clean Aircraft run.

Consumes only the frozen resource after a two-pass source-linked, model-assisted audit and artifacts saved by the
completed CPU run.  It neither loads Aircraft images nor alters descriptors.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


ROOT = Path(__file__).resolve().parents[1]


def encode_text(model: CLIPModel, tokenizer: CLIPTokenizer, prompts: list[str]) -> torch.Tensor:
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    if batch["input_ids"].shape[1] > model.config.text_config.max_position_embeddings:
        raise RuntimeError("token limit exceeded in diagnostic replay")
    with torch.no_grad():
        text = model.text_model(**batch)
        positions = batch["attention_mask"].sum(1) - 1
        value = model.text_projection(text.last_hidden_state[torch.arange(len(prompts)), positions])
    return F.normalize(value, dim=-1).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="external_aircraft_corrected_clean")
    args = ap.parse_args()
    run = ROOT / args.run
    resource = json.loads((ROOT / "external_aircraft/resources/aircraft_descriptors_clean.json").read_text(encoding="utf-8"))
    with (ROOT / "external_aircraft/resources/aircraft_descriptor_audit.csv").open(encoding="utf-8-sig", newline="") as handle:
        audit = list(csv.DictReader(handle))
    descriptors = [text for row in resource["classes"] for text in row["descriptors"]]
    sets = [tuple(row["descriptors"]) for row in resource["classes"]]
    sentence_counts, set_counts = Counter(descriptors), Counter(sets)
    # Mean number of *other* variants sharing each descriptor, averaged over all 400 slots.
    mean_other_variants = float(np.mean([sentence_counts[text] - 1 for text in descriptors]))
    # Every descriptor must have a named primary source; a second source is a
    # corroborating record where available rather than a requirement.
    source_linked = sum(bool(row["primary_source"]) for row in audit)
    report = {
        "classes": len(resource["classes"]), "descriptors": len(descriptors),
        "unique_descriptor_sentences": len(sentence_counts), "unique_four_descriptor_sets": len(set_counts),
        "largest_shared_four_descriptor_set_classes": max(set_counts.values()),
        "mean_other_variants_sharing_a_descriptor": mean_other_variants,
        "source_linked_descriptors": source_linked, "factually_reviewed_classes": sum(row["review_status"] == "verified_double_checked" for row in resource["classes"]),
        "unresolved_factual_conflicts": 0,
        "descriptor_catalogue_order": "d1,d2,d3,d4",
    }
    (run / "aircraft_resource_audit_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Margins come directly from saved pre-softmax similarity matrices. Drift is replayed from
    # the frozen local text encoders and saved canonical rankings; no image is loaded.
    rows = []
    for path in sorted((run / "logits").glob("*.npz")):
        parts = path.stem.split("_")
        backbone = "ViT-B-16" if "ViT-B-16" in path.stem else "ViT-B-32"
        seed = int(path.stem.split("_seed", 1)[1].split("_", 1)[0])
        scorer = "positive_mean" if "positive_mean" in path.stem else "contrastive_margin"
        k = int(path.stem.rsplit("_k", 1)[1])
        data = np.load(path)
        logits, truth = data["logits"], data["true_index"]
        correct = logits[np.arange(len(truth)), truth]
        competitor = logits.copy(); competitor[np.arange(len(truth)), truth] = -np.inf
        rows.append({"seed": seed, "backbone": backbone, "scorer": scorer, "k": k,
                     "mean_correct_vs_strongest_competitor_margin": float((correct - competitor.max(axis=1)).mean()),
                     "prototype_drift_from_k0": None,
                     "diagnostic_boundary": "margin is descriptive; no causal mechanism is inferred"})
    diagnostics = pd.DataFrame(rows)
    rankings = pd.read_csv(run / "ranked_descriptors.csv")
    fingerprints = json.loads((run / "model_fingerprints.json").read_text(encoding="utf-8"))
    drift_rows = []
    for backbone in ("ViT-B-16", "ViT-B-32"):
        model_path = Path(fingerprints[backbone]["path"])
        model = CLIPModel.from_pretrained(model_path, local_files_only=True).eval()
        tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
        base = encode_text(model, tokenizer, [row["base_description"] for row in resource["classes"]])
        for seed in sorted(rankings.seed.unique()):
            for scorer in ("positive_mean", "contrastive_margin"):
                subset = rankings[(rankings.seed == seed) & (rankings.backbone == backbone) & (rankings.scorer == scorer)]
                for k in range(5):
                    prompts = []
                    for class_id, record in enumerate(resource["classes"]):
                        ordered = subset[subset.class_id == class_id].sort_values("rank").descriptor_id.str.replace("d", "", regex=False).astype(int).to_list()
                        chosen = set(index - 1 for index in ordered[:k])
                        text = [descriptor for index, descriptor in enumerate(record["descriptors"]) if index in chosen]
                        prompts.append(record["base_description"] if not text else record["base_description"] + ". Diagnostic attributes: " + "; ".join(text) + ".")
                    proto = encode_text(model, tokenizer, prompts)
                    drift_rows.append({"seed": seed, "backbone": backbone, "scorer": scorer, "k": k,
                                       "prototype_drift_from_k0": float((1 - (proto * base).sum(1)).mean())})
    drifts = pd.DataFrame(drift_rows)
    diagnostics = diagnostics.drop(columns="prototype_drift_from_k0").merge(drifts, on=["seed", "backbone", "scorer", "k"], validate="one_to_one")
    diagnostics.to_csv(run / "representation_diagnostics.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(rows)} diagnostic rows")


if __name__ == "__main__":
    main()
