"""Contextual frozen-image support-centroid reference for the main benchmark.

This is deliberately not a text-prompt baseline: it directly averages labelled
support image embeddings.  It is included to bound practical classification
claims while leaving the single-text-prototype analysis as the paper's target.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "pest_vlm_dataset"
OUT = ROOT / "experiments" / "results" / "experiment_63_support_centroid_reference"
JOBS = (("ViT-B-32", ROOT / "experiments/results/visual_audit_crop4_features.npz"),
        ("ViT-B-16", ROOT / "experiments/results/visual_audit_crop4_features_b16.npz"))


def stems_for(split):
    payload = json.loads((DATASET / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    return [Path(path).stem for path in payload["all_samples"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {row["stem"]: row for row in map(json.loads, (DATASET / "annotations/metadata.jsonl").read_text(encoding="utf-8").splitlines())}
    class_ids = sorted({row["class_id"] for row in metadata.values()})
    rows, predictions = [], []
    support_stems = stems_for("train_pool")
    for backbone, feature_path in JOBS:
        cache = np.load(feature_path)
        index = {str(stem): i for i, stem in enumerate(cache["stems"])}
        image = cache["image_features"].astype(np.float64)
        prototypes = []
        for class_id in class_ids:
            indices = [index[stem] for stem in support_stems if metadata[stem]["class_id"] == class_id]
            centroid = image[indices].mean(axis=0)
            prototypes.append(centroid / np.linalg.norm(centroid))
        prototypes = np.asarray(prototypes)
        for split in ("val", "test"):
            stems = stems_for(split)
            indices = np.asarray([index[stem] for stem in stems])
            labels = np.asarray([class_ids.index(metadata[stem]["class_id"]) for stem in stems])
            pred = (image[indices] @ prototypes.T).argmax(axis=1)
            rows.append({"backbone": backbone, "split": split, "support_images": len(support_stems),
                         "n_images": len(stems), "n_correct": int((pred == labels).sum()),
                         "accuracy": accuracy_score(labels, pred), "macro_f1": f1_score(labels, pred, average="macro"),
                         "protocol": "normalised class mean of frozen labelled support-image embeddings"})
            predictions.extend({"backbone": backbone, "split": split, "stem": stem, "true_index": int(y), "predicted_index": int(p)}
                               for stem, y, p in zip(stems, labels, pred))
    pd.DataFrame(rows).to_csv(OUT / "support_centroid_reference.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "support_centroid_predictions.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
