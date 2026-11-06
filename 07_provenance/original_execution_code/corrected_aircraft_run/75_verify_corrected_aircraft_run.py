"""Read-only consistency audit for the corrected clean Aircraft run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_SHA = "67876cc858a9b6a1785b7eaa229a54dd40ad3394de7a7c989ad0d77b803808fc"
AUDIT_SHA = "bdb085a4bbfaa1ec8a732e2ec0930388cfe07bdee5e726391ef0221ff18433b5"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="external_aircraft_corrected_clean")
    args = ap.parse_args()
    output = ROOT / args.output
    errors = []
    pre = json.loads((output / "pre_run_checks.json").read_text(encoding="utf-8"))
    if pre.get("status") != "PASS": errors.append("pre-run checks did not pass")
    if pre.get("resource_sha256") != RESOURCE_SHA: errors.append("resource hash mismatch")
    if pre.get("audit_sha256") != AUDIT_SHA: errors.append("audit hash mismatch")
    frozen = json.loads((output / "PROTOCOL_CORRECTED_CLEAN_FROZEN.json").read_text(encoding="utf-8"))
    if frozen.get("test_access") != "unlocked_after_pre_run_checks_PASS": errors.append("test-unlock provenance missing")
    curves = pd.read_csv(output / "budget_curves.csv")
    selected = pd.read_csv(output / "selected_budgets.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    ranked = pd.read_csv(output / "ranked_descriptors.csv")
    bootstrap = pd.read_csv(output / "paired_bootstrap.csv")
    if len(curves) != 200 or set(curves.split) != {"validation", "test"}: errors.append("curve cardinality/splits invalid")
    if len(selected) != 20 or not selected.selected_k.between(0, 4).all(): errors.append("selected-budget rows invalid")
    if len(ranked) != 8000: errors.append("ranked-descriptor cardinality invalid")
    if predictions.groupby("split").size().to_dict() != {"test": 333300, "validation": 50000}: errors.append("prediction cardinality invalid")
    if len(bootstrap) != 60 or not (bootstrap.bootstrap_replicates == 20000).all(): errors.append("bootstrap cardinality invalid")
    if len(list((output / "feature_cache").glob("*.npz"))) != 2: errors.append("feature-cache count invalid")
    if len(list((output / "logits").glob("*.npz"))) != 100: errors.append("logit archive count invalid")
    if len(list((output / "bootstrap_inputs").glob("*.npz"))) != 60 or len(list((output / "bootstrap_outputs").glob("*.npz"))) != 60: errors.append("bootstrap archive count invalid")
    confusion = json.loads((output / "confusion_matrices.json").read_text(encoding="utf-8"))
    if len(confusion) != 100 or any(np.asarray(row["matrix"]).shape != (100, 100) for row in confusion): errors.append("confusion-matrix archive invalid")
    for backbone in ("ViT-B-16", "ViT-B-32"):
        cache = np.load(output / "feature_cache" / f"image_features_{backbone}.npz")
        if cache["features"].shape[0] != 10000 or len(cache["image_ids"]) != 10000: errors.append(f"{backbone} cache alignment invalid")
    for row in selected.itertuples(index=False):
        test = predictions[(predictions.split == "test") & (predictions.seed == row.seed) & (predictions.backbone == row.backbone) & (predictions.scorer == row.scorer)]
        for contrast, left_k, right_k in (("selected_minus_k4", int(row.selected_k), 4), ("k1_minus_k4", 1, 4), ("selected_minus_k1", int(row.selected_k), 1)):
            left, right = test[test.k == left_k].sort_values("image_id"), test[test.k == right_k].sort_values("image_id")
            d = (left.predicted_index.to_numpy() == left.true_index.to_numpy()).astype(float) - (right.predicted_index.to_numpy() == right.true_index.to_numpy()).astype(float)
            observed = np.mean([d[left.true_index.to_numpy() == class_id].mean() for class_id in range(100)])
            reported = bootstrap[(bootstrap.seed == row.seed) & (bootstrap.backbone == row.backbone) & (bootstrap.scorer == row.scorer) & (bootstrap.contrast == contrast)].iloc[0]
            if not np.isclose(observed, reported.macro_accuracy_difference): errors.append("paired-bootstrap observed macro effect mismatch")
    shutil.copy2(ROOT / "external_aircraft/config/model_fingerprints.json", output / "model_fingerprints.json")
    runtime = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "pillow": PIL.__version__, "scikit_learn": sklearn.__version__, "tokenizer": "CLIPTokenizer (local_files_only=True)", "image_processor": "CLIPProcessor (local_files_only=True)", "device": "cpu"}
    (output / "runtime_identity.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "resource_sha256": sha(ROOT / "external_aircraft/resources/aircraft_descriptors_clean.json"), "audit_sha256": sha(ROOT / "external_aircraft/resources/aircraft_descriptor_audit.csv"), "outputs": {"curves": len(curves), "selected_budgets": len(selected), "predictions": len(predictions), "rankings": len(ranked), "logit_archives": len(list((output / "logits").glob("*.npz"))), "bootstrap_contrasts": len(bootstrap)}}
    (output / "post_run_consistency_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(2)


if __name__ == "__main__":
    main()
