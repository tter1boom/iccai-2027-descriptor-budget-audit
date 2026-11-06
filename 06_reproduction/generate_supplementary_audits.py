"""Build compact, source-linked tables for the ICCAI supplementary appendix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty_like(raw)
    running = 0.0
    m = len(raw)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def write_classwise_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(ROOT / "05_results/aircraft_classwise_deltas.csv")
    source = source[
        source.backbone.eq("ViT-B-16")
        & source.ranker.eq("positive_mean")
        & source.comparison.eq("k1_vs_k0")
    ].copy()
    if len(source) != 500 or source.class_id.nunique() != 100:
        raise RuntimeError("Expected 100 classes x 5 support-resample cells for B/16 PM k=1 vs k=0")
    if not source.groupby("class_id").size().eq(5).all():
        raise RuntimeError("Each class must have exactly five support-resample records")

    resource = json.loads((ROOT / "02_resources/aircraft_descriptors_frozen.json").read_text(encoding="utf-8"))
    class_names = {int(row["class_id"]): row["name_en"] for row in resource["classes"]}
    rows = []
    for class_id, group in source.groupby("class_id", sort=True):
        rows.append({
            "class_id": int(class_id),
            "class_name": class_names[int(class_id)],
            "test_images_per_class": int(group.n_images.iloc[0]),
            "support_resamples": int(group.seed.nunique()),
            "resample_cells_with_any_gain": int((group.gained_correct > 0).sum()),
            "resample_cells_with_any_loss": int((group.lost_correct > 0).sum()),
            "gained_correct_across_resamples": int(group.gained_correct.sum()),
            "lost_correct_across_resamples": int(group.lost_correct.sum()),
            "net_correct_change_across_resamples": int(group.net_correct_change.sum()),
            "mean_accuracy_change_pp_across_resamples": float((group.comparison_accuracy - group.baseline_accuracy).mean() * 100),
        })
    per_class = pd.DataFrame(rows)
    per_class.to_csv(ROOT / "05_results/aircraft_per_class_gains_losses.csv", index=False)

    cells_gain = int((source.gained_correct > 0).sum())
    cells_loss = int((source.lost_correct > 0).sum())
    net_by_class = per_class.net_correct_change_across_resamples
    total_gain = int(source.gained_correct.sum())
    total_loss = int(source.lost_correct.sum())
    top = per_class.sort_values(["net_correct_change_across_resamples", "class_id"], ascending=[False, True]).iloc[0]
    summary = pd.DataFrame([
        {"measure": "Class-by-resample cells with at least one gained correct prediction", "value": f"{cells_gain}/500", "unit": "100 classes x 5 support resamples", "interpretation": "Descriptive; cells share the fixed test split."},
        {"measure": "Class-by-resample cells with at least one lost correct prediction", "value": f"{cells_loss}/500", "unit": "100 classes x 5 support resamples", "interpretation": "Descriptive; cells share the fixed test split."},
        {"measure": "Classes with positive net change across five resamples", "value": int((net_by_class > 0).sum()), "unit": "classes", "interpretation": "Net gained minus lost correct predictions, summed over resamples."},
        {"measure": "Classes with zero net change across five resamples", "value": int((net_by_class == 0).sum()), "unit": "classes", "interpretation": "Net gained minus lost correct predictions, summed over resamples."},
        {"measure": "Classes with negative net change across five resamples", "value": int((net_by_class < 0).sum()), "unit": "classes", "interpretation": "Net gained minus lost correct predictions, summed over resamples."},
        {"measure": "Gained correct predictions across five resamples", "value": total_gain, "unit": "class-image-resample events", "interpretation": "The same test images are evaluated in each support resample."},
        {"measure": "Lost correct predictions across five resamples", "value": total_loss, "unit": "class-image-resample events", "interpretation": "The same test images are evaluated in each support resample."},
        {"measure": "Net correct-prediction change across five resamples", "value": total_gain - total_loss, "unit": "class-image-resample events", "interpretation": "Gained minus lost events; not a count of unique test images."},
        {"measure": "Largest positive class net change", "value": f"{top.class_name}: {int(top.net_correct_change_across_resamples)}", "unit": "correct-prediction events", "interpretation": "Class-level sum over the five fixed test-set evaluations."},
    ])
    summary.to_csv(ROOT / "05_results/aircraft_per_class_summary.csv", index=False)
    return per_class, summary


def write_adjacent_p_audit() -> pd.DataFrame:
    result_path = ROOT / "05_results/adjacent_inference.csv"
    manifest_path = ROOT / "05_results/analysis_outputs/adjacent_budget_inference/analysis_manifest.json"
    frame = pd.read_csv(result_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(frame) != 16 or frame.n_permutations.nunique() != 1 or int(frame.n_permutations.iloc[0]) != 100000:
        raise RuntimeError("Adjacent p-value audit expects 16 contrasts x 100,000 sign flips")
    raw = (frame.exceedance_count.to_numpy(dtype=np.int64) + 1) / (frame.n_permutations.to_numpy(dtype=np.int64) + 1)
    holm = holm_adjust(raw)
    if not np.allclose(raw, frame.raw_p.to_numpy(dtype=float), rtol=0, atol=1e-15):
        raise RuntimeError("Raw Monte Carlo p-values do not match exact exceedance counts")
    if not np.allclose(holm, frame.holm_p.to_numpy(dtype=float), rtol=0, atol=1e-14):
        raise RuntimeError("Holm p-values do not match recomputation from the 16 exact raw p-values")
    frame.insert(0, "random_seed", int(manifest["random_seed"]))
    frame.to_csv(ROOT / "05_results/adjacent_signflip_p_value_audit.csv", index=False)
    return frame


def write_token_and_checkpoint_audits(models_root: Path) -> tuple[pd.DataFrame, dict]:
    replay_dir = ROOT / "05_results/replayed_additional_controls"
    token_rows = pd.read_csv(replay_dir / "token_audit.csv")
    token_rows = token_rows[token_rows.condition.str.fullmatch(r"random_recomposition_k[0-4]")].copy()
    expected_n = {0: 100, 1: 400, 2: 600, 3: 400, 4: 100}
    out = []
    for backbone in ("ViT-B-16", "ViT-B-32"):
        for k in range(5):
            row = token_rows[token_rows.backbone.eq(backbone) & token_rows.condition.eq(f"random_recomposition_k{k}")]
            if len(row) != 1:
                raise RuntimeError(f"Missing unique token audit for {backbone}, k={k}")
            item = row.iloc[0]
            if int(item.n_prompts) != expected_n[k]:
                raise RuntimeError(f"Token audit does not cover all catalogue subsets for {backbone}, k={k}")
            if int(item.n_over_context) != 0 or item.truncation != "disabled; overflow would fail":
                raise RuntimeError(f"Prompt token limit/truncation audit failed for {backbone}, k={k}")
            out.append({
                "backbone": backbone,
                "k": k,
                "unique_class_specific_prompts": int(item.n_prompts),
                "min_tokens": int(item.min_tokens),
                "max_tokens": int(item.max_tokens),
                "context_limit": int(item.context_length),
                "over_limit_prompts": int(item.n_over_context),
                "truncation": item.truncation,
                "coverage": "all class-specific catalogue subsets in canonical catalogue order",
            })
    token_summary = pd.DataFrame(out)
    token_summary.to_csv(ROOT / "01_protocol/prompt_token_audit_by_budget.csv", index=False)

    lock = json.loads((ROOT / "01_protocol/model_lock.json").read_text(encoding="utf-8"))
    models = {}
    tokenizer_names = {"config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    for backbone, record in lock["models"].items():
        local_dir = (models_root / record["relative_path"]).resolve()
        files = record["files"]
        actual = {}
        for name, expected_hash in files.items():
            path = local_dir / name
            if not path.is_file() or sha256(path) != expected_hash:
                raise RuntimeError(f"Checkpoint fingerprint mismatch while building metadata: {backbone}/{name}")
            actual[name] = expected_hash
        config = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
        models[backbone] = {
            "checkpoint_id": f"frozen local CLIP {backbone}",
            "relative_path_from_models_root": record["relative_path"],
            "upstream_model_id": "not recorded in the preserved lock; identified by local relative path and hashes",
            "model_weights_sha256": files["pytorch_model.bin"],
            "tokenizer_class": "CLIPTokenizer",
            "tokenizer_file_sha256": {name: digest for name, digest in actual.items() if name in tokenizer_names},
            "all_locked_checkpoint_files_sha256": actual,
            "context_length": int(config["text_config"]["max_position_embeddings"]),
            "device_policy": record.get("device_policy", "CPU"),
        }
    metadata = {
        "record_type": "checkpoint, tokenizer, runtime and code fingerprint manifest",
        "runtime_declared_in_environment_yml": {
            "python": "3.14.7",
            "pytorch": "2.13.0",
            "transformers": "5.16.1",
        },
        "runtime_observed_during_audit_generation": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizer_class": "CLIPTokenizer",
            "device": "CPU",
        },
        "source_commit": "not recorded in the preserved execution records",
        "active_code_fingerprint": "ACTIVE_CODE_MANIFEST.csv and 08_integrity/SHA256SUMS.tsv",
        "models": models,
        "prompt_token_audit": "prompt_token_audit_by_budget.csv; exhaustive class-specific subsets for k=0..4; truncation disabled",
    }
    ROOT.joinpath("01_protocol/checkpoint_tokenizer_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return token_summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True, help="Local root for frozen checkpoints in 01_protocol/model_lock.json")
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 14, 7):
        raise RuntimeError(f"Expected the pinned Python 3.14.7 environment, got {sys.version.split()[0]}")
    per_class, summary = write_classwise_tables()
    p_audit = write_adjacent_p_audit()
    tokens, metadata = write_token_and_checkpoint_audits(args.models_root.resolve())
    print(f"PASS: wrote 100 class rows and {len(summary)} summary measures")
    print(f"PASS: rebuilt 16 raw and Holm p-values from exact counts; seed={p_audit.random_seed.iloc[0]}")
    print(f"PASS: wrote {len(tokens)} per-backbone budget token rows; max={tokens.max_tokens.max()} / 77; overflow=0")
    print(f"PASS: checkpoint/tokenizer hashes match model_lock.json; runtime={metadata['runtime_observed_during_audit_generation']}")


if __name__ == "__main__":
    main()
