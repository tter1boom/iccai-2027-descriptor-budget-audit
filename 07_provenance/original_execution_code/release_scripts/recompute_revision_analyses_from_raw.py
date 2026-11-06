"""CPU-only replay of reviewer-driven analyses from per-image records/shards.

This is a statistics replay only: it never opens benchmark images or invokes a
vision model. ``--validate-only`` recomputes outputs in a temporary directory,
compares them with the current authoritative tables, and leaves the package
unchanged. ``--output-dir`` exports newly recomputed summaries separately.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "revision_experiments" / "analysis_package_v13" / "CODE"
MACHINE = ROOT / "machine_readable"
RESULTS = ROOT / "revision_experiments" / "analysis_package_v13" / "RESULTS"


def run(command: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())


def macro_metrics(group: pd.DataFrame, expected_classes: list[int] | None = None) -> tuple[float, float]:
    y = group["true_index"].to_numpy(dtype=int)
    p = group["predicted_index"].to_numpy(dtype=int)
    labels = np.asarray(expected_classes if expected_classes is not None else sorted(np.unique(y)))
    recall = np.asarray([np.mean(p[y == c] == c) if np.any(y == c) else 0.0 for c in labels])
    f1 = []
    for c in labels:
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        denominator = 2 * tp + fp + fn
        f1.append((2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(recall)), float(np.mean(f1))


def check_close(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], metrics: list[str], label: str,
                tolerance: float = 1e-6) -> None:
    joined = expected.merge(actual, on=keys, how="outer", suffixes=("_expected", "_actual"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(f"{label}: key mismatch; unmatched rows={joined.loc[~joined._merge.eq('both'), keys + ['_merge']].head().to_dict('records')}")
    for metric in metrics:
        left, right = joined[f"{metric}_expected"], joined[f"{metric}_actual"]
        finite = left.notna()
        if not np.allclose(left[finite], right[finite], atol=tolerance, rtol=0.0):
            error = (left[finite] - right[finite]).abs().max()
            raise RuntimeError(f"{label}: {metric} differs (max absolute error={error})")
    print(f"PASS: {label}: {len(joined)} summary rows regenerated from raw records")


def build_exp4() -> pd.DataFrame:
    rows: list[dict] = []
    prompt = pd.read_csv(RESULTS / "experiment_4_prompt_baselines" / "predictions.csv")
    for key, group in prompt.groupby(["domain", "backbone", "ranker", "prompt_condition"], sort=True):
        macro_acc, f1 = macro_metrics(group)
        top1 = float(np.mean(group.true_index.to_numpy() == group.predicted_index.to_numpy()))
        rows.append(dict(zip(["domain", "backbone", "ranker", "prompt_condition"], key),
                         macro_accuracy=macro_acc, top1_accuracy_secondary=top1, macro_f1=f1))

    main = MACHINE / "validation_and_predictions"
    kpred = pd.read_csv(main / "experiment_b_predictions.csv")
    base = pd.read_csv(main / "main_base_k0_per_image_predictions.csv")
    for backbone in sorted(kpred.backbone.unique()):
        for ranker in ("positive_mean", "contrastive_margin"):
            bg = base[(base.backbone == backbone) & (base.selector == ranker)]
            macro_acc, f1 = macro_metrics(bg)
            top1 = float(np.mean(bg.true_index.to_numpy() == bg.predicted_index.to_numpy()))
            rows.append(dict(domain="development", backbone=backbone, ranker=ranker,
                             prompt_condition="current_base", macro_accuracy=macro_acc,
                             top1_accuracy_secondary=top1, macro_f1=f1))
            for condition, k in (("fixed_k1", 1), ("full_k4", 4)):
                group = kpred[(kpred.backbone == backbone) & (kpred.selector == ranker) &
                              (kpred.composition == "recompose") & (kpred.split == "test") & (kpred.k == k)]
                macro_acc, f1 = macro_metrics(group, list(range(10)))
                top1 = float(np.mean(group.true_index.to_numpy() == group.predicted_index.to_numpy()))
                rows.append(dict(domain="development", backbone=backbone, ranker=ranker,
                                 prompt_condition=condition, macro_accuracy=macro_acc,
                                 top1_accuracy_secondary=top1, macro_f1=f1))

    aircraft = pd.read_csv(ROOT / "external_aircraft_clean" / "results" / "predictions.csv")
    aircraft = aircraft[(aircraft.split == "test") & aircraft.k.isin([0, 1, 4])]
    condition_by_k = {0: "current_base", 1: "fixed_k1", 4: "full_k4"}
    for (backbone, ranker, k, seed), group in aircraft.groupby(["backbone", "scorer", "k", "seed"], sort=True):
        macro_acc, f1 = macro_metrics(group, list(range(100)))
        top1 = float(np.mean(group.true_index.to_numpy() == group.predicted_index.to_numpy()))
        rows.append(dict(domain="aircraft", backbone=backbone, ranker=ranker,
                         prompt_condition=condition_by_k[int(k)], seed=int(seed), macro_accuracy=macro_acc,
                         top1_accuracy_secondary=top1, macro_f1=f1))
    result = pd.DataFrame(rows)
    result = result.groupby(["domain", "backbone", "ranker", "prompt_condition"], as_index=False, dropna=False).agg(
        macro_accuracy=("macro_accuracy", "mean"), top1_accuracy_secondary=("top1_accuracy_secondary", "mean"),
        macro_f1=("macro_f1", "mean"))
    return result


def build_exp5() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    raw = pd.read_csv(RESULTS / "experiment_5_matched_prior" / "predictions.csv")
    raw = raw[raw.analysis == "matched"]
    interface = {
        "single_recomposed_prototype": "single_recomposed_prototype",
        "independent_attribute_voting": "independent_attribute_voting",
        "class_anchored_descriptor_ensemble": "class_anchored_descriptor_ensemble",
    }
    for (backbone, seed, k, method), group in raw.groupby(["backbone", "seed", "k", "method"], sort=True):
        acc, _ = macro_metrics(group, list(range(100)))
        rows.append(dict(domain="aircraft", backbone=backbone, seed=int(seed), ranker="positive_mean",
                         k=int(k), interface=interface[method], test_macro_accuracy=acc))

    development = pd.read_csv(MACHINE / "validation_and_predictions" / "experiment_b_predictions.csv")
    development = development[(development.selector == "positive_mean") &
                               (development.split == "test")]
    name_map = {"recompose": "single_recomposed_prototype", "attribute_vote": "independent_attribute_voting",
                "base_plus_vote": "base_plus_vote"}
    for (backbone, k, composition), group in development.groupby(["backbone", "k", "composition"], sort=True):
        acc, _ = macro_metrics(group, list(range(10)))
        rows.append(dict(domain="development", backbone=backbone, seed=0, ranker="positive_mean",
                         k=int(k), interface=name_map[composition], test_macro_accuracy=acc))

    curves = pd.DataFrame(rows)
    as_rows: list[dict] = []
    as_raw = pd.read_csv(RESULTS / "experiment_5_matched_prior" / "predictions.csv")
    as_raw = as_raw[as_raw.analysis == "attributeselect_reference"]
    for (backbone, seed, k, method), group in as_raw.groupby(["backbone", "seed", "k", "method"], sort=True):
        acc, _ = macro_metrics(group, list(range(100)))
        as_rows.append(dict(backbone=backbone, seed=int(seed), k=int(k), reference=method,
                            test_macro_accuracy=acc))
    refs = pd.DataFrame(as_rows)
    return curves, refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="recompute in a temporary folder and compare; do not modify the package")
    parser.add_argument("--output-dir", type=Path, help="optional new directory for recomputed CSV summaries")
    args = parser.parse_args()
    if args.validate_only and args.output_dir:
        raise SystemExit("Choose either --validate-only or --output-dir")

    if args.output_dir and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {args.output_dir}")
    temp_context = tempfile.TemporaryDirectory(prefix="review_analysis_raw_replay_")
    work_root = Path(temp_context.name)
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        exp1_out = work_root / "experiment_1"
        exp2_out = work_root / "experiment_2"
        exp3_out = work_root / "experiment_3"
        exp1 = CODE / "run_exp1_aircraft_adjacent.py"
        exp2 = CODE / "run_exp2_class_concentration.py"
        exp3 = CODE / "summarize_exp3_checkpoint.py"
        run([sys.executable, "-B", str(exp1), "--predictions", str(ROOT / "external_aircraft_clean/results/predictions.csv"),
             "--output-dir", str(exp1_out)], ROOT)
        run([sys.executable, "-B", str(exp2), "--predictions", str(MACHINE / "validation_and_predictions/experiment_b_predictions.csv"),
             "--base-predictions", str(MACHINE / "validation_and_predictions/main_base_k0_per_image_predictions.csv"),
             "--selected", str(MACHINE / "validation_and_predictions/experiment_b_validation_selected.csv"),
             "--components", str(MACHINE / "group_clustered_inference/test_visual_component_map.csv"),
             "--metadata", str(MACHINE / "inputs/main_benchmark/annotations/metadata.jsonl"),
             "--output-dir", str(exp2_out)], CODE)
        exp3_source = RESULTS / "experiment_3_validation_size/server_runs/merged"
        run([sys.executable, "-B", str(exp3), "--checkpoint-dir", str(exp3_source), "--output-dir", str(exp3_out), "--replicates", "10000"], CODE)

        exp4 = build_exp4()
        exp5, as_refs = build_exp5()
        exp4.to_csv(work_root / "experiment_4_prompt_baselines.csv", index=False)
        exp5.to_csv(work_root / "experiment_5_matched_interface_curves.csv", index=False)
        as_refs.to_csv(work_root / "experiment_5_attribute_reference.csv", index=False)

        expected_root = MACHINE / "v13_analysis"
        check_close(pd.read_csv(exp1_out / "adjacent_results.csv"),
                    pd.read_csv(expected_root / "experiment_1_aircraft_adjacent/adjacent_results.csv"),
                    ["backbone", "ranker", "transition"], ["delta_pp", "ci95_low_pp", "ci95_high_pp", "holm_p_16"], "Experiment 1")

        corrected_exp2 = MACHINE / "recomputed_from_raw/experiment_2_class_concentration"
        if not (corrected_exp2 / "leave_one_class_out.csv").is_file():
            raise RuntimeError("Corrected canonical-macro Experiment 2 outputs are missing")
        check_close(pd.read_csv(exp2_out / "leave_one_class_out.csv"),
                    pd.read_csv(corrected_exp2 / "leave_one_class_out.csv"),
                    ["backbone", "ranker", "excluded_class_id"], ["delta_pp", "ci95_low_pp", "ci95_high_pp"], "Experiment 2 corrected macro-cluster LOCO")

        exp3_expected = pd.read_csv(expected_root / "experiment_3_validation_size/server_runs/exp3_validation_size/regret_by_validation_size.csv")
        exp3_recomputed = pd.read_csv(exp3_out / "regret_by_validation_size.csv")
        check_close(exp3_recomputed, exp3_expected,
                    ["backbone", "ranker", "n_validation_per_class"], ["mean_regret", "oracle_regret_ci95_low", "oracle_regret_ci95_high", "mean_selected_test_accuracy"], "Experiment 3")
        check_close(exp4, pd.read_csv(expected_root / "experiment_4_prompt_baselines/prompt_baselines.csv"),
                    ["domain", "backbone", "ranker", "prompt_condition"],
                    ["macro_accuracy", "top1_accuracy_secondary", "macro_f1"], "Experiment 4")
        check_close(exp5, pd.read_csv(expected_root / "experiment_5_matched_prior/matched_interface_curves.csv"),
                    ["domain", "backbone", "seed", "ranker", "k", "interface"], ["test_macro_accuracy"], "Experiment 5")
        stored_as = pd.concat([
            pd.read_csv(RESULTS / "experiment_5_matched_prior" / "attributeselect_mean.csv"),
            pd.read_csv(RESULTS / "experiment_5_matched_prior" / "attributeselect_probe.csv"),
        ], ignore_index=True).rename(columns={"method": "reference"})
        check_close(as_refs, stored_as, ["backbone", "seed", "k", "reference"], ["test_macro_accuracy"], "Experiment 5 AttributeSelect-style reference")
        if args.output_dir:
            out_root = args.output_dir.resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(exp1_out, out_root / "experiment_1", dirs_exist_ok=True)
            shutil.copytree(exp2_out, out_root / "experiment_2", dirs_exist_ok=True)
            shutil.copytree(exp3_out, out_root / "experiment_3", dirs_exist_ok=True)
            shutil.copy2(work_root / "experiment_4_prompt_baselines.csv", out_root / "experiment_4_prompt_baselines.csv")
            shutil.copy2(work_root / "experiment_5_matched_interface_curves.csv", out_root / "experiment_5_matched_interface_curves.csv")
            shutil.copy2(work_root / "experiment_5_attribute_reference.csv", out_root / "experiment_5_attribute_reference.csv")
            report = {"status": "PASS", "recomputed_from": "raw per-image predictions and complete Experiment 3 shards",
                      "model_inference": "none", "images_opened": False,
                      "outputs": ["experiment_1/", "experiment_2/", "experiment_3/", "experiment_4_prompt_baselines.csv", "experiment_5_matched_interface_curves.csv", "experiment_5_attribute_reference.csv"]}
            (out_root / "raw_replay_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("PASS: raw replay complete; all five reviewer-driven analysis summaries match the current authoritative sources.")
    finally:
        temp_context.cleanup()


if __name__ == "__main__":
    main()
