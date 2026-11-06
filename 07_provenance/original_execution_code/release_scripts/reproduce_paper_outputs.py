"""Validate every numerical paper claim and optionally replay figures to a safe output folder."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "machine_readable"
AN = MR / "v13_analysis"


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def close(actual: float, expected: float, label: str, tol: float = 1e-5) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: got {actual}, expected {expected} (tol={tol})")


def csv(path: Path, columns: tuple[str, ...] = (), rows: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path.relative_to(ROOT))
    frame = pd.read_csv(path)
    absent = [column for column in columns if column not in frame.columns]
    require(not absent, f"{path.name} missing columns: {absent}")
    if rows is not None:
        require(len(frame) == rows, f"{path.name}: expected {rows} rows, got {len(frame)}")
    return frame


def validate_claims() -> None:
    claim_path = MR / "paper_claims.json"
    claims = json.loads(claim_path.read_text(encoding="utf-8"))
    require(claims.get("version") == "v14", "claim ledger version is not v14")
    basis = claims.get("basis", "").lower()
    require("five reviewer-driven analyses" in basis and
            "documented post-review plan" in basis and
            "separate post hoc base-matched aircraft addendum" in basis and
            "consolidated record rather than a preregistration" in basis,
            "claim-ledger basis does not distinguish the original five analyses from the separate addendum")

    class_root = AN / "experiment_2_class_concentration"
    classwise = csv(class_root / "classwise_changes.csv",
                    ("backbone", "ranker", "class_name", "delta_correct"), 40)
    loco = csv(class_root / "leave_one_class_out.csv",
               ("backbone", "ranker", "excluded_class_name", "delta_pp"), 40)
    require(len(claims["development_class_concentration"]) == 4,
            "development class-concentration claim count must be four")
    for record in claims["development_class_concentration"]:
        group = classwise[(classwise.backbone == record["backbone"]) &
                          (classwise.ranker == record["ranker"])]
        require(len(group) == 10, "class-concentration condition must contain ten classes")
        delta = group.delta_correct.astype(int)
        close(delta.sum(), record["net_correct_gain"], "net correct gain")
        close((delta > 0).sum(), record["improved_classes"], "improved classes")
        close((delta < 0).sum(), record["worse_classes"], "worse classes")
        close((delta == 0).sum(), record["unchanged_classes"], "unchanged classes")
        drug = group[group.class_name.str.casefold() == "drugstore beetle"]
        require(len(drug) == 1, "drugstore class row missing or duplicated")
        close(int(drug.delta_correct.iloc[0]), record["drugstore_net_correct_gain"], "drugstore net gain")
        gains = delta.clip(lower=0)
        share = 100 * float(drug.delta_correct.clip(lower=0).iloc[0]) / float(gains.sum())
        close(share, record["drugstore_positive_gain_share_pct"], "drugstore positive-gain share")
        d = loco[(loco.backbone == record["backbone"]) & (loco.ranker == record["ranker"]) &
                 (loco.excluded_class_name.str.casefold() == "drugstore beetle")]
        require(len(d) == 1, "drugstore-excluded LOCO row missing or duplicated")
        close(d.delta_pp.iloc[0], record["gain_after_excluding_drugstore_pp"], "drugstore-excluded gain")

    adj_path = AN / "experiment_1_aircraft_adjacent" / "adjacent_results.csv"
    adj = csv(adj_path, ("backbone", "ranker", "transition", "delta_pp", "ci95_low_pp",
                         "ci95_high_pp", "holm_p_16"), 16)
    ac = claims["aircraft_adjacent"]
    supported = adj[adj.holm_p_16 < 0.05]
    require(len(adj) == ac["planned_contrasts"], "Aircraft adjacent contrast count mismatch")
    require(len(supported) == ac["holm_supported_contrasts"], "Holm-supported contrast count mismatch")
    require(int(((adj.holm_p_16 < 0.05) & (adj.delta_pp < 0)).sum()) ==
            ac["holm_supported_negative_contrasts"], "Holm-supported negative count mismatch")
    one = ac["only_supported_contrast"]
    match = supported[(supported.backbone == one["backbone"]) &
                      (supported.ranker == one["ranker"]) &
                      (supported.transition == one["transition"])]
    require(len(match) == 1, "the sole Holm-supported Aircraft contrast does not match the claim ledger")
    row = match.iloc[0]
    for column, key in (("delta_pp", "delta_pp"), ("ci95_low_pp", "ci95_pp"),
                        ("ci95_high_pp", "ci95_pp"), ("holm_p_16", "holm_p")):
        if key == "ci95_pp":
            pos = 0 if column == "ci95_low_pp" else 1
            close(row[column], one[key][pos], f"Aircraft {column}")
        else:
            close(row[column], one[key], f"Aircraft {column}")

    exp3 = AN / "experiment_3_validation_size" / "server_runs"
    exp3_leaf = exp3 / "exp3_validation_size"
    regret = csv(exp3_leaf / "regret_by_validation_size.csv",
                 ("backbone", "ranker", "n_validation_per_class", "mean_regret",
                  "oracle_regret_ci95_low", "oracle_regret_ci95_high"), 24)
    require(len(claims["aircraft_validation_regret"]) == 4,
            "regret claim ledger must contain four backbone/ranker conditions")
    for key, values in claims["aircraft_validation_regret"].items():
        backbone, ranker = key.split("|")
        group = regret[(regret.backbone == backbone) & (regret.ranker == ranker)].copy()
        group["nv"] = group.n_validation_per_class.astype(str)
        group = group.set_index("nv")
        for nv, target in (("5", values["nV5_pp"]), ("all", values["allV_pp"])):
            close(100 * group.loc[nv, "mean_regret"], target, f"{key} mean regret nV={nv}")
        intervals = values.get("mean_regret_ci95_pp", {})
        require(set(intervals) == {"1", "2", "5", "10", "20", "all"},
                f"{key} must ledger the mean-regret CI at all six validation sizes")
        for nv, pair in intervals.items():
            close(100 * group.loc[nv, "oracle_regret_ci95_low"], pair[0], f"{key} nV={nv} CI low")
            close(100 * group.loc[nv, "oracle_regret_ci95_high"], pair[1], f"{key} nV={nv} CI high")

    support_file = exp3_leaf / "full_validation_support_selections.csv"
    support = csv(support_file, ("support_seed", "selected_k", "test_oracle_k",
                                 "permutation_records_with_all_validation",
                                 "distinct_selected_k_values"), 20)
    require(support.support_seed.nunique() == 20, "full-validation support rows are not unique")
    require(support.permutation_records_with_all_validation.eq(50).all(),
            "expected 50 nested permutation records per support resample")
    require(support.distinct_selected_k_values.eq(1).all(),
            "complete-validation selection must be invariant within support resample")
    require((support.backbone == "ViT-B-32").all() and
            (support.ranker == "positive_mean").all(), "support selection summary condition mismatch")
    b32 = claims["aircraft_b32_pm"]
    close(int((support.selected_k == 3).sum()), b32["full_validation_k3_support_resamples"],
          "full-validation k=3 support frequency")
    close(len(support), b32["full_validation_support_resamples"], "full-validation support count")
    close(int((support.test_oracle_k == 1).sum()), b32["test_oracle_k1_support_resamples"],
          "test-oracle k=1 support frequency")
    close(int(support.permutation_records_with_all_validation.iloc[0]),
          b32["validation_permutation_records_per_support_resample"], "nested permutation count")
    require(b32["full_validation_permutation_records_are_duplicate_within_support"] is True,
            "full-validation duplicate-record qualification is missing")
    ranking = csv(exp3_leaf / "ranking_stability.csv",
                  ("backbone", "ranker", "top1_modal_agreement"), 400)
    top1 = ranking[(ranking.backbone == "ViT-B-32") & (ranking.ranker == "positive_mean")]
    close(100 * top1.top1_modal_agreement.mean(),
          b32["top1_modal_ranking_agreement_pct"], "top-1 modal ranking agreement")
    test_curves = csv(exp3 / "merged" / "test_curves.csv",
                      ("backbone", "ranker", "support_seed", "k", "test_macro_accuracy"), 400)
    oracle = test_curves[(test_curves.backbone == "ViT-B-32") &
                         (test_curves.ranker == "positive_mean")].pivot(
                             index="support_seed", columns="k", values="test_macro_accuracy").idxmax(axis=1)
    for r in support.itertuples(index=False):
        close(oracle.loc[r.support_seed], r.test_oracle_k, "support-level test-oracle k")

    prompt = csv(AN / "experiment_4_prompt_baselines" / "classwise_prompt_baselines.csv",
                 ("domain", "backbone", "ranker", "prompt_condition", "class_id", "recall"), 6560)
    prompt_summary = prompt.groupby(["domain", "backbone", "ranker", "prompt_condition"]).recall.mean()
    prompt_map = {"|".join(map(str, key)): float(value) for key, value in prompt_summary.items()}
    prompt_metrics = csv(AN / "experiment_4_prompt_baselines" / "prompt_baselines.csv",
                        ("domain", "backbone", "ranker", "prompt_condition", "macro_accuracy",
                         "top1_accuracy_secondary", "macro_f1"), 32)
    metric_map = {"|".join(map(str, (r.domain, r.backbone, r.ranker, r.prompt_condition))): r
                  for r in prompt_metrics.itertuples(index=False)}
    claimed_prompt = claims["prompt_macro_accuracy"]
    require(set(prompt_map) == set(claimed_prompt), "prompt-baseline claim keys do not match the CSV")
    require(set(metric_map) == set(claimed_prompt), "prompt-baseline metric rows do not match the claim keys")
    for key, value in prompt_map.items():
        close(value, claimed_prompt[key], f"prompt baseline {key}", tol=1e-7)
        close(metric_map[key].macro_accuracy, value, f"prompt primary macro accuracy {key}", tol=1e-7)
        require(0 <= float(metric_map[key].top1_accuracy_secondary) <= 1,
                f"secondary top-1 accuracy is out of range for {key}")

    source_only = csv(MR / "source_only_exploratory_intervals" /
                      "source_only_exploratory_paired_intervals.csv",
                      ("backbone", "selector", "comparison", "accuracy_difference_candidate_minus_comparator",
                       "component_clustered_bootstrap_ci95_low", "component_clustered_bootstrap_ci95_high"), 8)
    source_claims = claims["source_only_exploratory_ci_pp"]
    require(len(source_claims) == 4, "source-only exploratory compact-versus-full CI claim count must be four")
    for key, target in source_claims.items():
        backbone, selector = key.split("|")
        row = source_only[(source_only.backbone == backbone) & (source_only.selector == selector) &
                          (source_only.comparison == target["contrast"])]
        require(len(row) == 1, f"source-only CI row is missing or duplicated: {key}")
        observed = row.iloc[0]
        close(round(100 * observed.accuracy_difference_candidate_minus_comparator, 2),
              target["delta_pp"], f"source-only {key} difference")
        close(round(100 * observed.component_clustered_bootstrap_ci95_low, 2),
              target["ci95_pp"][0], f"source-only {key} CI low")
        close(round(100 * observed.component_clustered_bootstrap_ci95_high, 2),
              target["ci95_pp"][1], f"source-only {key} CI high")

    matched = csv(AN / "experiment_5_matched_prior" / "matched_interface_curves.csv",
                  ("domain", "backbone", "ranker", "k", "interface", "test_macro_accuracy"), 144)
    matched = matched[(matched.domain == "aircraft") & (matched.ranker == "positive_mean")]
    summary = matched.groupby(["backbone", "interface", "k"]).test_macro_accuracy.agg(["mean", "std", "count"])
    require(len(claims["matched_interface_mean_pct"]) == 24, "matched interface claim count mismatch")
    for r in claims["matched_interface_mean_pct"]:
        key = (r["backbone"], r["interface"], r["k"])
        got = summary.loc[key]
        close(100 * got["mean"], r["mean_pct"], f"matched mean {key}")
        close(100 * got["std"], r["seed_sd_pp"], f"matched seed SD {key}")

    centroid = csv(MR / "support_centroid_reference.csv",
                   ("backbone", "split", "accuracy"))
    centroid = centroid[centroid.split == "test"]
    observed_centroid = {r.backbone: 100 * float(r.accuracy) for r in centroid.itertuples()}
    require(set(observed_centroid) == set(claims["centroid_test_top1_accuracy_pct"]),
            "centroid claim keys mismatch")
    for key, value in observed_centroid.items():
        close(value, claims["centroid_test_top1_accuracy_pct"][key], f"centroid {key}")
    print("PASS: all numerical fields in paper_claims.json match packaged source CSVs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true",
                        help="validate numerical claims without generating any files")
    parser.add_argument("--out-dir", type=Path,
                        help="new figure-output folder outside the package; never overwrites managed files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_claims()
    if args.validate_only:
        return
    if args.out_dir is None:
        out = Path(tempfile.mkdtemp(prefix="paper_replay_"))
    else:
        out = args.out_dir.expanduser().resolve()
        if out == ROOT or ROOT in out.parents:
            raise ValueError("Replay output must be outside the hash-managed package")
        out.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, str(ROOT / "scripts" / "make_paper_figures.py"),
        "--analysis-root", str(AN),
        "--source-only-root", str(MR / "source_only_resource"),
        "--aircraft-curves", str(MR / "budget_curves.csv"),
        "--out-dir", str(out),
    ]
    subprocess.run(command, check=True)
    (out / "paper_claims_replay.json").write_text(
        json.dumps({"claims_validated": True, "figure_files": sorted(p.name for p in out.iterdir())},
                   indent=2) + "\n", encoding="utf-8")
    print(f"PASS: figures regenerated in isolated output folder: {out}")


if __name__ == "__main__":
    main()
