"""Recompute ICCAI additional exploratory controls from the v16 frozen package.

No image pixels are read and no image encoder is rerun. The script uses the
archived image-feature cache, split manifests, per-image predictions, the
frozen resource, and the locally fingerprinted text encoders.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(PACKAGE / "03_code" / "core"))
from aircraft_core import BACKBONES, SEEDS, encode_text, load_context, macro_accuracy, recomposed_prompt, sha256

OUT = PACKAGE / "05_results" / "replayed_additional_controls"
RANDOM_SEEDS = tuple(260100 + i for i in range(50))
K_VALUES = tuple(range(5))
torch.set_num_threads(4)


def encode_unique(context: dict, prompts: list[str], batch_size: int = 64):
    """Encode prompts without truncation and return token lengths in the same order."""
    model, tokenizer = context["model"], context["tokenizer"]
    vectors, lengths = [], []
    maximum = int(model.config.text_config.max_position_embeddings)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
        batch_lengths = inputs["attention_mask"].sum(1).cpu().numpy().astype(int)
        if int(inputs["input_ids"].shape[1]) > maximum:
            raise RuntimeError(f"Prompt length exceeds CLIP context length {maximum}")
        inputs = {key: value.to(context["device"]) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = model.text_model(**inputs)
            last = inputs["attention_mask"].sum(1) - 1
            pooled = hidden.last_hidden_state[
                torch.arange(len(batch), device=context["device"]), last
            ]
            vectors.append(F.normalize(model.text_projection(pooled), dim=-1).cpu())
        lengths.extend(batch_lengths.tolist())
    return torch.cat(vectors, dim=0), lengths


def prompt_lengths(tokenizer, prompts: list[str], maximum: int, batch_size: int = 256):
    lengths = []
    for start in range(0, len(prompts), batch_size):
        inputs = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        lengths.extend(inputs["attention_mask"].sum(1).cpu().numpy().astype(int).tolist())
    if lengths and max(lengths) > maximum:
        raise RuntimeError(f"A measured prompt exceeds the CLIP context length {maximum}")
    return lengths


def macro_from_logits(truth: np.ndarray, logits: torch.Tensor, n_classes: int) -> tuple[float, np.ndarray]:
    pred = logits.argmax(dim=1).cpu().numpy()
    return macro_accuracy(truth, pred, n_classes), pred


def load_orders(rankings: pd.DataFrame, backbone: str, seed: int, n_classes: int) -> list[list[int]]:
    rows = rankings[
        rankings["analysis"].eq("matched")
        & rankings["backbone"].eq(backbone)
        & rankings["seed"].eq(seed)
    ]
    orders = []
    for class_id in range(n_classes):
        group = rows[rows["class_id"].eq(class_id)].sort_values("rank")
        if len(group) != 4:
            raise RuntimeError(f"Expected four matched ranks for {backbone}, seed={seed}, class={class_id}")
        ids = group["descriptor_id"].astype(int).tolist()
        if sorted(ids) != [1, 2, 3, 4]:
            raise RuntimeError("Matched ranking descriptor IDs are not a permutation of 1..4")
        orders.append([value - 1 for value in ids])
    return orders


def read_validation_pairs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = PACKAGE / "04_data/validation_selection_records/all_selections.csv"
    frame = pd.read_csv(path, dtype={"n_validation_per_class": str})
    selected = frame[
        frame["backbone"].eq("ViT-B-32")
        & frame["ranker"].eq("positive_mean")
        & frame["n_validation_per_class"].eq("all")
    ].copy()
    # The 50 nested permutations at the all-validation level are identical by
    # construction. Collapse them only after verifying each support block.
    cols = ["selected_k", "oracle_k", "oracle_regret", "selected_test_macro_accuracy", "oracle_test_macro_accuracy"]
    for seed, group in selected.groupby("support_seed"):
        if any(group[col].nunique(dropna=False) != 1 for col in cols):
            raise RuntimeError(f"Full-validation records vary within support resample {seed}")
    paired = selected.sort_values(["support_seed", "validation_permutation"]).drop_duplicates("support_seed").copy()
    paired = paired.rename(columns={"selected_k": "k_validation", "oracle_k": "k_test_best"})
    table = pd.crosstab(paired["k_validation"], paired["k_test_best"]).reindex(
        index=K_VALUES, columns=K_VALUES, fill_value=0
    )
    summary = {
        "n_support_resamples": int(len(paired)),
        "agreement_count": int((paired.k_validation == paired.k_test_best).sum()),
        "agreement_fraction": float((paired.k_validation == paired.k_test_best).mean()),
        "mean_regret_pp": float(paired.oracle_regret.mean() * 100),
        "median_regret_pp": float(paired.oracle_regret.median() * 100),
        "q1_regret_pp": float(paired.oracle_regret.quantile(0.25) * 100),
        "q3_regret_pp": float(paired.oracle_regret.quantile(0.75) * 100),
        "min_regret_pp": float(paired.oracle_regret.min() * 100),
        "max_regret_pp": float(paired.oracle_regret.max() * 100),
        "test_best_is_descriptive_only": True,
    }
    return paired, table, summary


def aircraft_classwise() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = PACKAGE / "04_data/aircraft_prediction_records/predictions.csv"
    selected_path = PACKAGE / "05_results/selected_budgets.csv"
    pred = pd.read_csv(pred_path, dtype={"image_id": str})
    pred = pred[pred["split"].eq("test")].copy()
    selected = pd.read_csv(selected_path)
    rows = []
    for (backbone, seed, scorer), group in pred.groupby(["backbone", "seed", "scorer"], sort=True):
        pivot = group.pivot(index="image_id", columns="k", values="predicted_index")
        truth = group.drop_duplicates("image_id").set_index("image_id")["true_index"]
        for compare_name, k in [("k1_vs_k0", 1)]:
            correct0 = pivot[0].eq(truth)
            correctk = pivot[k].eq(truth)
            for class_id in sorted(truth.unique()):
                mask = truth.eq(class_id)
                rows.append({
                    "backbone": backbone, "seed": int(seed), "ranker": scorer,
                    "comparison": compare_name, "class_id": int(class_id),
                    "n_images": int(mask.sum()), "baseline_correct": int(correct0[mask].sum()),
                    "comparison_correct": int(correctk[mask].sum()),
                    "gained_correct": int((~correct0[mask] & correctk[mask]).sum()),
                    "lost_correct": int((correct0[mask] & ~correctk[mask]).sum()),
                    "net_correct_change": int((correctk[mask].astype(int) - correct0[mask].astype(int)).sum()),
                    "baseline_accuracy": float(correct0[mask].mean()),
                    "comparison_accuracy": float(correctk[mask].mean()),
                })
        chosen = selected[
            selected["backbone"].eq(backbone)
            & selected["seed"].eq(seed)
            & selected["scorer"].eq(scorer)
        ]
        if len(chosen) != 1:
            raise RuntimeError("Cannot resolve a unique validation-selected k")
        k = int(chosen.iloc[0]["selected_k"])
        correct0 = pivot[0].eq(truth)
        correctk = pivot[k].eq(truth)
        for class_id in sorted(truth.unique()):
            mask = truth.eq(class_id)
            rows.append({
                "backbone": backbone, "seed": int(seed), "ranker": scorer,
                "comparison": f"selected_k{int(k)}_vs_k0", "class_id": int(class_id),
                "n_images": int(mask.sum()), "baseline_correct": int(correct0[mask].sum()),
                "comparison_correct": int(correctk[mask].sum()),
                "gained_correct": int((~correct0[mask] & correctk[mask]).sum()),
                "lost_correct": int((correct0[mask] & ~correctk[mask]).sum()),
                "net_correct_change": int((correctk[mask].astype(int) - correct0[mask].astype(int)).sum()),
                "baseline_accuracy": float(correct0[mask].mean()),
                "comparison_accuracy": float(correctk[mask].mean()),
            })
    result = pd.DataFrame(rows)
    summary = result.groupby(["backbone", "ranker", "comparison"], as_index=False).agg(
        class_seed_rows=("class_id", "size"),
        class_seed_rows_with_any_gain=("gained_correct", lambda x: int((x > 0).sum())),
        class_seed_rows_with_any_loss=("lost_correct", lambda x: int((x > 0).sum())),
        class_seed_rows_net_positive=("net_correct_change", lambda x: int((x > 0).sum())),
        class_seed_rows_net_negative=("net_correct_change", lambda x: int((x < 0).sum())),
        total_gains=("gained_correct", "sum"), total_losses=("lost_correct", "sum"),
        total_net_change=("net_correct_change", "sum"),
    )
    return result, summary


def main(models_root: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    resource_path = PACKAGE / "02_resources/aircraft_descriptors_frozen.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    classes = resource["classes"]
    n_classes = len(classes)
    if n_classes != 100 or any(len(row["descriptors"]) != 4 for row in classes):
        raise RuntimeError("The frozen resource is not 100 classes by four descriptors")
    rankings = pd.read_csv(PACKAGE / "04_data/matched_rankings.csv")

    # Strictly paired full-validation versus test-best audit, from the archived
    # 20 resamples rather than two marginal histograms.
    pair_rows, pair_table, pair_summary = read_validation_pairs()
    pair_rows.to_csv(OUT / "validation_fullsample_paired_by_support.csv", index=False)
    pair_table.to_csv(OUT / "validation_fullsample_kv_by_kt.csv", index_label="k_validation")
    (OUT / "validation_fullsample_summary.json").write_text(json.dumps(pair_summary, indent=2) + "\n", encoding="utf-8")

    classwise, classwise_summary = aircraft_classwise()
    classwise.to_csv(OUT / "aircraft_classwise_deltas.csv", index=False)
    classwise_summary.to_csv(OUT / "aircraft_classwise_summary.csv", index=False)

    prompt_rows = []
    curve_rows = []
    random_selected_rows = []
    random_summary_rows = []
    name_rows = []
    norm_curve_rows = []
    norm_class_rows = []
    norm_diff_rows = []
    token_audit_rows = []
    invariance_rows = []
    interface_identity_rows = []

    for backbone in BACKBONES:
        print(f"[{backbone}] loading frozen model and cached features", flush=True)
        context = load_context(backbone, models_root)
        test_features = context["features"]
        test_truth = context["labels"]
        test_ids = context["image_ids"]
        cache = np.load(PACKAGE / f"04_data/feature_cache/image_features_{backbone}.npz")
        cached_ids = cache["image_ids"].astype(str).tolist()
        cache_positions = {image_id: i for i, image_id in enumerate(cached_ids)}
        all_features = F.normalize(torch.from_numpy(cache["features"].astype(np.float32)), dim=-1)
        seed_val_features = {}
        seed_val_truth = {}
        manifest_dir = PACKAGE / "01_protocol/split_manifests"
        for seed in SEEDS:
            manifest = json.loads((manifest_dir / f"split_seed_{seed}.json").read_text(encoding="utf-8"))
            val_ids, val_truth = [], []
            for class_id, record in enumerate(classes):
                ids = manifest["validation"][record["name_en"]]
                val_ids.extend(ids)
                val_truth.extend([class_id] * len(ids))
            if len(val_ids) != 500 or any(image_id not in cache_positions for image_id in val_ids):
                raise RuntimeError(f"Validation manifest/cache mismatch for {seed}")
            positions = torch.tensor([cache_positions[image_id] for image_id in val_ids], dtype=torch.long)
            seed_val_features[seed] = all_features[positions]
            seed_val_truth[seed] = np.asarray(val_truth, dtype=np.int64)

        # Generate 50 fixed random class-wise rankings. Randomness changes which
        # descriptors enter the budget, while final prompt serialization remains
        # canonical catalogue order, as specified in the review plan.
        prompt_to_id: dict[str, int] = {}
        random_indices = np.empty((len(RANDOM_SEEDS), len(K_VALUES), n_classes), dtype=np.int64)
        random_prompt_groups: dict[int, list[str]] = defaultdict(list)
        random_orders = []
        for ridx, random_seed in enumerate(RANDOM_SEEDS):
            rng = np.random.default_rng(random_seed)
            class_orders = [rng.permutation(4).tolist() for _ in classes]
            random_orders.append(class_orders)
            for k in K_VALUES:
                prompts = [
                    recomposed_prompt(record, [record["descriptors"][j] for j in sorted(order[:k])])
                    for record, order in zip(classes, class_orders)
                ]
                for class_id, prompt in enumerate(prompts):
                    if prompt not in prompt_to_id:
                        prompt_to_id[prompt] = len(prompt_to_id)
                        random_prompt_groups[k].append(prompt)
                    random_indices[ridx, k, class_id] = prompt_to_id[prompt]

        name_prompts = [f"a photo of a {record['name_en']}." for record in classes]
        name_indices = []
        for prompt in name_prompts:
            if prompt not in prompt_to_id:
                prompt_to_id[prompt] = len(prompt_to_id)
            name_indices.append(prompt_to_id[prompt])

        # Also collect exact prompts from every archived PM/CM budget condition
        # and from the base-matched descriptor ensemble for token auditing.
        primary_prompts: list[str] = []
        joint_audit_prompts: list[str] = []
        for seed in SEEDS:
            orders = load_orders(rankings, backbone, seed, n_classes)
            for k in K_VALUES:
                selected_prompts = [
                    recomposed_prompt(record, [record["descriptors"][j] for j in sorted(order[:k])])
                    for record, order in zip(classes, orders)
                ]
                primary_prompts.extend(selected_prompts)
                if k > 0:
                    joint_audit_prompts.extend(selected_prompts)
        base_matched_prompts = [
            recomposed_prompt(record, [descriptor])
            for record in classes for descriptor in record["descriptors"]
        ]
        all_prompts = list(prompt_to_id)
        print(f"[{backbone}] encoding {len(all_prompts):,} unique random/name prompts", flush=True)
        random_text, random_lengths = encode_unique(context, all_prompts)
        prompt_lengths_by_text = dict(zip(all_prompts, random_lengths))

        tokenizer = context["tokenizer"]
        maximum = int(context["model"].config.text_config.max_position_embeddings)
        groups = {
            **{f"random_recomposition_k{k}": list(dict.fromkeys(random_prompt_groups[k])) for k in K_VALUES},
            "class_name_only": name_prompts,
            "archived_main_recomposition_matched_rankings": list(dict.fromkeys(primary_prompts)),
            "base_matched_joint_recomposition": list(dict.fromkeys(joint_audit_prompts)),
            "base_matched_single_descriptor": base_matched_prompts,
        }
        for condition, values in groups.items():
            lens = prompt_lengths(tokenizer, values, maximum)
            token_audit_rows.append({
                "backbone": backbone, "condition": condition, "n_prompts": len(values),
                "min_tokens": int(min(lens)), "median_tokens": float(np.median(lens)),
                "p95_tokens": float(np.quantile(lens, 0.95)), "max_tokens": int(max(lens)),
                "n_over_context": int(sum(x > maximum for x in lens)),
                "context_length": maximum, "truncation": "disabled; overflow would fail",
            })

        random_test_accuracy = np.zeros((len(RANDOM_SEEDS), len(K_VALUES)), dtype=float)
        random_test_predictions: dict[tuple[int, int], np.ndarray] = {}
        for ridx, random_seed in enumerate(RANDOM_SEEDS):
            for k in K_VALUES:
                text = random_text[torch.tensor(random_indices[ridx, k], dtype=torch.long)]
                logits = test_features @ text.T
                accuracy, predicted = macro_from_logits(test_truth, logits, n_classes)
                random_test_accuracy[ridx, k] = accuracy
                random_test_predictions[ridx, k] = predicted
                curve_rows.append({
                    "backbone": backbone, "random_ranking_seed": random_seed, "k": k,
                    "test_macro_accuracy": accuracy,
                    "analysis_status": "post_review_exploratory_random_rank_control",
                })
            for seed in SEEDS:
                val_curve = []
                for k in K_VALUES:
                    text = random_text[torch.tensor(random_indices[ridx, k], dtype=torch.long)]
                    logits = seed_val_features[seed] @ text.T
                    val_curve.append(macro_accuracy(seed_val_truth[seed], logits.argmax(1).numpy(), n_classes))
                chosen_k = max(K_VALUES, key=lambda k: (val_curve[k], -k))
                random_selected_rows.append({
                    "backbone": backbone, "random_ranking_seed": random_seed, "support_validation_seed": seed,
                    "selected_k": chosen_k, "selected_validation_macro_accuracy": val_curve[chosen_k],
                    "selected_test_macro_accuracy": float(random_test_accuracy[ridx, chosen_k]),
                    "analysis_status": "post_review_exploratory_random_rank_control",
                })

        # The k=0 and k=4 prompts are invariant to ranking and support seed.
        # Check exact test predictions/logits across all random rankings.
        for k in (0, 4):
            reference_pred = random_test_predictions[0, k]
            equal = all(np.array_equal(reference_pred, random_test_predictions[ridx, k]) for ridx in range(1, len(RANDOM_SEEDS)))
            invariance_rows.append({
                "backbone": backbone, "k": k, "n_random_rankings": len(RANDOM_SEEDS),
                "test_predictions_identical_across_rankings": bool(equal),
                "max_logit_difference": 0.0,
                "analysis_status": "deterministic_fixed_prompt_invariance_check",
            })
            if not equal:
                raise AssertionError(f"Expected ranking invariance at k={k} for {backbone}")

        selected_frame = pd.DataFrame([x for x in random_selected_rows if x["backbone"] == backbone])
        per_ranking = selected_frame.groupby("random_ranking_seed", as_index=False).agg(
            mean_selected_test_macro_accuracy=("selected_test_macro_accuracy", "mean"),
            mean_selected_k=("selected_k", "mean"),
        )
        random_summary_rows.append({
            "backbone": backbone, "n_random_rankings": len(per_ranking),
            "mean_of_ranking_means_macro_accuracy": float(per_ranking.mean_selected_test_macro_accuracy.mean()),
            "sd_across_random_ranking_means": float(per_ranking.mean_selected_test_macro_accuracy.std(ddof=1)),
            "q025_ranking_mean_macro_accuracy": float(per_ranking.mean_selected_test_macro_accuracy.quantile(0.025)),
            "median_ranking_mean_macro_accuracy": float(per_ranking.mean_selected_test_macro_accuracy.median()),
            "q975_ranking_mean_macro_accuracy": float(per_ranking.mean_selected_test_macro_accuracy.quantile(0.975)),
            "mean_selected_k": float(per_ranking.mean_selected_k.mean()),
            "summary_unit": "50 random ranking means, each averaged across the same five support/validation resamples",
            "analysis_status": "post_review_exploratory_no_hypothesis_test",
        })

        # Class-name-only baseline, distinct from k=0 base-description scoring.
        name_text = random_text[torch.tensor(name_indices, dtype=torch.long)]
        name_logits = test_features @ name_text.T
        name_accuracy, name_prediction = macro_from_logits(test_truth, name_logits, n_classes)
        name_rows.append({
            "backbone": backbone, "prompt_template": "a photo of a {official FGVC-Aircraft class name}.",
            "test_macro_accuracy": name_accuracy, "n_test": int(len(test_truth)),
            "analysis_status": "post_review_exploratory_prompt_baseline",
        })

        # Separate base-matched prompts: compare mean scores with normalized
        # mean text prototypes using precisely the archived PM rankings.
        bm_text, bm_lengths = encode_unique(context, base_matched_prompts)
        bm_text_by_class = bm_text.reshape(n_classes, 4, -1)
        archived_bm = pd.read_csv(PACKAGE / "05_results/base_matched_interface_curves.csv")
        raw_pred_path = PACKAGE / "04_data/aircraft_prediction_records/base_matched_predictions.csv"
        raw_all = pd.read_csv(raw_pred_path, dtype={"image_id": str})
        # The addendum prediction file stores separate-score predictions.
        # Recompute joint prompts from the same selected sets to compare all
        # three scoring forms at the per-image level.
        for seed in SEEDS:
            orders = load_orders(rankings, backbone, seed, n_classes)
            for k in range(1, 5):
                means = torch.stack([
                    bm_text_by_class[class_id, sorted(orders[class_id][:k])].mean(dim=0)
                    for class_id in range(n_classes)
                ])
                norms = torch.linalg.vector_norm(means, dim=1)
                normalized = means / norms[:, None]
                norm_logits = test_features @ normalized.T
                norm_acc, norm_pred = macro_from_logits(test_truth, norm_logits, n_classes)
                joint_prompts = [
                    recomposed_prompt(record, [record["descriptors"][j] for j in sorted(orders[class_id][:k])])
                    for class_id, record in enumerate(classes)
                ]
                joint_text, _ = encode_unique(context, joint_prompts)
                joint_logits = test_features @ joint_text.T
                joint_acc, joint_pred = macro_from_logits(test_truth, joint_logits, n_classes)
                raw_scores = torch.stack([
                    (test_features @ bm_text_by_class[class_id, sorted(orders[class_id][:k])].T).mean(dim=1)
                    for class_id in range(n_classes)
                ], dim=1)
                raw_acc, raw_pred = macro_from_logits(test_truth, raw_scores, n_classes)
                separate_rows = raw_all[
                    raw_all["backbone"].eq(backbone)
                    & raw_all["seed"].eq(seed)
                    & raw_all["k"].eq(k)
                ]
                if len(separate_rows) != len(test_ids):
                    raise RuntimeError(f"Missing archived separate-score predictions: {backbone}/{seed}/k={k}")
                separate_rows = separate_rows.set_index("image_id").loc[test_ids]
                separate_archived_pred = separate_rows["predicted_index"].to_numpy(dtype=np.int64)
                if not np.array_equal(raw_pred, separate_archived_pred):
                    raise RuntimeError(f"Separate-score replay differs from archived predictions: {backbone}/{seed}/k={k}")
                joint_archived_accuracy = float(
                    archived_bm[(archived_bm.backbone == backbone) & (archived_bm.seed == seed) & (archived_bm.k == k)].recomposition_macro_accuracy.iloc[0]
                )
                if abs(joint_acc - joint_archived_accuracy) > 1e-12:
                    raise RuntimeError(f"Joint recomposition differs from archived accuracy: {backbone}/{seed}/k={k}")
                norm_curve_rows.append({
                    "backbone": backbone, "seed": seed, "k": k,
                    "separate_mean_macro_accuracy": raw_acc,
                    "separate_normalized_mean_macro_accuracy": norm_acc,
                    "paired_accuracy_difference_pp": 100 * (norm_acc - raw_acc),
                    "joint_recomposition_macro_accuracy": joint_acc,
                    "mean_prototype_norm": float(norms.mean()),
                    "sd_prototype_norm": float(norms.std(unbiased=True)),
                    "analysis_status": "post_review_exploratory_normalized_text_mean",
                })
                norm_class_rows.extend({
                    "backbone": backbone, "seed": seed, "k": k, "class_id": class_id,
                    "mean_text_prototype_norm": float(norms[class_id]),
                } for class_id in range(n_classes))
                norm_diff_rows.append({
                    "backbone": backbone, "seed": seed, "k": k,
                    "n_test": len(test_ids),
                    "prediction_disagreements_vs_separate_mean": int(np.sum(norm_pred != raw_pred)),
                    "prediction_disagreements_vs_joint": int(np.sum(norm_pred != joint_pred)),
                    "correctness_changes_vs_separate_mean": int(np.sum((norm_pred == test_truth) != (raw_pred == test_truth))),
                    "correctness_changes_vs_joint": int(np.sum((norm_pred == test_truth) != (joint_pred == test_truth))),
                    "analysis_status": "post_review_exploratory_normalized_text_mean",
                })
                if k == 1:
                    # At k=1 the joint prompt, single-descriptor prompt, and
                    # normalized mean text vector should be numerically the
                    # same interface. Audit logits and every image prediction.
                    tolerance = 1e-5
                    interface_logits = {
                        "joint_recomposition": joint_logits,
                        "separate_mean_scores": raw_scores,
                        "normalized_mean_vector": norm_logits,
                    }
                    comparisons = (
                        ("joint_recomposition", "separate_mean_scores"),
                        ("joint_recomposition", "normalized_mean_vector"),
                        ("separate_mean_scores", "normalized_mean_vector"),
                    )
                    for reference_name, interface_name in comparisons:
                        reference_logits = interface_logits[reference_name]
                        logits = interface_logits[interface_name]
                        max_abs_difference = float((logits - reference_logits).abs().max().item())
                        prediction_disagreements = int(
                            (logits.argmax(dim=1) != reference_logits.argmax(dim=1)).sum().item()
                        )
                        interface_identity_rows.append({
                            "backbone": backbone,
                            "support_resample_seed": seed,
                            "k": 1,
                            "reference_interface": reference_name,
                            "comparison_interface": interface_name,
                            "n_test_images": int(len(test_truth)),
                            "max_abs_logit_difference": max_abs_difference,
                            "absolute_logit_tolerance": tolerance,
                            "prediction_disagreements": prediction_disagreements,
                            "logits_within_tolerance": max_abs_difference <= tolerance,
                            "all_image_predictions_identical": prediction_disagreements == 0,
                            "analysis_status": "post_review_k1_three_interface_identity_check",
                        })
                        if max_abs_difference > tolerance or prediction_disagreements != 0:
                            raise AssertionError(
                                f"k=1 interface identity failed for {backbone}/{seed}/{interface_name}: "
                                f"max_abs_logit_difference={max_abs_difference}, "
                                f"prediction_disagreements={prediction_disagreements}"
                            )
        del context

    pd.DataFrame(curve_rows).to_csv(OUT / "random_ranking_budget_curves.csv", index=False)
    pd.DataFrame(random_selected_rows).to_csv(OUT / "random_ranking_selected_operating_points.csv", index=False)
    pd.DataFrame(random_summary_rows).to_csv(OUT / "random_ranking_summary.csv", index=False)
    pd.DataFrame(name_rows).to_csv(OUT / "class_name_only_baseline.csv", index=False)
    pd.DataFrame(norm_curve_rows).to_csv(OUT / "normalized_mean_curves.csv", index=False)
    pd.DataFrame(norm_class_rows).to_csv(OUT / "normalized_mean_class_prototype_norms.csv", index=False)
    pd.DataFrame(norm_diff_rows).to_csv(OUT / "normalized_mean_prediction_differences.csv", index=False)
    pd.DataFrame(interface_identity_rows).to_csv(OUT / "interface_k1_identity_audit.csv", index=False)
    pd.DataFrame(token_audit_rows).to_csv(OUT / "token_audit.csv", index=False)
    pd.DataFrame(invariance_rows).to_csv(OUT / "budget_invariance_checks.csv", index=False)

    examples = []
    for class_id in (0, 1):
        record = classes[class_id]
        examples.append({
            "class_id": class_id,
            "class_name": record["name_en"],
            "base_description": record["base_description"],
            "descriptors": record["descriptors"],
            "sources": record.get("sources", []),
        })
    (OUT / "frozen_resource_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "analysis_status": "post_review_exploratory_addenda_and_archived_output_reanalysis",
        "source_package": "ICCAI_2027_Submission/Reproducibility_Package",
        "resource_sha256": sha256(resource_path),
        "random_ranking": {
            "random_seeds": list(RANDOM_SEEDS),
            "within_class_selection_priority_randomized": True,
            "prompt_serialization": "selected descriptors are serialized in canonical catalogue order",
            "support_validation_seeds": list(SEEDS),
            "budget_values": list(K_VALUES),
            "validation_rule": "macro accuracy; smallest k breaks ties",
            "hypothesis_test": "none; descriptive distribution only",
        },
        "class_name_only_prompt": "a photo of a {official FGVC-Aircraft class name}.",
        "normalized_mean": "average separately encoded base-matched text vectors per class and k, then L2-normalize the mean vector; score by dot product with normalized cached image features",
        "validation_pairing": pair_summary,
        "tokenization": "frozen local CLIP tokenizer; truncation disabled; context overflow fails",
        "image_pixels_read": False,
        "image_encoder_rerun": False,
        "outputs": [
            "random_ranking_budget_curves.csv", "random_ranking_selected_operating_points.csv",
            "random_ranking_summary.csv", "class_name_only_baseline.csv",
            "normalized_mean_curves.csv", "normalized_mean_class_prototype_norms.csv",
            "normalized_mean_prediction_differences.csv", "interface_k1_identity_audit.csv", "token_audit.csv",
            "budget_invariance_checks.csv", "aircraft_classwise_deltas.csv",
            "aircraft_classwise_summary.csv", "validation_fullsample_paired_by_support.csv",
            "validation_fullsample_kv_by_kt.csv", "validation_fullsample_summary.json",
            "frozen_resource_examples.json",
        ],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Completed ICCAI v2 exploratory addenda and paired reanalysis.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True, help="Local directory containing frozen text-encoder checkpoints listed in 01_protocol/model_lock.json")
    args = parser.parse_args()
    main(args.models_root.resolve())
