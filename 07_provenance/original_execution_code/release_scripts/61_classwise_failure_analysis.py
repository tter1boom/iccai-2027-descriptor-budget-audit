# -*- coding: utf-8 -*-
"""Class-wise recall and confusion audits at the validation-selected budgets.

This is a descriptive heterogeneity analysis on the same fixed 300-image Q
split. It does not create new independent replication or per-class p-values.
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "experiment_61_classwise_failure_analysis"
PREDICTIONS = ROOT / "experiments" / "results" / "experiment_47_novelty_rebuild_D" / "experiment_d_per_image_metrics.csv"
SELECTED = ROOT / "experiments" / "results" / "experiment_41_novelty_rebuild_B" / "experiment_b_validation_selected.csv"
METADATA = ROOT / "pest_vlm_dataset" / "annotations" / "metadata.jsonl"
FIG = OUT / "figures"
sys.path.insert(0, r"<redacted-local-auxiliary-path>")
from audit_panel_alignment import require_matplotlib_panel_alignment


mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.2,
                     "ytick.labelsize": 6.2, "pdf.fonttype": 42, "svg.fonttype": "none"})


def setting_label(backbone, selector):
    return f"{backbone.replace('ViT-B-', 'B/')} {'+' if selector == 'positive_mean' else 'Δ'}"


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    metadata = [json.loads(line) for line in METADATA.read_text(encoding="utf-8").splitlines() if line]
    names = {row["class_id"]: row["label_en"] for row in metadata}
    class_ids = sorted(names)
    predictions = pd.read_csv(PREDICTIONS)
    selected = pd.read_csv(SELECTED)
    selected = selected[selected.composition.eq("recompose")]
    audit_rows, confusion_rows, base_prediction_rows, selected_prediction_rows = [], [], [], []
    recall_matrix, labels = [], []
    for setting in selected.itertuples(index=False):
        backbone, selector, chosen_k = setting.backbone, setting.selector, int(setting.val_selected_k)
        label = setting_label(backbone, selector)
        base = predictions[(predictions.domain == "main") & (predictions.backbone == backbone) &
                           (predictions.selector == selector) & (predictions.k == 0)].sort_values("stem")
        chosen = predictions[(predictions.domain == "main") & (predictions.backbone == backbone) &
                             (predictions.selector == selector) & (predictions.k == chosen_k)].sort_values("stem")
        assert base.stem.tolist() == chosen.stem.tolist() and len(base) == 300
        base_prediction_rows.extend({"backbone": backbone, "selector": selector, "setting": label, "k": 0,
                                     "stem": row.stem, "true_index": int(row.true_index), "predicted_index": int(row.predicted_index),
                                     "correct": bool(row.true_index == row.predicted_index)} for row in base.itertuples(index=False))
        selected_prediction_rows.extend({"backbone": backbone, "selector": selector, "setting": label, "k": chosen_k,
                                         "stem": row.stem, "true_index": int(row.true_index), "predicted_index": int(row.predicted_index),
                                         "correct": bool(row.true_index == row.predicted_index)} for row in chosen.itertuples(index=False))
        recalls = []
        for class_id in class_ids:
            b = base[base.true_index.eq(class_id)]
            c = chosen[chosen.true_index.eq(class_id)]
            base_correct, selected_correct = int((b.predicted_index == class_id).sum()), int((c.predicted_index == class_id).sum())
            audit_rows.append({"backbone": backbone, "selector": selector, "setting": label, "selected_k": chosen_k,
                               "class_id": class_id, "class_name": names[class_id], "n_query": len(b),
                               "base_correct": base_correct, "base_recall": base_correct / len(b),
                               "selected_correct": selected_correct, "selected_recall": selected_correct / len(c),
                               "selected_minus_base_recall_pp": (selected_correct - base_correct) * 100 / len(c)})
            recalls.append((selected_correct - base_correct) * 100 / len(c))
        labels.append(label); recall_matrix.append(recalls)
        for condition, frame in (("base_k0", base), (f"selected_k{chosen_k}", chosen)):
            cm = confusion_matrix(frame.true_index, frame.predicted_index, labels=class_ids)
            for true_id in class_ids:
                for predicted_id in class_ids:
                    confusion_rows.append({"backbone": backbone, "selector": selector, "setting": label,
                                           "selected_k": chosen_k, "condition": condition,
                                           "true_class_id": true_id, "true_class_name": names[true_id],
                                           "predicted_class_id": predicted_id, "predicted_class_name": names[predicted_id],
                                           "count": int(cm[true_id, predicted_id])})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "classwise_recall_at_validation_selected_budget.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT / "confusion_matrices_base_and_selected.csv", index=False)
    pd.DataFrame(base_prediction_rows).to_csv(OUT / "main_base_k0_per_image_predictions.csv", index=False)
    pd.DataFrame(selected_prediction_rows).to_csv(OUT / "main_validation_selected_per_image_predictions.csv", index=False)
    summary = audit.groupby(["class_id", "class_name"], as_index=False).agg(
        n_settings=("setting", "count"), mean_recall_change_pp=("selected_minus_base_recall_pp", "mean"),
        all_settings_declined=("selected_minus_base_recall_pp", lambda values: bool((values < 0).all())))
    summary.to_csv(OUT / "classwise_heterogeneity_summary.csv", index=False)

    # Full names are preserved in the companion table; compact, unambiguous
    # labels keep the ten-category panel readable at final publication size.
    display_names = ["Sil", "Boo", "Dru", "Cig", "Shi", "Spr", "Pso", "Cel", "Pow", "MBS"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.85), sharex=True, sharey=True, constrained_layout=True)
    for ax, label, deltas in zip(axes.flat, labels, recall_matrix):
        colors = ["#b35806" if value < 0 else "#2166ac" for value in deltas]
        ax.bar(np.arange(len(deltas)), deltas, color=colors, width=.72)
        ax.axhline(0, color="#555555", linewidth=.6)
        ax.set_title(label + " selected minus k=0", fontweight="bold")
        ax.set_xticks(range(10), display_names)
        ax.set_ylabel("Recall change (pp)")
        ax.grid(axis="y", color="#dddddd", linewidth=.45)
    base = FIG / "classwise_recall_changes"
    require_matplotlib_panel_alignment(fig, json_out=f"{base}.alignment.json", overlay_svg=f"{base}.alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {}), (".tiff", {"dpi": 600})):
        fig.savefig(str(base) + suffix, bbox_inches="tight", **options)
    plt.close(fig)
    target = audit[audit.class_name.eq("minute brown scavenger beetle")].copy()
    lines = ["# Class-wise heterogeneity and failure analysis", "", "The global validation-selected budget is shared across classes. This descriptive audit reports recall on the fixed 30-image-per-class evaluation split; it does not conduct per-class hypothesis tests or create independent replication.", "", "## Minute brown scavenger beetle", "", "| Setting | k* | Base correct / 30 | Selected correct / 30 | Recall change (pp) |", "|---|---:|---:|---:|---:|"]
    for row in target.itertuples(index=False):
        lines.append(f"| {row.setting} | {row.selected_k} | {row.base_correct}/30 | {row.selected_correct}/30 | {row.selected_minus_base_recall_pp:+.2f} |")
    lines += ["", "The complete class-wise recall table and 10 × 10 base/selected confusion matrices are supplied as CSV files beside this report."]
    (OUT / "CLASSWISE_FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(target[["setting", "selected_k", "base_correct", "selected_correct", "selected_minus_base_recall_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
