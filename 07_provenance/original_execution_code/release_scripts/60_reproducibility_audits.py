# -*- coding: utf-8 -*-
"""Emit auditable split-component and BIOSCAN inventory records for the package."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "experiment_60_reproducibility_audits"
GROUPS = ROOT / "pest_vlm_dataset" / "annotations" / "leakage_groups.json"
SPLITS = ROOT / "pest_vlm_dataset" / "splits"
BIOSCAN = ROOT / "experiments" / "results" / "experiment_44_novelty_rebuild_C"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split_by_stem = {}
    for name in ("train_pool", "val", "test"):
        payload = json.loads((SPLITS / f"{name}.json").read_text(encoding="utf-8"))
        for path in payload["all_samples"]:
            split_by_stem[Path(path).stem] = name
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))
    rows = []
    covered = set()
    for group in groups["groups"]:
        memberships = [split_by_stem[stem] for stem in group["stems"] if stem in split_by_stem]
        covered.update(stem for stem in group["stems"] if stem in split_by_stem)
        rows.append({"component_id": group["group_id"], "class_id": group["class_id"], "n_images": len(memberships),
                     "splits_present": ";".join(sorted(set(memberships))), "n_splits_present": len(set(memberships)),
                     "cross_split_component": len(set(memberships)) > 1, "rule": groups["rule"], "seed": groups["seed"]})
    component = pd.DataFrame(rows).sort_values("component_id")
    component.to_csv(OUT / "visual_component_split_audit.csv", index=False)
    summary = pd.DataFrame([{"n_images_all_splits": len(split_by_stem), "n_images_in_non_singleton_components": len(covered),
                             "n_non_singleton_components": len(component), "n_cross_split_components": int(component.cross_split_component.sum()),
                             "rule": groups["rule"], "seed": groups["seed"]}])
    summary.to_csv(OUT / "visual_component_split_audit_summary.csv", index=False)

    assignments = pd.read_csv(BIOSCAN / "experiment_c_split_assignments.csv")
    class_columns = [column for column in assignments.columns if "class" in column.lower() or "tax" in column.lower()]
    keep = [column for column in ["seed", *class_columns] if column in assignments.columns]
    assignments.to_csv(OUT / "bioscan_seedwise_split_assignments.csv", index=False)
    inventory = assignments.groupby(class_columns, dropna=False).size().reset_index(name="rows") if class_columns else pd.DataFrame({"rows": [len(assignments)]})
    inventory.to_csv(OUT / "bioscan_taxon_inventory.csv", index=False)
    detail_files = [path.name for path in BIOSCAN.glob("experiment_c*.csv")]
    (OUT / "REPRODUCIBILITY_AUDIT.md").write_text(
        "# Reproducibility audits\n\n"
        f"Visual groups used `{groups['rule']}` (seed {groups['seed']}). The audit found "
        f"{int(component.cross_split_component.sum())} cross-split non-singleton component(s). "
        "The BIOSCAN inventory and seed-wise assignments are copied as machine-readable records; all experiment-C curves, rankings, selections and predictions remain in the accompanying BIOSCAN results directory.\n\n"
        "Files discovered: " + ", ".join(sorted(detail_files)) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
