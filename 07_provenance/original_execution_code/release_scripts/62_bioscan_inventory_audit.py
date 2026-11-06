# -*- coding: utf-8 -*-
"""Create non-duplicated BIOSCAN taxon and split inventories for release."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "results" / "experiment_44_novelty_rebuild_C"
OUT = ROOT / "experiments" / "results" / "experiment_62_bioscan_inventory_audit"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The backbone-specific files encode the same seed partitions.  Retain one
    # unique partition record, rather than concatenate both backbones.
    frames = [pd.read_csv(SOURCE / "experiment_c_ViT-B-32_split_assignments.csv"),
              pd.read_csv(SOURCE / "experiment_c_ViT-B-16_split_assignments.csv")]
    combined = pd.concat(frames, ignore_index=True)
    keys = ["seed", "split", "stem", "class_index", "species"]
    partitions = combined.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    assert not partitions.duplicated(keys).any()
    partitions.to_csv(OUT / "bioscan_seed_partitions_unique.csv", index=False)

    seed_counts = partitions.groupby(["seed", "species", "class_index", "split"], as_index=False).stem.nunique().rename(columns={"stem": "n_unique_images"})
    seed_counts.to_csv(OUT / "bioscan_seed_taxon_split_counts.csv", index=False)
    total_counts = partitions.groupby(["seed", "split"], as_index=False).stem.nunique().rename(columns={"stem": "n_unique_images"})
    total_counts.to_csv(OUT / "bioscan_seed_split_totals.csv", index=False)
    taxon = partitions.groupby(["class_index", "species"], as_index=False).stem.nunique().rename(columns={"stem": "n_unique_images_across_all_seeds"})
    taxon.to_csv(OUT / "bioscan_taxon_inventory.csv", index=False)
    overlap = []
    for seed, frame in partitions.groupby("seed", sort=True):
        sets = {name: set(part.stem) for name, part in frame.groupby("split")}
        overlap.append({"seed": seed, "support_n": len(sets["support"]), "validation_n": len(sets["validation"]), "query_n": len(sets["query"]),
                        "support_validation_overlap": len(sets["support"] & sets["validation"]),
                        "support_query_overlap": len(sets["support"] & sets["query"]),
                        "validation_query_overlap": len(sets["validation"] & sets["query"]),
                        "union_n": len(set.union(*sets.values()))})
    audit = pd.DataFrame(overlap)
    assert (audit[["support_validation_overlap", "support_query_overlap", "validation_query_overlap"]].to_numpy() == 0).all()
    assert (audit[["support_n", "validation_n", "query_n", "union_n"]].to_numpy() == [100, 50, 150, 300]).all()
    audit.to_csv(OUT / "bioscan_seed_partition_integrity_audit.csv", index=False)
    (OUT / "BIOSCAN_INVENTORY_AUDIT.md").write_text(
        "# BIOSCAN inventory audit\n\n"
        "Backbone-specific partition files were deduplicated by seed, split, stem, class index and species. "
        "Each seed contains 10 taxa and 300 unique images: 100 support, 50 validation and 150 query. "
        "All three pairwise split overlaps are zero. `bioscan_taxon_inventory.csv` reports unique images rather than duplicated backbone rows.\n",
        encoding="utf-8")
    print(taxon.to_string(index=False)); print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
