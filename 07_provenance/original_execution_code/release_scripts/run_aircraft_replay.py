"""Package-relative replay of clean Aircraft curves and operating-point selection.

This command uses the released per-image predictions. It does not require the
copyrighted image dataset or local CLIP checkpoints and does not rerun feature
extraction. The full end-to-end execution record is preserved separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def macro_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(recall_score(true, predicted, labels=np.arange(100), average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("aircraft_replay_output"))
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    results = package / "external_aircraft_clean" / "results"
    predictions_path = results / "predictions.csv"
    if not predictions_path.is_file():
        raise SystemExit("predictions.csv is available in the reproducibility archive, not the lightweight submission package")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    predictions = pd.read_csv(predictions_path)
    curve_rows = []
    for keys, block in predictions.groupby(["seed", "backbone", "scorer", "split", "k"]):
        seed, backbone, scorer, split, k = keys
        curve_rows.append({
            "seed": seed, "backbone": backbone, "scorer": scorer, "split": split, "k": k,
            "macro_accuracy": macro_accuracy(block["true_index"].to_numpy(), block["predicted_index"].to_numpy()),
            "top1_accuracy_secondary": float((block["true_index"] == block["predicted_index"]).mean()),
            "n": len(block),
        })
    curves = pd.DataFrame(curve_rows).sort_values(["seed", "backbone", "scorer", "split", "k"])
    selected_rows = []
    for (seed, backbone, scorer), block in curves[curves["split"] == "validation"].groupby(["seed", "backbone", "scorer"]):
        best = block["macro_accuracy"].max()
        selected_k = int(block.loc[block["macro_accuracy"] == best, "k"].min())
        selected_rows.append({"seed": seed, "backbone": backbone, "scorer": scorer,
                              "selected_k": selected_k, "validation_macro_accuracy": best})
    selected = pd.DataFrame(selected_rows).sort_values(["seed", "backbone", "scorer"])

    curves.to_csv(args.output / "budget_curves.csv", index=False)
    selected.to_csv(args.output / "selected_budgets.csv", index=False)
    released_curves = pd.read_csv(results / "budget_curves.csv").sort_values(["seed", "backbone", "scorer", "split", "k"])
    released_selected = pd.read_csv(results / "selected_budgets.csv").sort_values(["seed", "backbone", "scorer"])
    if not np.allclose(curves["macro_accuracy"], released_curves["macro_accuracy"], atol=1e-12):
        raise SystemExit("FAIL: replayed budget curves differ from released curves")
    if not np.array_equal(selected["selected_k"].to_numpy(), released_selected["selected_k"].to_numpy()):
        raise SystemExit("FAIL: replayed selected budgets differ from released selections")
    print("PASS: package-relative Aircraft prediction replay matches released curves and selections")


if __name__ == "__main__":
    main()
