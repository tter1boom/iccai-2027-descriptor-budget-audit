"""Summarise matched bootstrap selected-budget versus fixed-k=1 comparisons."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "experiments/results/experiment_64_same_support_calibration"))
    args = parser.parse_args()
    inp = Path(args.input_dir)
    out = inp / "bootstrap_selected_vs_fixed_k1.csv"
    all_k = pd.read_csv(inp / "teacher_bootstrap_all_k_predictions.csv")
    selected = pd.read_csv(inp / "teacher_bootstrap.csv")
    rows = []
    for row in selected.itertuples(index=False):
        block = all_k[(all_k.backbone == row.backbone) & (all_k.selector == row.selector) & (all_k.replicate == row.replicate)]
        score = {}
        for k, frame in block.groupby("k"):
            score[int(k)] = float((frame.predicted_index.to_numpy() == frame.true_index.to_numpy()).mean())
        rows.append({"backbone": row.backbone, "selector": row.selector, "replicate": row.replicate,
                     "selected_k": row.selected_k, "selected_accuracy": score[int(row.selected_k)],
                     "fixed_k1_accuracy": score[1], "selected_minus_fixed_k1": score[int(row.selected_k)] - score[1]})
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    summary = (frame.groupby(["backbone", "selector"], as_index=False)
               .agg(n_replicates=("replicate", "size"), mean_selected_accuracy=("selected_accuracy", "mean"),
                    mean_fixed_k1_accuracy=("fixed_k1_accuracy", "mean"), mean_difference=("selected_minus_fixed_k1", "mean"),
                    sd_difference=("selected_minus_fixed_k1", "std"), improved=("selected_minus_fixed_k1", lambda s: int((s > 0).sum())),
                    tied=("selected_minus_fixed_k1", lambda s: int((s == 0).sum())), worsened=("selected_minus_fixed_k1", lambda s: int((s < 0).sum())),
                    selected_k_frequency=("selected_k", lambda s: "; ".join(f"k={k}:{n}" for k, n in s.value_counts().sort_index().items()))))
    summary.to_csv(inp / "bootstrap_selected_vs_fixed_k1_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
