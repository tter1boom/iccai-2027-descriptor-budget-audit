"""Build Figure S1 from source-only curves and corrected historical intervals.

No numerical claim report is generated here. All paired intervals come from
the canonical image-level class-macro visual-component bootstrap output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "machine_readable" / "source_only_resource"
STATS = ROOT / "machine_readable" / "group_clustered_inference" / "group_clustered_paired_statistics.csv"
FIG_DEFAULT = SOURCE / "figures"
sys.path.insert(0, r"<redacted-local-auxiliary-path>")
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": .7,
})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=FIG_DEFAULT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat([
        pd.read_csv(SOURCE / "source_only_ViT-B-32_budget_curves.csv"),
        pd.read_csv(SOURCE / "source_only_ViT-B-16_budget_curves.csv"),
    ], ignore_index=True)
    stats = pd.read_csv(STATS)
    stats = stats[stats.comparison.eq("baseline_k0_vs_validation_selected")].copy()

    fig = plt.figure(figsize=(7.1, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    colors = {"positive_mean": "#1f77b4", "contrastive_margin": "#d55e00"}
    labels = {"positive_mean": "Positive mean", "contrastive_margin": "Contrastive margin"}
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax, backbone in zip(axes, ("ViT-B-32", "ViT-B-16")):
        test = curves[(curves.backbone == backbone) & curves.split.eq("test")]
        for scorer in ("positive_mean", "contrastive_margin"):
            part = test[test.selector.eq(scorer)].sort_values("k")
            ax.plot(part.k, 100 * part.accuracy, marker="o", markersize=3,
                    linewidth=1.25, color=colors[scorer], label=labels[scorer])
            selected = curves[(curves.backbone == backbone) & (curves.selector == scorer) & curves.split.eq("val")].sort_values(["accuracy", "k"], ascending=[False, True]).iloc[0]
            ax.scatter([selected.k], [100 * part.loc[part.k.eq(selected.k), "accuracy"].iloc[0]],
                       marker="D", s=22, color=colors[scorer], zorder=4)
        ax.set_xticks(range(5)); ax.set_xlabel("Appended descriptors, k")
        ax.set_ylabel("Fixed-evaluation accuracy (%)")
        ax.set_title(backbone.replace("ViT-", "") + " source-only resource", fontweight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
    axes[0].legend(loc="best", frameon=False)

    ax = fig.add_subplot(grid[1, :])
    display = {"positive_mean": "PM", "contrastive_margin": "CM"}
    order = [("ViT-B-32", "positive_mean"), ("ViT-B-32", "contrastive_margin"),
             ("ViT-B-16", "positive_mean"), ("ViT-B-16", "contrastive_margin")]
    rows = []
    for backbone, scorer in order:
        hit = stats[(stats.backbone == backbone) & (stats.selector == scorer)]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one corrected historical contrast for {backbone}/{scorer}")
        rows.append(hit.iloc[0])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    delta = 100 * selected.accuracy_difference_candidate_minus_comparator.to_numpy()
    low = 100 * selected.group_clustered_bootstrap_ci95_low.to_numpy()
    high = 100 * selected.group_clustered_bootstrap_ci95_high.to_numpy()
    y = np.arange(len(selected))[::-1]
    ax.errorbar(delta, y, xerr=np.vstack((delta - low, high - delta)), fmt="o",
                color="#3b3b3b", capsize=2.5, markersize=4)
    ax.axvline(0, color="#888888", linewidth=.7)
    ax.set_yticks(y, [f"{b.replace('ViT-', '')} {display[s]}" for b, s in order])
    ax.set_xlabel("Selected − base accuracy (percentage points)")
    ax.set_title("Archived historical resource: exploratory paired analysis (macro cluster bootstrap, 95% CI)", fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    base = args.out_dir / "source_only_and_group_clustered_audits"
    require_matplotlib_panel_alignment(fig, json_out=f"{base}.alignment.json",
                                       overlay_svg=f"{base}.alignment.svg",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS: regenerated exploratory Figure S1 assets in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
