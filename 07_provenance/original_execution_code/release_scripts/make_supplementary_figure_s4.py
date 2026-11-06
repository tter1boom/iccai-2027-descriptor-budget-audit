"""Rebuild Supplementary Figure S4 from the packaged diagnostic CSV."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_v14"
DATA_CANDIDATES = (
    ROOT / "machine_readable" / "representation_diagnostics.csv",
    ROOT / "external_aircraft_clean" / "results" / "representation_diagnostics.csv",
)
DATA = next((path for path in DATA_CANDIDATES if path.is_file()), DATA_CANDIDATES[0])

skill_scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
if skill_scripts:
    sys.path.insert(0, skill_scripts)
from audit_panel_alignment import require_matplotlib_panel_alignment


def main() -> None:
    df = pd.read_csv(DATA)
    required = {
        "seed", "backbone", "scorer", "k",
        "mean_correct_vs_strongest_competitor_margin", "prototype_drift_from_k0",
    }
    if missing := required - set(df.columns):
        raise ValueError(f"Diagnostic CSV is missing columns: {sorted(missing)}")
    if len(df) != 100 or df["seed"].nunique() != 5:
        raise ValueError("Expected five frozen resamples across the complete 100-row design")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "axes.titlesize": 7.6,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "lines.linewidth": 1.35,
    })

    backbone_colors = {"ViT-B-16": "#2478A8", "ViT-B-32": "#C36B35"}
    scorer_styles = {
        "positive_mean": ("-", "o", "Positive mean"),
        "contrastive_margin": ("--", "s", "Contrastive margin"),
    }
    panels = [
        ("prototype_drift_from_k0", "Prototype drift from k = 0\n(1 − cosine similarity)"),
        ("mean_correct_vs_strongest_competitor_margin",
         "Correct-class minus strongest-competitor cosine score"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharex=True)
    for ax, (metric, ylabel), letter in zip(axes, panels, ("a", "b")):
        for backbone in ("ViT-B-16", "ViT-B-32"):
            for scorer, (linestyle, marker, scorer_label) in scorer_styles.items():
                group = (df[(df.backbone == backbone) & (df.scorer == scorer)]
                         .groupby("k")[metric].agg(["mean", "std"]).reset_index())
                group = group.sort_values("k")
                ax.errorbar(
                    group["k"], group["mean"], yerr=group["std"].fillna(0),
                    color=backbone_colors[backbone], linestyle=linestyle, marker=marker,
                    markersize=3.2, capsize=2.0, elinewidth=0.8,
                    label=f"{backbone}, {scorer_label}", zorder=3,
                )
        ax.set_xlabel("Descriptor budget, k")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(5))
        ax.grid(axis="y", color="#D8DDE2", linewidth=0.55, zorder=0)
        ax.tick_params(direction="out", length=2.5, width=0.65)
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes,
                fontsize=8.0, fontweight="bold", va="bottom", ha="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, handlelength=2.2, columnspacing=1.4, handletextpad=0.5)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.91, bottom=0.29, wspace=0.36)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "FigureS4_development_domain_representation_diagnostics"
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure S4 written to {OUT}")


if __name__ == "__main__":
    main()
