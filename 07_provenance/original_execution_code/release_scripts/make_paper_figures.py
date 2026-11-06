"""Build the manuscript figures from frozen, machine-readable CSV inputs.

No model inference or experiment is run here.  All panels are descriptive
summaries of the supplied result files; inferential procedures are read from
their frozen analysis outputs and reported in the captions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


INK = "#203449"
MUTED = "#52677A"
GRID = "#D9E4EC"
PM = "#2F6F9F"
CM = "#D57A32"
B16 = "#356E8C"
B32 = "#2A9D8F"
RECOMPOSE = "#2F6F9F"
VOTE = "#D89A31"
ANCHORED = "#2A9D8F"
K_COLORS = ["#809DB7", "#2F6F9F", "#55A99B", "#D9A441", "#C96E5A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True,
                        help="Directory containing experiment_1..experiment_5 result folders")
    parser.add_argument("--source-only-root", type=Path, required=True,
                        help="Directory with the two source-only budget-curve CSVs")
    parser.add_argument("--aircraft-curves", type=Path, required=True,
                        help="Frozen clean Aircraft budget_curves.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def manual_alignment_check(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    """Portable fallback matching the equal-row geometry used in this suite."""
    fig.canvas.draw()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    boxes = [ax.get_position() for ax in axes]
    if len(boxes) < 2:
        status = "NOT APPLICABLE"
    else:
        tol = 1.5 / 72.0
        widths = [b.width * fig.get_figwidth() for b in boxes]
        heights = [b.height * fig.get_figheight() for b in boxes]
        bottoms = [b.y0 * fig.get_figheight() for b in boxes]
        if max(widths) - min(widths) > tol or max(heights) - min(heights) > tol or max(bottoms) - min(bottoms) > tol:
            raise RuntimeError(f"panel alignment failed for {stem}: {widths=}, {heights=}, {bottoms=}")
        status = "PASS"
    record = {
        "figure": stem,
        "backend": "matplotlib",
        "status": status,
        "tolerance_pt": 1.5,
        "axes": [
            {"index": i + 1, "x0_in": b.x0 * fig.get_figwidth(),
             "y0_in": b.y0 * fig.get_figheight(), "width_in": b.width * fig.get_figwidth(),
             "height_in": b.height * fig.get_figheight()}
            for i, b in enumerate(boxes)
        ],
    }
    (out_dir / f"{stem}.alignment-layout.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")


def alignment_gate(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    helper_dir = os.environ.get("NATURE_FIGURE_SKILL_SCRIPTS")
    if helper_dir and Path(helper_dir).is_dir():
        sys.path.insert(0, helper_dir)
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        manual_alignment_check(fig, stem, out_dir)
        return
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(out_dir / f"{stem}.alignment-layout.json"),
        overlay_svg=str(out_dir / f"{stem}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )


def save_figure(fig: plt.Figure, stem: str, out_dir: Path, aligned: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if aligned:
        alignment_gate(fig, stem, out_dir)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.035, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom", color=INK)


def source_only_curves(root: Path) -> pd.DataFrame:
    frames = []
    for backbone in ("ViT-B-16", "ViT-B-32"):
        frame = pd.read_csv(root / f"source_only_{backbone}_budget_curves.csv")
        frame = frame[(frame["split"] == "test") & frame["k"].between(0, 4)].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def figure1(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.025, "Fixed textual\nresource", "versioned A", "#DCEAF7"),
        (0.220, "Support-conditioned\nranking", "rule r; support S", "#E9E2F4"),
        (0.415, "Budget choice", "k = 0,…,4", "#FBE8C8"),
        (0.610, "Consumption\ninterface", "recompose / vote /\nensemble", "#DDF0E8"),
        (0.805, "Frozen VLM\nresponse", "backbone M; evaluate Q", "#F4DEE2"),
    ]
    w, y, h = 0.165, 0.66, 0.22
    for x, title, subtitle, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                               facecolor=color, edgecolor="#456174", linewidth=0.9)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.23, subtitle, ha="center", va="center",
                fontsize=5.4, color=MUTED, linespacing=0.95)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w + 0.004
        x1 = boxes[i + 1][0] - 0.004
        ax.annotate("", xy=(x1, y + h / 2), xytext=(x0, y + h / 2),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#456174"})

    controller = FancyBboxPatch((0.39, 0.43), 0.23, 0.12,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor="#FFF3D8", edgecolor="#C78931",
                                linewidth=0.9, linestyle="--")
    ax.add_patch(controller)
    ax.text(0.505, 0.507, "Optional validation controller", ha="center", va="center",
            fontsize=6.1, fontweight="bold", color=INK)
    ax.text(0.505, 0.464, "V selects k* (no guarantee)", ha="center",
            va="center", fontsize=5.4, color=MUTED)
    ax.annotate("", xy=(0.50, 0.65), xytext=(0.50, 0.55),
                arrowprops={"arrowstyle": "-|>", "lw": 0.9, "linestyle": "--", "color": "#C78931"})

    tiers = [
        (0.03, 0.27, 0.285, "Historical development resource", "Exploratory; construction not split-isolated", "#E5EFF8"),
        (0.357, 0.27, 0.285, "Image-blind source-only resource", "Post hoc provenance sensitivity", "#E8F3EE"),
        (0.684, 0.27, 0.285, "Clean FGVC-Aircraft", "One external domain; frozen protocol", "#F9EBD7"),
    ]
    for x, yy, ww, title, sub, color in tiers:
        ax.add_patch(FancyBboxPatch((x, yy), ww, 0.12,
                                    boxstyle="round,pad=0.01,rounding_size=0.012",
                                    facecolor=color, edgecolor="#7890A1", linewidth=0.75))
        ax.text(x + ww / 2, yy + 0.078, title, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color=INK)
        ax.text(x + ww / 2, yy + 0.034, sub, ha="center", va="center",
                fontsize=5.5, color=MUTED)
    ax.text(0.5, 0.16, "Five reviewer-driven analyses followed a documented post-review plan",
            ha="center", fontsize=7, fontweight="bold", color="#395B73")
    ax.text(0.5, 0.085,
            "Final protocol is consolidated, not a preregistration; descriptor response is conditional on A, r, g, M and D.",
            ha="center", fontsize=6.4, color=INK)
    ax.text(0.5, 0.035, "S, V and Q remain disjoint; no new dataset, model or image was introduced in the five analyses.",
            ha="center", fontsize=5.8, color=MUTED)
    save_figure(fig, "Figure1_study_design_and_evidence_hierarchy", out_dir, aligned=False)


def figure2(analysis_root: Path, source_only_root: Path, out_dir: Path) -> None:
    curves = source_only_curves(source_only_root)
    changes = pd.read_csv(analysis_root / "experiment_2_class_concentration" / "classwise_changes.csv")
    curves.to_csv(out_dir / "Figure2_source_only_curve_data.csv", index=False)
    changes.to_csv(out_dir / "Figure2_historical_class_concentration_data.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=False, constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.88, wspace=0.48)
    for ax, backbone, label in zip(axes[:2], ("ViT-B-16", "ViT-B-32"), ("a", "b")):
        part = curves[curves["backbone"] == backbone]
        for scorer, color, marker, name in (
            ("positive_mean", PM, "o", "Positive mean"),
            ("contrastive_margin", CM, "s", "Contrastive margin"),
        ):
            sub = part[part["selector"] == scorer].sort_values("k")
            ax.plot(sub["k"], 100 * sub["accuracy"], color=color, marker=marker,
                    markersize=3.6, linewidth=1.8, label=name)
        ax.set_title(backbone.replace("ViT-", ""), fontweight="bold", pad=3)
        ax.set_xticks(range(5))
        ax.set_xlabel("Descriptor budget k")
        ax.set_ylabel("Test accuracy (%)" if label == "a" else "")
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        panel_label(ax, label)
    axes[0].set_ylim(25, 48)
    axes[1].legend(loc="lower left", fontsize=5.8, frameon=False)

    summary = []
    order = [("ViT-B-32", "positive_mean"), ("ViT-B-32", "contrastive_margin"),
             ("ViT-B-16", "positive_mean"), ("ViT-B-16", "contrastive_margin")]
    for backbone, scorer in order:
        g = changes[(changes.backbone == backbone) & (changes.ranker == scorer)]
        drug = g[g.class_name.str.contains("drugstore", case=False)]
        remaining = g[~g.class_name.str.contains("drugstore", case=False)]
        n_remaining = int(remaining["n"].sum())
        remaining_gain_pp = 100 * float(remaining["delta_correct"].sum()) / n_remaining
        summary.append({
            "backbone": backbone, "ranker": scorer,
            "gain_without_drugstore_pp": remaining_gain_pp,
            "drugstore_share_positive_gain": float(g["positive_gain_concentration"].iloc[0]),
        })
    loco = pd.DataFrame(summary)
    loco.to_csv(out_dir / "Figure2_leave_one_class_out_data.csv", index=False)
    ax = axes[2]
    yy = np.arange(len(loco))
    bar_colors = [B32, B32, B16, B16]
    bars = ax.barh(yy, loco["gain_without_drugstore_pp"], color=bar_colors,
                   edgecolor="white", linewidth=0.5, height=0.62)
    for i, (bar, row) in enumerate(zip(bars, loco.itertuples(index=False))):
        if row.ranker == "contrastive_margin":
            bar.set_hatch("///")
        ax.text(max(row.gain_without_drugstore_pp, 0) + 0.12, i,
                f"{row.gain_without_drugstore_pp:.2f} | {100*row.drugstore_share_positive_gain:.1f}%",
                va="center", ha="left", fontsize=5.5, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.7)
    ax.set_yticks(yy, [
        "B/32 PM", "B/32 CM", "B/16 PM", "B/16 CM",
    ])
    ax.invert_yaxis()
    ax.set_xlim(-0.15, 8.5)
    ax.set_xlabel("LOCO gain (pp)\nDrugstore share (%)", fontsize=6.1)
    ax.set_title("Historical resource", fontweight="bold", pad=3)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    panel_label(ax, "c")
    save_figure(fig, "Figure2_resource_sensitivity_and_class_concentration", out_dir)


def figure3(analysis_root: Path, aircraft_curves_path: Path, out_dir: Path) -> None:
    curves = pd.read_csv(aircraft_curves_path)
    curves = curves[(curves["split"] == "test") & curves["k"].between(0, 4)].copy()
    adjacent = pd.read_csv(analysis_root / "experiment_1_aircraft_adjacent" / "adjacent_results.csv")
    curves.to_csv(out_dir / "Figure3_aircraft_budget_curves.csv", index=False)
    adjacent.to_csv(out_dir / "Figure3_adjacent_contrasts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.15), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.16, top=0.87, wspace=0.56)
    ax = axes[0]
    color_by_backbone = {"ViT-B-16": B16, "ViT-B-32": B32}
    style_by_ranker = {"positive_mean": "-", "contrastive_margin": "--"}
    marker_by_ranker = {"positive_mean": "o", "contrastive_margin": "s"}
    for (backbone, scorer, seed), g in curves.groupby(["backbone", "scorer", "seed"]):
        g = g.sort_values("k")
        ax.plot(g.k, 100 * g.macro_accuracy, color=color_by_backbone[backbone],
                linestyle=style_by_ranker[scorer], linewidth=0.65, alpha=0.19, zorder=1)
    for (backbone, scorer), g in curves.groupby(["backbone", "scorer"]):
        mean = g.groupby("k").macro_accuracy.mean()
        ax.plot(mean.index, 100 * mean.values, color=color_by_backbone[backbone],
                linestyle=style_by_ranker[scorer], marker=marker_by_ranker[scorer],
                markersize=3.5, linewidth=1.9, zorder=3)
    ax.set_xticks(range(5))
    ax.set_xlabel("Descriptor budget k")
    ax.set_ylabel("Aircraft test macro accuracy (%)")
    ax.set_ylim(15, 31)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    panel_label(ax, "a")
    legend_handles = [
        Line2D([0], [0], color=B16, lw=2, label="ViT-B/16"),
        Line2D([0], [0], color=B32, lw=2, label="ViT-B/32"),
        Line2D([0], [0], color=INK, lw=1.8, linestyle="-", label="Positive mean"),
        Line2D([0], [0], color=INK, lw=1.8, linestyle="--", label="Contrastive margin"),
        Line2D([0], [0], color=MUTED, lw=0.7, alpha=0.35, label="Individual resample"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 0.985),
               fontsize=5.4, ncol=3, columnspacing=0.8, handlelength=1.6)

    ax = axes[1]
    adjacent = adjacent.copy()
    adjacent["backbone_short"] = adjacent.backbone.map({"ViT-B-16": "B/16", "ViT-B-32": "B/32"})
    adjacent["ranker_short"] = adjacent.ranker.map({"positive_mean": "PM", "contrastive_margin": "CM"})
    adjacent["row_label"] = adjacent.backbone_short + " " + adjacent.ranker_short + " " + adjacent.transition
    adjacent = adjacent.sort_values(
        ["backbone", "ranker", "k_from"], ascending=[False, False, True]).reset_index(drop=True)
    ypos = np.arange(len(adjacent))
    supported = adjacent.holm_p_16 < 0.05
    colors = [B16 if x == "ViT-B-16" else B32 for x in adjacent.backbone]
    for i, row in adjacent.iterrows():
        ax.plot([row.ci95_low_pp, row.ci95_high_pp], [i, i], color=colors[i],
                linewidth=1.0, alpha=0.88, zorder=1)
        ax.plot(row.delta_pp, i, marker="o", markersize=3.7,
                markerfacecolor=(colors[i] if supported.iloc[i] else "white"),
                markeredgecolor=colors[i], markeredgewidth=1.0, zorder=3)
    ax.axvline(0, color="#7B8790", linewidth=0.75, linestyle=":")
    ax.set_yticks(ypos, adjacent.row_label, fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(-2.3, 3.2)
    ax.set_xlabel("Adjacent-budget change (pp), 95% paired bootstrap CI")
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    panel_label(ax, "b")
    ax.text(0.98, 0.015, "Filled marker: Holm-adjusted p < 0.05",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color=MUTED)
    save_figure(fig, "Figure3_aircraft_curves_and_adjacent_uncertainty", out_dir)


def figure4(analysis_root: Path, out_dir: Path) -> None:
    base = analysis_root / "experiment_3_validation_size" / "server_runs" / "exp3_validation_size"
    regret = pd.read_csv(base / "regret_by_validation_size.csv")
    selected = pd.read_csv(base / "selected_k_distribution.csv")
    support_selections = pd.read_csv(base / "full_validation_support_selections.csv")
    test_curves = pd.read_csv(analysis_root / "experiment_3_validation_size" / "server_runs" / "merged" / "test_curves.csv")
    regret.to_csv(out_dir / "Figure4_validation_regret_by_size.csv", index=False)
    selected.to_csv(out_dir / "Figure4_validation_selected_k_distribution.csv", index=False)
    test_curves.to_csv(out_dir / "Figure4_test_oracle_curves.csv", index=False)

    order_sizes = [1, 2, 5, 10, 20, "all"]
    size_x = {str(x): i for i, x in enumerate(order_sizes)}
    colors = {
        ("ViT-B-16", "positive_mean"): "#356E8C",
        ("ViT-B-16", "contrastive_margin"): "#8B72A7",
        ("ViT-B-32", "positive_mean"): "#2A9D8F",
        ("ViT-B-32", "contrastive_margin"): "#D57A32",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    ax = axes[0]
    for (backbone, ranker), g in regret.groupby(["backbone", "ranker"]):
        g = g.copy()
        g["x"] = g.n_validation_per_class.astype(str).map(size_x)
        g = g.sort_values("x")
        low = 100 * g.oracle_regret_ci95_low.astype(float)
        high = 100 * g.oracle_regret_ci95_high.astype(float)
        ax.plot(g.x, 100 * g.mean_regret, color=colors[(backbone, ranker)],
                marker="o", markersize=3.3, linewidth=1.55,
                label=f"{backbone.replace('ViT-', '')} {'PM' if ranker == 'positive_mean' else 'CM'}")
        ax.fill_between(g.x.to_numpy(dtype=float), low.to_numpy(dtype=float), high.to_numpy(dtype=float),
                        color=colors[(backbone, ranker)], alpha=0.14, linewidth=0, zorder=1)
    ax.set_xticks(range(6), ["1", "2", "5", "10", "20", "All"])
    ax.set_xlabel("Validation images per class")
    ax.set_ylabel("Mean regret vs test-best k (pp)")
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(fontsize=5.6, loc="upper right", ncol=2, columnspacing=0.8)
    panel_label(ax, "a")

    ax = axes[1]
    condition = ("ViT-B-32", "positive_mean")
    full = support_selections[(support_selections.backbone == condition[0]) &
                              (support_selections.ranker == condition[1])]
    if len(full) != 20 or full.support_seed.nunique() != 20:
        raise ValueError("Complete-validation operating-point panel requires 20 unique support resamples")
    if not full.groupby("support_seed").selected_k.nunique().eq(1).all():
        raise ValueError("Complete-validation selection is not unique within support resample")
    full_freq = full.selected_k.value_counts().reindex(range(5), fill_value=0)
    test_part = test_curves[(test_curves.backbone == condition[0]) &
                            (test_curves.ranker == condition[1])]
    pivot = test_part.pivot(index="support_seed", columns="k", values="test_macro_accuracy")
    oracle_freq = pivot.idxmax(axis=1).value_counts().reindex(range(5), fill_value=0)
    dist = pd.DataFrame({
        "k": range(5),
        "full_validation_selected_support_resamples": full_freq.values,
        "full_validation_support_resamples": int(full_freq.sum()),
        "full_validation_selection_pct": full_freq.values / full_freq.sum() * 100,
        "test_oracle_support_resamples": oracle_freq.values,
        "test_oracle_support_n": int(oracle_freq.sum()),
        "test_oracle_pct": oracle_freq.values / oracle_freq.sum() * 100,
    })
    dist.to_csv(out_dir / "Figure4_B32_PM_selection_vs_oracle.csv", index=False)
    yloc = [0.15, 0.60]
    labels = ["Test-oracle k\n(20 support resamples)", "Selected k with all V\n(20 support resamples)"]
    for yy, col in zip(yloc, ("test_oracle_pct", "full_validation_selection_pct")):
        left = 0
        for k, row in dist.iterrows():
            width = float(row[col])
            if width > 0:
                ax.barh(yy, width, left=left, height=0.19, color=K_COLORS[k],
                        edgecolor="white", linewidth=0.55)
                if width >= 9:
                    ax.text(left + width / 2, yy, f"k={k}\n{width:.0f}%", ha="center",
                            va="center", fontsize=5.3, color="white" if k in (1, 4) else INK,
                            fontweight="bold")
                left += width
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.15, 0.92)
    ax.set_yticks(yloc, labels, fontsize=6.0)
    ax.set_xlabel("Frequency (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.96, "Top-1 modal agreement: 91.2%\nMean best-second gap: 0.40 pp",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.4, color=MUTED,
            linespacing=1.0)
    panel_label(ax, "b")
    save_figure(fig, "Figure4_validation_size_and_operating_point_mismatch", out_dir)


def figure5(analysis_root: Path, out_dir: Path) -> None:
    data = pd.read_csv(analysis_root / "experiment_5_matched_prior" / "matched_interface_curves.csv")
    data = data[(data.domain == "aircraft") & (data.ranker == "positive_mean") &
                data.interface.isin(["single_recomposed_prototype", "independent_attribute_voting",
                                     "class_anchored_descriptor_ensemble"])].copy()
    data.to_csv(out_dir / "Figure5_matched_interface_curves_raw.csv", index=False)
    summary = data.groupby(["backbone", "interface", "k"]).test_macro_accuracy.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(out_dir / "Figure5_matched_interface_summary.csv", index=False)

    names = {
        "single_recomposed_prototype": ("Single recomposed prototype", RECOMPOSE, "o", "-"),
        "independent_attribute_voting": ("Independent attribute voting", VOTE, "s", "--"),
        "class_anchored_descriptor_ensemble": ("Class-anchored ensemble", ANCHORED, "D", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.30, top=0.84, wspace=0.24)
    for ax, backbone, label in zip(axes, ("ViT-B-16", "ViT-B-32"), ("a", "b")):
        for interface, (display, color, marker, linestyle) in names.items():
            g = summary[(summary.backbone == backbone) & (summary.interface == interface)].sort_values("k")
            ax.errorbar(g.k, 100 * g["mean"], yerr=100 * g["std"].fillna(0),
                        color=color, marker=marker, linestyle=linestyle, linewidth=1.55,
                        markersize=3.2, capsize=2.1, elinewidth=0.8, label=display)
        ax.set_title(backbone.replace("ViT-", ""), fontweight="bold", pad=3)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Descriptor budget k")
        ax.set_ylabel("Aircraft test macro accuracy (%)" if label == "a" else "")
        ax.set_ylim(0, 30)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        panel_label(ax, label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, fontsize=5.5, frameon=False, columnspacing=0.9, handlelength=1.7)
    save_figure(fig, "Figure5_matched_descriptor_consumption_interfaces", out_dir)


def main() -> None:
    args = parse_args()
    configure_plotting()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure1(args.out_dir)
    figure2(args.analysis_root, args.source_only_root, args.out_dir)
    figure3(args.analysis_root, args.aircraft_curves, args.out_dir)
    figure4(args.analysis_root, args.out_dir)
    figure5(args.analysis_root, args.out_dir)
    print(f"Figures written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
