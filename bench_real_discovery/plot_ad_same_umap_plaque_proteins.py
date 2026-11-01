"""Recolor the final AD z-CC/s-CC UMAP geometries by plaque proteins.

The script intentionally reads precomputed UMAP coordinates so background and
salient panels are directly comparable to the final cell-type/plaque figure.
Plaque annotations are used only for post-fit visualization and evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


REPRESENTATIONS = ["background_cellcharter", "salient_cellcharter"]
ROW_NAMES = {
    "background_cellcharter": "Haruka background (z-CC)",
    "salient_cellcharter": "Haruka salient (s-CC)",
}
METRIC_REP = {
    "background_cellcharter": "z_cc",
    "salient_cellcharter": "s_cc",
}
PLAQUE_COLORS = {
    "No plaque": "#d7d7d7",
    "p-tau only": "#d9485f",
    "Aβ only": "#6f6ad8",
    "Aβ + p-tau": "#f2a43a",
}


def metric_text(metrics: pd.DataFrame, representation: str, plaque: str) -> str:
    row = metrics[
        (metrics["model"] == "new_sw100_lambda20")
        & (metrics["representation"] == METRIC_REP[representation])
        & (metrics["plaque"] == plaque)
    ]
    if len(row) != 1:
        return ""
    row = row.iloc[0]
    return (
        f"spatial-block AUROC {row['spatial_block_auroc']:.3f}  |  "
        f"AP {row['spatial_block_average_precision']:.3f}"
    )


def scatter_binary(ax, frame: pd.DataFrame, column: str, color: str, label: str) -> None:
    negative = frame[column].eq(0)
    positive = ~negative
    ax.scatter(
        frame.loc[negative, "UMAP1"],
        frame.loc[negative, "UMAP2"],
        s=4,
        c="#d9d9d9",
        alpha=0.32,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        frame.loc[positive, "UMAP1"],
        frame.loc[positive, "UMAP2"],
        s=8,
        c=color,
        alpha=0.88,
        linewidths=0,
        rasterized=True,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#d9d9d9", markersize=5,
               label=f"Negative (n={negative.sum():,})"),
        Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5,
               label=f"{label}+ (n={positive.sum():,})"),
    ]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)


def scatter_classes(ax, frame: pd.DataFrame) -> None:
    for plaque_class in PLAQUE_COLORS:
        selected = frame["plaque_class"].eq(plaque_class)
        ax.scatter(
            frame.loc[selected, "UMAP1"],
            frame.loc[selected, "UMAP2"],
            s=4 if plaque_class == "No plaque" else 8,
            c=PLAQUE_COLORS[plaque_class],
            alpha=0.30 if plaque_class == "No plaque" else 0.88,
            linewidths=0,
            rasterized=True,
            label=f"{plaque_class} (n={selected.sum():,})",
        )
    ax.legend(loc="best", frameon=False, fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coords = pd.read_csv(args.coordinates)
    adata = ad.read_h5ad(args.data, backed="r")
    plaque = adata.obs[["plaque_tau", "plaque_abeta", "plaque_any"]].copy()
    plaque.index = plaque.index.astype(str)
    plaque = plaque.astype(int)
    coords = coords.join(plaque, on="obs_name", validate="many_to_one")
    if coords[["plaque_tau", "plaque_abeta", "plaque_any"]].isna().any().any():
        raise ValueError("Some UMAP cells do not have plaque annotations")

    coords["plaque_class"] = "No plaque"
    coords.loc[coords["plaque_tau"].eq(1), "plaque_class"] = "p-tau only"
    coords.loc[coords["plaque_abeta"].eq(1), "plaque_class"] = "Aβ only"
    coords.loc[
        coords["plaque_tau"].eq(1) & coords["plaque_abeta"].eq(1),
        "plaque_class",
    ] = "Aβ + p-tau"

    metrics = pd.read_csv(args.metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    for row_index, representation in enumerate(REPRESENTATIONS):
        frame = coords[coords["representation"].eq(representation)].copy()

        scatter_binary(axes[row_index, 0], frame, "plaque_tau", "#d9485f", "p-tau")
        axes[row_index, 0].set_title(
            f"p-tau\n{metric_text(metrics, representation, 'plaque_tau')}", fontsize=11
        )

        scatter_binary(axes[row_index, 1], frame, "plaque_abeta", "#6f6ad8", "Aβ")
        axes[row_index, 1].set_title(
            f"Aβ\n{metric_text(metrics, representation, 'plaque_abeta')}", fontsize=11
        )

        scatter_classes(axes[row_index, 2], frame)
        axes[row_index, 2].set_title(
            f"Plaque class\n{metric_text(metrics, representation, 'plaque_any')}", fontsize=11
        )

        for col_index, ax in enumerate(axes[row_index]):
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2" if col_index == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        axes[row_index, 0].text(
            -0.12,
            0.5,
            ROW_NAMES[representation],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle(
        "Jill AD disease_r1 — identical z-CC/s-CC UMAPs recolored by plaque proteins",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.002,
        "Plaque labels were not used for training; reported scores use spatial-block cross-validation.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    coords.to_csv(args.output.with_name(args.output.stem + "_coordinates.csv"), index=False)
    print(args.output)
    print(args.output.with_name(args.output.stem + "_coordinates.csv"))


if __name__ == "__main__":
    main()
