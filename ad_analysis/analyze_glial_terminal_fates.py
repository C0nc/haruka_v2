#!/usr/bin/env python3
"""Restricted AD terminal-fate analysis for microglia/macrophage, astrocytes and oligodendrocytes."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid


ROOT = Path("/home/yancui/haruka_v2")
RESULTS = ROOT / "ad_analysis/scc_dpt_cellrank2"
OUT = RESULTS / "glia_terminal_analysis"
sys.path.insert(0, str(ROOT / "ad_analysis"))
from analyze_initial_molecular_terminal_fates import meta_analyze  # noqa: E402


STAR_LINEAGES = ["Micro", "Astro", "Oligo"]


def terminal_masks(frame: pd.DataFrame):
    terminal = pd.to_numeric(frame.terminal_state, errors="coerce")
    plaque = pd.to_numeric(frame.plaque_associated_terminal, errors="coerce")
    observed = terminal.notna().to_numpy()
    plaque_mask = observed & terminal.eq(plaque).to_numpy()
    alternative_mask = observed & ~terminal.eq(plaque).to_numpy()
    return plaque_mask, alternative_mask


def filtered_arrows(frame: pd.DataFrame, density: float = 0.14):
    xy = frame[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    velocity = frame[["cellrank_projected_velocity_x", "cellrank_projected_velocity_y"]].to_numpy(dtype=float)
    x_grid, v_grid = compute_velocity_on_grid(
        xy, velocity, density=density, smooth=0.7, min_mass=20, autoscale=True
    )
    lo, hi = np.quantile(xy, 0.02, axis=0), np.quantile(xy, 0.98, axis=0)
    magnitude = np.linalg.norm(v_grid, axis=1)
    keep = (
        (x_grid >= lo).all(axis=1)
        & (x_grid <= hi).all(axis=1)
        & (magnitude >= np.quantile(magnitude, 0.20))
    )
    return x_grid[keep], v_grid[keep]


def draw_panel(ax, frame: pd.DataFrame, title: str, show_legend: bool = False):
    if frame.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No cells", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return None, 0
    ordered = frame.sort_values("dpt_pseudotime")
    points = ax.scatter(
        ordered.spatial_x,
        ordered.spatial_y,
        c=ordered.dpt_pseudotime,
        cmap="magma",
        vmin=0,
        vmax=1,
        s=30,
        linewidths=0,
        alpha=0.82,
        rasterized=True,
    )
    try:
        x_grid, v_grid = filtered_arrows(frame)
    except (ValueError, IndexError):
        x_grid = np.empty((0, 2))
        v_grid = np.empty((0, 2))
    if len(x_grid):
        ax.quiver(
            x_grid[:, 0],
            x_grid[:, 1],
            v_grid[:, 0],
            v_grid[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.004,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.2,
            alpha=0.9,
            zorder=4,
        )
    plaque, alternative = terminal_masks(frame)
    ax.scatter(
        frame.loc[alternative, "spatial_x"],
        frame.loc[alternative, "spatial_y"],
        marker="X",
        s=105,
        c="#57C665",
        edgecolors="black",
        linewidths=0.5,
        label="Alternative terminal",
        zorder=5,
    )
    ax.scatter(
        frame.loc[plaque, "spatial_x"],
        frame.loc[plaque, "spatial_y"],
        marker="*",
        s=155,
        c="#25D9E8",
        edgecolors="black",
        linewidths=0.55,
        label="Plaque-proximal terminal",
        zorder=6,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    if show_legend:
        ax.legend(loc="upper left", fontsize=9, frameon=True)
    return points, len(x_grid)


def plot_starmap(scores: pd.DataFrame, projected: pd.DataFrame, out: Path):
    frame = scores[(scores.dataset == "starmap") & scores.main_cluster.isin(STAR_LINEAGES)].merge(
        projected[
            [
                "dataset",
                "replicate",
                "main_cluster",
                "cell_id",
                "cellrank_projected_velocity_x",
                "cellrank_projected_velocity_y",
            ]
        ],
        on=["dataset", "replicate", "main_cluster", "cell_id"],
    )
    reps = sorted(frame.replicate.unique())
    fig, axes = plt.subplots(len(STAR_LINEAGES), len(reps), figsize=(9 * len(reps), 7.2 * len(STAR_LINEAGES)), squeeze=False)
    arrow_rows = []
    for row, lineage in enumerate(STAR_LINEAGES):
        for col, replicate in enumerate(reps):
            group = frame[(frame.main_cluster == lineage) & (frame.replicate == replicate)].copy()
            points, n_arrows = draw_panel(
                axes[row, col], group, f"{lineage} · {replicate}", show_legend=(row == 0 and col == 0)
            )
            arrow_rows.append({"dataset": "starmap", "lineage": lineage, "replicate": replicate, "n_cells": len(group), "n_arrows": n_arrows})
    valid_points = next(
        (collection for axis in axes.ravel() for collection in axis.collections
         if hasattr(collection, "get_array") and collection.get_array() is not None),
        None,
    )
    if valid_points is not None:
        fig.colorbar(valid_points, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01, label="Within-lineage s-CC DPT")
    fig.suptitle("STARmap AD: glial-lineage CellRank trajectories and spatial terminal stages", y=1.002)
    fig.savefig(out / "starmap_micro_astro_oligo_spatial_terminal_stages.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return frame, pd.DataFrame(arrow_rows)


def plot_codex(scores: pd.DataFrame, projected: pd.DataFrame, out: Path):
    frame = scores[scores.dataset == "codex"].merge(
        projected[
            [
                "dataset",
                "replicate",
                "main_cluster",
                "cell_id",
                "cellrank_projected_velocity_x",
                "cellrank_projected_velocity_y",
            ]
        ],
        on=["dataset", "replicate", "main_cluster", "cell_id"],
    )
    reps = sorted(frame.replicate.unique())
    fig, axes = plt.subplots(1, len(reps), figsize=(8.5 * len(reps), 7.5), squeeze=False)
    rows = []
    for col, replicate in enumerate(reps):
        group = frame[frame.replicate == replicate].copy()
        points, n_arrows = draw_panel(
            axes[0, col], group, f"Myeloid/macrophage · {replicate}", show_legend=(col == 0)
        )
        rows.append({"dataset": "codex", "lineage": "Myeloid/macrophage", "replicate": replicate, "n_cells": len(group), "n_arrows": n_arrows})
    fig.colorbar(points, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01, label="Within-state s-CC DPT")
    fig.suptitle("CODEX AD: myeloid CellRank trajectories and spatial terminal stages", y=1.01)
    fig.savefig(out / "codex_myeloid_macrophage_spatial_terminal_stages.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return frame, pd.DataFrame(rows)


def lineage_molecular_analysis(out: Path):
    groupwise = pd.read_csv(
        RESULTS / "starmap_initial_molecular_terminal_associations_by_group.csv.gz",
        dtype={"main_cluster": str},
    )
    metas, top_parts = [], []
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    for ax, lineage in zip(axes, STAR_LINEAGES):
        meta = meta_analyze(groupwise[groupwise.main_cluster == lineage])
        meta.insert(1, "lineage", lineage)
        meta["same_direction_both_replicates"] = (
            meta.positive_group_fraction.eq(1.0) | meta.positive_group_fraction.eq(0.0)
        )
        metas.append(meta)
        consistent = meta[meta.same_direction_both_replicates]
        selected = pd.concat(
            [consistent.nlargest(8, "meta_spearman_rho"), consistent.nsmallest(8, "meta_spearman_rho")]
        ).drop_duplicates("feature").sort_values("meta_spearman_rho")
        top_parts.append(selected)
        colors = np.where(selected.meta_spearman_rho >= 0, "#20A6B8", "#61B15A")
        ax.barh(selected.feature, selected.meta_spearman_rho, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(lineage)
        ax.set_xlabel("Early-cell meta Spearman ρ")
        ax.set_ylabel("")
    fig.suptitle(
        "STARmap: initial molecular associations with glial terminal allocation\n"
        "positive → plaque-proximal; negative → alternative; same direction in both replicates"
    )
    fig.tight_layout()
    fig.savefig(out / "starmap_glial_initial_molecular_terminal_predictors.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    all_meta = pd.concat(metas, ignore_index=True)
    all_top = pd.concat(top_parts, ignore_index=True)
    all_meta.to_csv(out / "starmap_glial_initial_molecular_terminal_associations.csv", index=False)
    all_top.to_csv(out / "starmap_glial_top_initial_molecular_terminal_predictors.csv", index=False)
    codex_meta = pd.read_csv(RESULTS / "codex_initial_molecular_terminal_associations_meta.csv")
    codex_meta.to_csv(out / "codex_myeloid_initial_protein_terminal_associations.csv", index=False)
    return all_meta, all_top, codex_meta


def terminal_counts(frame: pd.DataFrame, lineage_key: str) -> pd.DataFrame:
    plaque, alternative = terminal_masks(frame)
    temp = frame[["dataset", "replicate", lineage_key]].copy()
    temp["terminal_stage"] = np.where(
        plaque, "Plaque-proximal terminal", np.where(alternative, "Alternative terminal", "Transient")
    )
    return temp.groupby(["dataset", "replicate", lineage_key, "terminal_stage"]).size().rename("n_cells").reset_index()


def main():
    global RESULTS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    RESULTS = args.root
    OUT = args.out if args.out is not None else RESULTS / "glia_terminal_analysis"
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(RESULTS / "cellrank_cell_scores.csv.gz", dtype={"cell_id": str, "main_cluster": str})
    projected = pd.read_csv(
        RESULTS / "cellrank_sparse_spatial_projection_cells.csv.gz",
        dtype={"cell_id": str, "main_cluster": str},
    )
    starmap, star_arrows = plot_starmap(scores, projected, OUT)
    codex, codex_arrows = plot_codex(scores, projected, OUT)
    star_counts = terminal_counts(starmap, "main_cluster").rename(columns={"main_cluster": "lineage"})
    codex = codex.copy()
    codex["lineage"] = "Myeloid/macrophage"
    codex_counts = terminal_counts(codex, "lineage")
    pd.concat([star_counts, codex_counts], ignore_index=True).to_csv(OUT / "terminal_stage_counts.csv", index=False)
    pd.concat([star_arrows, codex_arrows], ignore_index=True).to_csv(OUT / "spatial_panel_counts.csv", index=False)
    lineage_molecular_analysis(OUT)
    print(f"Saved restricted glial terminal analysis to {OUT}")


if __name__ == "__main__":
    main()
