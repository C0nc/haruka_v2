#!/usr/bin/env python3
"""Sparse, large-arrow CellRank 2 projection on AD tissue coordinates.

The CellRank PseudotimeKernel and its official ``plot_projection`` function are
used on a spatial KNN graph that is block-restricted by replicate and broad
biological cluster.  ``density=0.14`` and ``min_mass=20`` retain only a small,
well-supported subset of large aggregate arrows.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import anndata as ad
import cellrank as cr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from cellrank.kernels.utils._projection import TmatProjection
from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid


DEFAULT_ROOT = Path("/home/yancui/haruka_v2/ad_analysis/scc_dpt_cellrank2")


def block_spatial_graph(frame: pd.DataFrame, k: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Construct a spatial KNN graph without crossing biological clusters."""
    n = len(frame)
    rows, cols, weights, distances = [], [], [], []
    for _, loc in frame.groupby("main_cluster", sort=True).groups.items():
        take = np.asarray(list(loc), dtype=int)
        if len(take) < 3:
            continue
        xy = frame.iloc[take][["spatial_x", "spatial_y"]].to_numpy(dtype=float)
        kk = min(k + 1, len(take))
        dist, ind = NearestNeighbors(n_neighbors=kk).fit(xy).kneighbors(xy)
        dist, ind = dist[:, 1:], ind[:, 1:]
        sigma = max(float(np.median(dist)), 1e-8)
        weight = np.exp(-0.5 * np.square(dist / sigma))
        rows.append(np.repeat(take, ind.shape[1]))
        cols.append(take[ind.ravel()])
        weights.append(weight.ravel())
        distances.append(dist.ravel())
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    conn = sparse.csr_matrix((np.concatenate(weights), (row, col)), shape=(n, n))
    dmat = sparse.csr_matrix((np.concatenate(distances), (row, col)), shape=(n, n))
    conn = conn.maximum(conn.T)
    dmat = dmat.maximum(dmat.T)
    # Preserve self-transition rows after the forward-pseudotime bias.
    conn.setdiag(1.0)
    dmat.setdiag(0)
    dmat.eliminate_zeros()
    return conn, dmat


def make_kernel(frame: pd.DataFrame, k: int):
    obs = frame[["dpt_pseudotime", "main_cluster"]].copy()
    obs.index = pd.Index(frame.cell_id.astype(str), name="cell_id")
    a = ad.AnnData(X=np.zeros((len(frame), 1), dtype=np.float32), obs=obs)
    a.obsm["X_spatial"] = frame[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32)
    conn, dmat = block_spatial_graph(frame.reset_index(drop=True), k=k)
    a.obsp["connectivities"] = conn
    a.obsp["distances"] = dmat
    a.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": k, "method": "spatial_block_knn", "metric": "euclidean"},
    }
    kernel = cr.kernels.PseudotimeKernel(a, time_key="dpt_pseudotime")
    kernel.compute_transition_matrix(
        threshold_scheme="soft", n_jobs=1, show_progress_bar=False, check_irreducibility=False
    )
    return a, kernel, conn


def plot_dataset(
    dataset: str,
    cells: pd.DataFrame,
    out: Path,
    k: int,
    density: float,
    min_mass: float,
) -> pd.DataFrame:
    reps = sorted(cells.replicate.unique())
    fig, axes = plt.subplots(1, len(reps), figsize=(8.2 * len(reps), 7.5), squeeze=False)
    projected = []
    for ax, replicate in zip(axes[0], reps):
        frame = cells[cells.replicate == replicate].reset_index(drop=True)
        a, kernel, conn = make_kernel(frame, k=k)
        projection = TmatProjection(kernel, basis="spatial")
        projection.project(key_added="T_fwd", recompute=True, connectivities=conn)
        vector = np.asarray(a.obsm["T_fwd_spatial"])
        xy = np.asarray(a.obsm["X_spatial"])
        x_grid, v_grid = compute_velocity_on_grid(
            xy, vector, density=density, smooth=0.7, min_mass=min_mass, autoscale=True
        )
        margin_lo = np.quantile(xy, 0.02, axis=0)
        margin_hi = np.quantile(xy, 0.98, axis=0)
        magnitude = np.linalg.norm(v_grid, axis=1)
        keep = (
            (x_grid >= margin_lo).all(axis=1)
            & (x_grid <= margin_hi).all(axis=1)
            & (magnitude >= np.quantile(magnitude, 0.20))
        )
        x_grid, v_grid = x_grid[keep], v_grid[keep]
        kernel.plot_projection(
            basis="spatial",
            key_added="T_fwd",
            recompute=False,
            stream=False,
            connectivities=conn,
            X_grid=x_grid,
            V_grid=v_grid,
            density=density,
            min_mass=min_mass,
            smooth=0.7,
            arrow_size=3.0,
            arrow_length=3.5,
            arrow_color="black",
            color="dpt_pseudotime",
            color_map="magma",
            size=30,
            alpha=0.72,
            colorbar=True,
            legend_loc="none",
            title=f"{replicate}: sparse CellRank spatial projection",
            frameon=False,
            ax=ax,
            show=False,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_ylabel("")
        terminal = pd.to_numeric(frame["terminal_state"], errors="coerce")
        plaque_terminal = pd.to_numeric(frame["plaque_associated_terminal"], errors="coerce")
        is_terminal = terminal.notna().to_numpy()
        is_plaque_terminal = is_terminal & terminal.eq(plaque_terminal).to_numpy()
        is_alternative_terminal = is_terminal & ~terminal.eq(plaque_terminal).to_numpy()
        ax.scatter(
            frame.loc[is_alternative_terminal, "spatial_x"],
            frame.loc[is_alternative_terminal, "spatial_y"],
            marker="X", s=95, c="#57C665", edgecolors="black", linewidths=0.45,
            label="Alternative terminal", zorder=5,
        )
        ax.scatter(
            frame.loc[is_plaque_terminal, "spatial_x"],
            frame.loc[is_plaque_terminal, "spatial_y"],
            marker="*", s=145, c="#25D9E8", edgecolors="black", linewidths=0.5,
            label="Plaque-proximal terminal", zorder=6,
        )
        if ax is axes[0, 0]:
            ax.legend(loc="upper left", frameon=True, fontsize=9, markerscale=0.8)
        vector = np.asarray(a.obsm["T_fwd_spatial"])
        projected.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "replicate": replicate,
                    "main_cluster": frame.main_cluster.astype(str),
                    "cell_id": frame.cell_id.astype(str),
                    "dpt_pseudotime": frame.dpt_pseudotime,
                    "cellrank_projected_velocity_x": vector[:, 0],
                    "cellrank_projected_velocity_y": vector[:, 1],
                }
            )
        )
    fig.suptitle(
        f"{dataset.upper()}: Haruka s-CC pseudotime projected with CellRank 2\n"
        f"small, interior high-support arrows only (density={density:g}, minimum mass={min_mass:g}%)",
        y=1.015,
    )
    fig.savefig(out / f"{dataset}_cellrank_sparse_spatial_arrows.png", dpi=240, bbox_inches="tight")
    # This is now the default arrow view requested by the analysis protocol.
    fig.savefig(out / f"{dataset}_spatial_dpt_velocity_arrows.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pd.concat(projected, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--spatial-neighbors", type=int, default=15)
    parser.add_argument("--density", type=float, default=0.14)
    parser.add_argument("--min-mass", type=float, default=20.0)
    args = parser.parse_args()
    cells = pd.read_csv(args.root / "cellrank_cell_scores.csv.gz", dtype={"cell_id": str, "main_cluster": str})
    projections = []
    for dataset, frame in cells.groupby("dataset", sort=True):
        projections.append(
            plot_dataset(
                dataset,
                frame,
                args.root,
                k=args.spatial_neighbors,
                density=args.density,
                min_mass=args.min_mass,
            )
        )
    pd.concat(projections, ignore_index=True).to_csv(
        args.root / "cellrank_sparse_spatial_projection_cells.csv.gz", index=False
    )
    print(f"Saved sparse CellRank spatial projections to {args.root}")


if __name__ == "__main__":
    main()
