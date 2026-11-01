#!/usr/bin/env python3
"""Plot spatially local vector fields along Haruka s-CC/CellRank DPT.

This is a visualization of local pseudotime flow, not RNA velocity.  Because
the CellRank state graph is built in s-CC space, its edges can connect distant
tissue locations.  We therefore estimate direction only among the 15 nearest
physical neighbors within each replicate and main biological cluster.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DEFAULT_ROOT = Path("/home/yancui/haruka_v2/ad_analysis/scc_dpt_cellrank2")


def local_dpt_velocity(group: pd.DataFrame, k: int = 15) -> np.ndarray:
    """Estimate a spatially local, forward-DPT vector at every cell."""
    xy = group[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    pt = group["dpt_pseudotime"].to_numpy(dtype=float)
    n_neighbors = min(k + 1, len(group))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(xy)
    dist, ind = nn.kneighbors(xy)
    dist, ind = dist[:, 1:], ind[:, 1:]
    delta_xy = xy[ind] - xy[:, None, :]
    delta_pt = pt[ind] - pt[:, None]
    sigma = np.maximum(np.median(dist, axis=1, keepdims=True), 1e-8)
    spatial_weight = np.exp(-0.5 * np.square(dist / sigma))
    forward_weight = np.maximum(delta_pt, 0.0)
    weight = spatial_weight * forward_weight
    denom = weight.sum(axis=1, keepdims=True)
    velocity = np.divide(
        (weight[..., None] * delta_xy).sum(axis=1),
        denom,
        out=np.zeros((len(group), 2), dtype=float),
        where=denom > 0,
    )
    # Keep only direction; magnitude is not a calibrated physical speed.
    norm = np.linalg.norm(velocity, axis=1, keepdims=True)
    return np.divide(velocity, norm, out=np.zeros_like(velocity), where=norm > 0)


def compute_cell_velocity(cells: pd.DataFrame, k: int) -> pd.DataFrame:
    parts = []
    for (_, _, _), loc in cells.groupby(["dataset", "replicate", "main_cluster"], sort=True).groups.items():
        take = np.asarray(list(loc), dtype=int)
        group = cells.iloc[take].copy()
        velocity = local_dpt_velocity(group, k=k)
        group["spatial_dpt_velocity_x"] = velocity[:, 0]
        group["spatial_dpt_velocity_y"] = velocity[:, 1]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def grid_velocity(group: pd.DataFrame, grid_n: int = 28, smooth_k: int = 60) -> pd.DataFrame:
    xy = group[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    vel = group[["spatial_dpt_velocity_x", "spatial_dpt_velocity_y"]].to_numpy(dtype=float)
    xlo, xhi = np.quantile(xy[:, 0], [0.005, 0.995])
    ylo, yhi = np.quantile(xy[:, 1], [0.005, 0.995])
    gx, gy = np.meshgrid(np.linspace(xlo, xhi, grid_n), np.linspace(ylo, yhi, grid_n))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    n_neighbors = min(smooth_k, len(group))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(xy)
    dist, ind = nn.kneighbors(grid)
    local_scale = np.maximum(np.median(dist, axis=1, keepdims=True), 1e-8)
    weight = np.exp(-0.5 * np.square(dist / local_scale))
    gv = (weight[..., None] * vel[ind]).sum(axis=1) / weight.sum(axis=1, keepdims=True)
    norm = np.linalg.norm(gv, axis=1)

    # Suppress grid points in tissue gaps and low-coherence locations.
    cell_nn = NearestNeighbors(n_neighbors=min(16, len(group))).fit(xy).kneighbors(xy)[0][:, -1]
    support_radius = 1.5 * np.median(cell_nn)
    keep = (dist[:, 0] <= support_radius) & (norm >= np.nanquantile(norm, 0.25))
    gv = np.divide(gv, norm[:, None], out=np.zeros_like(gv), where=norm[:, None] > 0)
    return pd.DataFrame(
        {
            "grid_x": grid[keep, 0],
            "grid_y": grid[keep, 1],
            "velocity_x": gv[keep, 0],
            "velocity_y": gv[keep, 1],
            "coherence": norm[keep],
        }
    )


def plot_dataset(dataset: str, cells: pd.DataFrame, out: Path, grid_n: int):
    reps = sorted(cells.replicate.unique())
    fig, axes = plt.subplots(1, len(reps), figsize=(8 * len(reps), 7.2), squeeze=False)
    grid_parts = []
    for ax, replicate in zip(axes[0], reps):
        group = cells[cells.replicate == replicate].copy()
        order = np.argsort(group.dpt_pseudotime.to_numpy())
        ordered = group.iloc[order]
        points = ax.scatter(
            ordered.spatial_x,
            ordered.spatial_y,
            c=ordered.dpt_pseudotime,
            cmap="magma",
            vmin=0,
            vmax=1,
            s=4 if dataset == "starmap" else 3,
            linewidths=0,
            rasterized=True,
        )
        grid = grid_velocity(group, grid_n=grid_n)
        grid.insert(0, "replicate", replicate)
        grid.insert(0, "dataset", dataset)
        grid_parts.append(grid)
        diag = np.hypot(group.spatial_x.max() - group.spatial_x.min(), group.spatial_y.max() - group.spatial_y.min())
        arrow_length = 0.018 * diag
        ax.quiver(
            grid.grid_x,
            grid.grid_y,
            grid.velocity_x * arrow_length,
            grid.velocity_y * arrow_length,
            color="white",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0032,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.2,
        )
        ax.set_title(f"{replicate}: local forward-DPT flow")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    fig.colorbar(points, ax=axes.ravel().tolist(), fraction=0.022, pad=0.015, label="Within-cell-class s-CC DPT")
    fig.suptitle(
        f"{dataset.upper()}: spatial pseudotime vector field from Haruka s-CC + CellRank 2\n"
        "arrows use only ~15 nearest physical neighbors within each biological cluster",
        y=1.01,
    )
    fig.savefig(out / f"{dataset}_spatial_dpt_velocity_arrows.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pd.concat(grid_parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--spatial-neighbors", type=int, default=15)
    parser.add_argument("--grid-n", type=int, default=28)
    args = parser.parse_args()
    cells = pd.read_csv(args.root / "cellrank_cell_scores.csv.gz", dtype={"cell_id": str, "main_cluster": str})
    cells = compute_cell_velocity(cells, k=args.spatial_neighbors)
    cells[
        [
            "dataset",
            "replicate",
            "main_cluster",
            "cell_id",
            "dpt_pseudotime",
            "spatial_dpt_velocity_x",
            "spatial_dpt_velocity_y",
        ]
    ].to_csv(args.root / "spatial_dpt_velocity_cells.csv.gz", index=False)
    grids = []
    for dataset, group in cells.groupby("dataset", sort=True):
        grids.append(plot_dataset(dataset, group, args.root, args.grid_n))
    pd.concat(grids, ignore_index=True).to_csv(args.root / "spatial_dpt_velocity_grid.csv", index=False)
    print(f"Saved spatial DPT velocity figures and tables to {args.root}")


if __name__ == "__main__":
    main()
