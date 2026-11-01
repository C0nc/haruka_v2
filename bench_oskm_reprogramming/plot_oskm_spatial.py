#!/usr/bin/env python
"""Plot condition-only tissue maps for OSKM salient discoveries."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def categorical(axis, xy, labels, title):
    values = pd.Categorical(labels)
    colors = sns.color_palette("husl", len(values.categories))
    for i, category in enumerate(values.categories):
        keep = values.codes == i
        axis.scatter(
            xy[keep, 0], xy[keep, 1], s=1, color=colors[i],
            label=str(category), alpha=0.8, rasterized=True,
        )
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    scores = pd.read_csv(
        args.analysis / "oskm_cell_scores.csv.gz", index_col=0
    )
    target = data.obs["condition"].astype(str).to_numpy() == "oskm"
    condition = data[target].copy()
    scores = scores.reindex(condition.obs_names.astype(str))
    if scores["s_cluster"].isna().any():
        raise ValueError("Cell-score rows do not align to the condition AnnData")
    mouse_values = sorted(condition.obs["graph_id"].astype(str).unique())
    fig, axes = plt.subplots(4, len(mouse_values), figsize=(5 * len(mouse_values), 16))
    response_limits = np.nanpercentile(
        scores["response_heterogeneity"], [2, 98]
    )
    for column, mouse in enumerate(mouse_values):
        keep = condition.obs["graph_id"].astype(str).to_numpy() == mouse
        xy = np.asarray(condition.obsm["spatial"])[keep]
        obs = condition.obs.iloc[np.flatnonzero(keep)]
        score = scores.iloc[np.flatnonzero(keep)]
        categorical(axes[0, column], xy, obs["subregion"], f"{mouse}: subregion")
        categorical(axes[1, column], xy, obs["cell_type"], f"{mouse}: cell type")
        categorical(axes[2, column], xy, score["s_cluster"], f"{mouse}: s-CC cluster")
        points = axes[3, column].scatter(
            xy[:, 0], xy[:, 1], c=score["response_heterogeneity"], s=1,
            cmap="coolwarm", vmin=response_limits[0], vmax=response_limits[1],
            alpha=0.8, rasterized=True,
        )
        axes[3, column].set_title(f"{mouse}: youngward response", fontsize=9)
        axes[3, column].set_aspect("equal")
        axes[3, column].set_xticks([])
        axes[3, column].set_yticks([])
    fig.colorbar(points, ax=axes[3, :].tolist(), fraction=0.012, pad=0.01)
    fig.suptitle("OSKM condition tissue maps (all four treated mice)", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.98, 0.99))
    output = args.analysis / "figures" / "oskm_condition_spatial_maps.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
