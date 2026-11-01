"""Render one intended biological value for each Haruka latent slot."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def embedding(root, name, seed):
    x = np.load(root / f"{name}.npy")
    return UMAP(
        n_neighbors=30, min_dist=0.2, metric="euclidean", random_state=seed
    ).fit_transform(StandardScaler().fit_transform(x))


def categorical_panel(ax, xy, values, title, palette_name="tab10", grey=None):
    values = np.asarray(values).astype(str)
    levels = sorted(level for level in np.unique(values) if level != grey)
    palette = dict(zip(levels, sns.color_palette(palette_name, len(levels))))
    if grey is not None:
        mask = values == grey
        ax.scatter(xy[mask, 0], xy[mask, 1], s=2, color="#D9D9D9", alpha=.35,
                   linewidths=0, rasterized=True)
    for level in levels:
        mask = values == level
        ax.scatter(xy[mask, 0], xy[mask, 1], s=3, color=palette[level], alpha=.72,
                   linewidths=0, rasterized=True)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=palette[level], markeredgewidth=0, label=level)
        for level in levels
    ]
    ax.legend(handles=handles, title=title, loc="upper left",
              bbox_to_anchor=(0, -0.07), ncol=min(4, len(levels)),
              fontsize=7, title_fontsize=8, frameon=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--responder_cells", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.emb_dir)
    adata = ad.read_h5ad(args.data)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    response = pd.read_csv(args.responder_cells).set_index("obs_name")
    response = response.reindex(adata.obs_names)
    target_anchor = (
        adata.obs.is_anchor.to_numpy(bool)
        & (adata.obs.function.astype(str).to_numpy() != "NT")
    )
    coords = {
        name: embedding(root, name, args.seed)
        for name in ("background", "spatial", "salient", "salient_spatial")
    }

    sns.set_theme(style="white", context="notebook")
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.8), constrained_layout=True)

    cell_type = adata.obs.cell_type.astype(str).to_numpy()
    categorical_panel(
        axes[0], coords["background"], cell_type,
        "cell type", "tab10", grey="Unknown",
    )
    axes[0].set_title("Background intrinsic\ncell identity")

    niche = adata.obs.niche.astype(str).to_numpy()
    categorical_panel(
        axes[1], coords["spatial"], niche,
        "tissue / niche", "Set2", grey="Unknown",
    )
    axes[1].set_title("Background spatial\ntissue structure")

    xy = coords["salient"]
    axes[2].scatter(xy[:, 0], xy[:, 1], s=2, color="#D9D9D9", alpha=.25,
                    linewidths=0, rasterized=True)
    score = response.intrinsic_response_percentile.to_numpy(float)
    valid = target_anchor & np.isfinite(score)
    points = axes[2].scatter(
        xy[valid, 0], xy[valid, 1], c=score[valid], cmap="viridis",
        vmin=0, vmax=1, s=9, alpha=.9, linewidths=0, rasterized=True,
    )
    responder = response.responder.fillna(False).to_numpy(bool) & target_anchor
    axes[2].scatter(
        xy[responder, 0], xy[responder, 1], facecolors="none",
        edgecolors="#D62728", linewidths=.6, s=20, rasterized=True,
    )
    colorbar = fig.colorbar(points, ax=axes[2], fraction=.046, pad=.03)
    colorbar.set_label("intrinsic response percentile")
    axes[2].set_title("Salient intrinsic\nresponder heterogeneity")

    xy = coords["salient_spatial"]
    axes[3].scatter(xy[:, 0], xy[:, 1], s=2, color="#D9D9D9", alpha=.25,
                    linewidths=0, rasterized=True)
    function = adata.obs.function.astype(str).to_numpy()
    levels = sorted(np.unique(function[target_anchor]))
    palette = dict(zip(levels, sns.color_palette("husl", len(levels))))
    for level in levels:
        take = target_anchor & (function == level)
        axes[3].scatter(
            xy[take, 0], xy[take, 1], s=8, color=palette[level], alpha=.85,
            linewidths=0, rasterized=True,
        )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=palette[level], markeredgewidth=0, label=level)
        for level in levels
    ]
    axes[3].legend(handles=handles, title="target function", loc="upper left",
                   bbox_to_anchor=(0, -0.07), ncol=4, fontsize=7,
                   title_fontsize=8, frameon=False)
    axes[3].set_title("Salient spatial\nperturbation function")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.suptitle(
        "Haruka four biological representations (batch key omitted from display)",
        fontsize=15,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {output}")


if __name__ == "__main__":
    main()
