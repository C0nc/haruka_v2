"""Plot guide and slice organization in each saved Haruka representation."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    emb_dir = Path(args.emb_dir)
    order = np.load(emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    obs = adata.obs
    mask = obs.is_anchor.to_numpy(bool) & (obs.perturbation.astype(str).to_numpy() != "NT")
    guides = obs.perturbation.astype(str).to_numpy()[mask]
    slices = obs.slice_id.astype(str).to_numpy()[mask]
    names = ("background", "spatial", "salient", "salient_spatial")
    titles = ("Background intrinsic", "Background spatial", "Salient intrinsic", "Salient spatial")
    guide_levels = sorted(np.unique(guides))
    slice_levels = sorted(np.unique(slices))
    guide_palette = dict(zip(guide_levels, sns.color_palette("husl", len(guide_levels))))
    slice_palette = dict(zip(slice_levels, sns.color_palette("Set1", len(slice_levels))))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for column, (name, title) in enumerate(zip(names, titles)):
        embedding = np.load(emb_dir / f"{name}.npy")[mask]
        xy = UMAP(
            n_neighbors=30, min_dist=0.2, metric="euclidean",
            random_state=args.seed,
        ).fit_transform(StandardScaler().fit_transform(embedding))
        for level in guide_levels:
            take = guides == level
            axes[0, column].scatter(
                xy[take, 0], xy[take, 1], s=6, alpha=0.75,
                color=guide_palette[level], label=level, linewidths=0,
            )
        for level in slice_levels:
            take = slices == level
            axes[1, column].scatter(
                xy[take, 0], xy[take, 1], s=6, alpha=0.75,
                color=slice_palette[level], label=level, linewidths=0,
            )
        axes[0, column].set_title(title)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_xlabel("UMAP 1")
            axes[row, column].set_ylabel("UMAP 2")

    axes[0, 0].legend(
        title="targeting sgRNA", bbox_to_anchor=(0, 1.20), loc="lower left",
        ncol=7, fontsize=7, title_fontsize=8, frameon=False,
    )
    axes[1, 0].legend(
        title="slice", bbox_to_anchor=(0, -0.14), loc="upper left",
        ncol=3, fontsize=8, title_fontsize=8, frameon=False,
    )
    fig.supylabel("Top: sgRNA class        Bottom: cancer slice", fontsize=11)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {output}")


if __name__ == "__main__":
    main()
