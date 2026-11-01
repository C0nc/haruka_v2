"""Descriptive four-slot UMAPs for the clean real-data Haruka suite.

The cross-specimen probes produced by ``score_haruka_study.py`` are the
quantitative evaluation.  This script provides a standardized visual companion
without fitting classifiers or modifying the learned representations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP


SLOTS = ["background", "spatial", "salient", "salient_spatial"]
TITLES = [
    "Background intrinsic",
    "Background spatial",
    "Salient intrinsic",
    "Salient spatial",
]


def balanced_sample(groups: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Sample approximately equally across specimens for a readable plot."""
    rng = np.random.default_rng(seed)
    levels = np.unique(groups)
    quota = max(1, maximum // len(levels))
    selected = []
    for level in levels:
        idx = np.flatnonzero(groups == level)
        selected.append(rng.choice(idx, min(quota, len(idx)), replace=False))
    return np.sort(np.concatenate(selected))


def scatter_categories(ax, xy, values, palette_name="tab20", legend=True):
    values = np.asarray(values).astype(str)
    levels = sorted(np.unique(values))
    palette = dict(zip(levels, sns.color_palette(palette_name, len(levels))))
    for level in levels:
        mask = values == level
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=3, alpha=0.72,
            linewidths=0, rasterized=True, color=palette[level],
            label=level,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if legend:
        ncol = 1 if len(levels) <= 8 else 2
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
            fontsize=6, title_fontsize=7, markerscale=2.2, ncol=ncol,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--max-cells", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    emb_dir = Path(args.emb)
    names = np.load(emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Series(np.arange(data.n_obs), index=data.obs_names.astype(str)).loc[names].to_numpy()
    data = data[order].copy()
    reps = {slot: np.load(emb_dir / f"{slot}.npy") for slot in SLOTS}

    group_key = "specimen_id" if "specimen_id" in data.obs else "graph_id"
    groups = data.obs[group_key].astype(str).to_numpy()
    idx = balanced_sample(groups, min(args.max_cells, data.n_obs), args.seed)
    obs = data.obs.iloc[idx]

    expected_keys = ["cell_type", "tissue_domain", "stage", "stage"]
    missing = sorted(set(expected_keys + ["condition"]) - set(obs.columns))
    if missing:
        raise KeyError(f"Missing biological annotations: {missing}")

    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 4, figsize=(25, 11), constrained_layout=True)
    for column, (slot, title, expected_key) in enumerate(zip(SLOTS, TITLES, expected_keys)):
        scaled = StandardScaler().fit_transform(reps[slot][idx])
        xy = UMAP(
            n_neighbors=30, min_dist=0.25, metric="euclidean",
            random_state=args.seed, n_jobs=1,
        ).fit_transform(scaled)

        scatter_categories(axes[0, column], xy, obs[expected_key].astype(str), legend=True)
        axes[0, column].set_title(f"{title}: {expected_key}")
        axes[0, column].get_legend().set_title(expected_key)

        scatter_categories(
            axes[1, column], xy, obs["condition"].astype(str),
            palette_name="Set1", legend=(column == 0),
        )
        axes[1, column].set_title(f"{title}: condition")
        if column == 0:
            axes[1, column].get_legend().set_title("condition")

    fig.suptitle(
        f"Clean Haruka four-slot biology: {args.study}\n"
        "Top: expected slot annotation; bottom: control/disease arm",
        fontsize=15,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {output}")


if __name__ == "__main__":
    main()
