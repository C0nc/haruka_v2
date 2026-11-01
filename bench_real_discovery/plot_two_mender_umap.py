"""Plot only the background and salient MENDER representations.

The background panel uses all cells.  The salient panel excludes the control
arm before subsampling, scaling, and fitting UMAP, so control cells cannot
affect either the displayed point cloud or the salient UMAP geometry.
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


def balanced_sample(groups: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Return a reproducible, approximately group-balanced subset."""
    if len(groups) <= maximum:
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    levels = np.unique(groups)
    quota = max(1, maximum // len(levels))
    selected = []
    for level in levels:
        idx = np.flatnonzero(groups == level)
        selected.append(rng.choice(idx, min(quota, len(idx)), replace=False))
    chosen = np.concatenate(selected)
    if len(chosen) < maximum:
        remaining = np.setdiff1d(np.arange(len(groups)), chosen, assume_unique=False)
        extra = rng.choice(remaining, min(maximum - len(chosen), len(remaining)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen[:maximum])


def embed(z: np.ndarray, seed: int) -> np.ndarray:
    z = StandardScaler().fit_transform(z)
    return UMAP(
        n_neighbors=30,
        min_dist=0.25,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(z)


def plaque_class(obs: pd.DataFrame) -> np.ndarray:
    tau = obs["plaque_tau"].to_numpy(dtype=int) > 0
    abeta = obs["plaque_abeta"].to_numpy(dtype=int) > 0
    result = np.full(len(obs), "No plaque", dtype=object)
    result[tau & ~abeta] = "p-tau only"
    result[~tau & abeta] = "Aβ only"
    result[tau & abeta] = "Aβ + p-tau"
    return result.astype(str)


def scatter_categories(ax, xy: np.ndarray, values: np.ndarray, highlight_none: bool = False) -> None:
    values = np.asarray(values).astype(str)
    levels = sorted(np.unique(values))
    if highlight_none and "No plaque" in levels:
        levels.remove("No plaque")
        levels.insert(0, "No plaque")
    colors = sns.color_palette("husl", max(1, len(levels)))
    palette = dict(zip(levels, colors))
    if "No plaque" in palette:
        palette["No plaque"] = (0.78, 0.78, 0.78)
    for level in levels:
        take = values == level
        muted = level == "No plaque"
        ax.scatter(
            xy[take, 0],
            xy[take, 1],
            s=2.2 if muted else 5.0,
            alpha=0.30 if muted else 0.76,
            linewidths=0,
            rasterized=True,
            color=palette[level],
            label=f"{level} (n={int(take.sum()):,})",
            zorder=1 if muted else 2,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=7,
        markerscale=2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--study", choices=["ad", "spacseq"], required=True)
    parser.add_argument("--max-background", type=int, default=15000)
    parser.add_argument("--max-salient", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    backed = ad.read_h5ad(args.data, backed="r")
    obs_all = backed.obs.copy()
    names = np.load(Path(args.emb) / "obs_names.npy", allow_pickle=True).astype(str)
    missing = pd.Index(names).difference(obs_all.index.astype(str))
    if len(missing):
        raise KeyError(f"Embedding cells absent from data: {missing[:5].tolist()}")
    obs = obs_all.loc[names].copy()
    condition = obs["condition"].astype(str).to_numpy()
    target = condition != "control"

    background = np.load(Path(args.emb) / "spatial.npy").astype(np.float32)
    salient = np.load(Path(args.emb) / "salient_spatial.npy").astype(np.float32)
    if len(background) != len(obs) or len(salient) != len(obs):
        raise ValueError("Embedding and observation row counts differ")

    if args.study == "ad":
        background_label = obs["tissue_domain"].astype(str).to_numpy()
        salient_label = plaque_class(obs)
        background_title = "Background MENDER: tissue domain (all cells)"
        salient_title = "Salient MENDER: plaque class (AD cells only)"
        study_title = "Jill AD: background and target-only salient spatial representations"
    else:
        background_label = obs["niche"].astype(str).to_numpy()
        salient_label = obs["perturbation"].astype(str).to_numpy()
        background_title = "Background MENDER: tissue niche (all cells)"
        salient_title = "Salient MENDER: targeting sgRNA (target cells only)"
        study_title = "SPARC-seq: background and target-only salient spatial representations"

    group_key = "graph_id" if "graph_id" in obs else "condition"
    bg_idx = balanced_sample(
        obs[group_key].astype(str).to_numpy(), args.max_background, args.seed
    )
    target_idx = np.flatnonzero(target)
    sal_local = balanced_sample(
        obs.iloc[target_idx][group_key].astype(str).to_numpy(),
        args.max_salient,
        args.seed + 1,
    )
    sal_idx = target_idx[sal_local]

    bg_xy = embed(background[bg_idx], args.seed)
    sal_xy = embed(salient[sal_idx], args.seed)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.1), constrained_layout=True)
    scatter_categories(axes[0], bg_xy, background_label[bg_idx])
    scatter_categories(
        axes[1], sal_xy, salient_label[sal_idx], highlight_none=args.study == "ad"
    )
    axes[0].set_title(background_title)
    axes[1].set_title(salient_title)
    fig.suptitle(study_title, fontsize=14)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    bg_table = pd.DataFrame(
        {
            "obs_name": obs.index.to_numpy()[bg_idx],
            "representation": "background_mender",
            "included_arm": "all",
            "label": background_label[bg_idx],
            "UMAP1": bg_xy[:, 0],
            "UMAP2": bg_xy[:, 1],
        }
    )
    sal_table = pd.DataFrame(
        {
            "obs_name": obs.index.to_numpy()[sal_idx],
            "representation": "salient_mender",
            "included_arm": "target_only",
            "label": salient_label[sal_idx],
            "UMAP1": sal_xy[:, 0],
            "UMAP2": sal_xy[:, 1],
        }
    )
    coords_out = out.with_name(f"{out.stem}_coordinates.csv")
    pd.concat([bg_table, sal_table], ignore_index=True).to_csv(coords_out, index=False)
    print(f"[save] {out}")
    print(f"[save] {coords_out}")
    print(
        f"[audit] background={len(bg_idx)} all-arm cells; "
        f"salient={len(sal_idx)} target cells; salient controls=0"
    )


if __name__ == "__main__":
    main()
