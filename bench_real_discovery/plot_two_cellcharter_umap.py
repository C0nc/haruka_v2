"""Plot Haruka CellCharter background/salient UMAPs with slice audits."""

from __future__ import annotations

import argparse
import json
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
    if len(groups) <= maximum:
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    levels = np.unique(groups)
    quota = max(1, maximum // len(levels))
    chosen = []
    for level in levels:
        idx = np.flatnonzero(groups == level)
        chosen.append(rng.choice(idx, min(quota, len(idx)), replace=False))
    chosen = np.concatenate(chosen)
    if len(chosen) < maximum:
        left = np.setdiff1d(np.arange(len(groups)), chosen, assume_unique=False)
        chosen = np.concatenate(
            [chosen, rng.choice(left, min(maximum - len(chosen), len(left)), replace=False)]
        )
    return np.sort(chosen[:maximum])


def umap_xy(x: np.ndarray, seed: int) -> np.ndarray:
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float32))
    return UMAP(
        n_neighbors=30, min_dist=0.25, metric="euclidean",
        random_state=seed, n_jobs=1,
    ).fit_transform(x)


def plaque_class(obs: pd.DataFrame) -> np.ndarray:
    tau = obs["plaque_tau"].to_numpy(dtype=int) > 0
    abeta = obs["plaque_abeta"].to_numpy(dtype=int) > 0
    label = np.full(len(obs), "No plaque", dtype=object)
    label[tau & ~abeta] = "p-tau only"
    label[~tau & abeta] = "Aβ only"
    label[tau & abeta] = "Aβ + p-tau"
    return label.astype(str)


def scatter(ax, xy: np.ndarray, values: np.ndarray, muted_level: str | None = None) -> None:
    values = np.asarray(values).astype(str)
    levels = sorted(np.unique(values))
    colors = sns.color_palette("husl", max(len(levels), 1))
    palette = dict(zip(levels, colors))
    if muted_level in palette:
        palette[muted_level] = (0.76, 0.76, 0.76)
        levels.remove(muted_level)
        levels.insert(0, muted_level)
    for level in levels:
        take = values == level
        muted = level == muted_level
        ax.scatter(
            xy[take, 0], xy[take, 1],
            s=2.0 if muted else 4.5,
            alpha=0.28 if muted else 0.72,
            linewidths=0, rasterized=True,
            color=palette[level], label=f"{level} (n={take.sum():,})",
        )
    ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, fontsize=6.5, markerscale=2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--study", choices=["ad", "spacseq"], required=True)
    parser.add_argument(
        "--spacseq-background-key",
        default="niche",
        help="SPARC annotation used to color the background embedding.",
    )
    parser.add_argument(
        "--ad-background-key",
        default="tissue_domain",
        help="AD annotation used to color the background embedding.",
    )
    parser.add_argument("--max-background", type=int, default=15000)
    parser.add_argument("--max-salient", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--background-target-only",
        action="store_true",
        help="Fit and draw the background UMAP using condition/target cells only",
    )
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="Write a one-row background-only audit figure",
    )
    args = parser.parse_args()

    backed = ad.read_h5ad(args.data, backed="r")
    obs_all = backed.obs.copy()
    names = np.load(Path(args.emb) / "obs_names.npy", allow_pickle=True).astype(str)
    missing = pd.Index(names).difference(obs_all.index.astype(str))
    if len(missing):
        raise KeyError(f"Embedding cells absent from data: {missing[:5].tolist()}")
    obs = obs_all.loc[names].copy()
    if "slice_id" not in obs:
        raise KeyError("Plot requires obs['slice_id']")
    target = obs["condition"].astype(str).to_numpy() == "disease"

    background = np.load(Path(args.emb) / "spatial.npy").astype(np.float32)
    salient = np.load(Path(args.emb) / "salient_spatial.npy").astype(np.float32)
    meta_path = Path(args.emb) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    penalty = meta.get("zs_independence_penalty")
    lambda_label = f"λ={float(penalty):g}" if penalty is not None else "λ=?"
    if len(background) != len(obs) or len(salient) != len(obs):
        raise ValueError("Embedding and observation row counts differ")

    if args.study == "ad":
        if args.ad_background_key not in obs:
            raise KeyError(f"Missing AD background annotation: {args.ad_background_key}")
        bg_label = obs[args.ad_background_key].astype(str).to_numpy()
        sal_label = plaque_class(obs)
        bg_name, sal_name = args.ad_background_key.replace("_", " "), "plaque class"
        title = "Jill AD"
        muted = "No plaque"
    else:
        if args.spacseq_background_key not in obs:
            raise KeyError(f"Missing SPARC background annotation: {args.spacseq_background_key}")
        bg_label = obs[args.spacseq_background_key].astype(str).to_numpy()
        sal_label = obs["perturbation"].astype(str).to_numpy()
        bg_name, sal_name = args.spacseq_background_key.replace("_", " "), "targeting sgRNA"
        title = "SPARC-seq"
        muted = None

    slices = obs["slice_id"].astype(str).to_numpy()
    target_idx = np.flatnonzero(target)
    bg_pool = target_idx if args.background_target_only else np.arange(len(obs))
    bg_idx = bg_pool[
        balanced_sample(slices[bg_pool], args.max_background, args.seed)
    ]
    sal_idx = target_idx[
        balanced_sample(slices[target_idx], args.max_salient, args.seed + 1)
    ]
    bg_xy = umap_xy(background[bg_idx], args.seed)
    sal_xy = umap_xy(salient[sal_idx], args.seed + 1)
    bg_arm = "target only" if args.background_target_only else "all cells"

    sns.set_theme(style="white")
    if args.background_only:
        fig, axes = plt.subplots(1, 3, figsize=(23.5, 6.0), constrained_layout=True)
        scatter(axes[0], bg_xy, bg_label[bg_idx])
        scatter(axes[1], bg_xy, sal_label[bg_idx], muted)
        scatter(axes[2], bg_xy, slices[bg_idx])
        arm = "target only" if args.background_target_only else "all cells"
        axes[0].set_title(f"Background CellCharter — {bg_name} ({arm})")
        axes[1].set_title(f"Same background geometry — {sal_name}")
        axes[2].set_title("Same background geometry — slice_id")
        fig.suptitle(f"{title}: Haruka {lambda_label} background ({arm})", fontsize=15)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=240, bbox_inches="tight")
        plt.close(fig)
        bg_table = pd.DataFrame({
            "obs_name": obs.index.to_numpy()[bg_idx],
            "representation": "background_cellcharter",
            "included_arm": "target_only" if args.background_target_only else "all",
            "background_label": bg_label[bg_idx],
            "condition_label": sal_label[bg_idx],
            "slice_id": slices[bg_idx],
            "UMAP1": bg_xy[:, 0], "UMAP2": bg_xy[:, 1],
        })
        coords = out.with_name(f"{out.stem}_coordinates.csv")
        bg_table.to_csv(coords, index=False)
        print(f"[save] {out}")
        print(f"[save] {coords}")
        print(f"[audit] background={len(bg_idx)} {arm} cells; graph boundary=slice_id")
        return

    fig, axes = plt.subplots(2, 3, figsize=(23.5, 12.0), constrained_layout=True)
    scatter(axes[0, 0], bg_xy, bg_label[bg_idx])
    scatter(axes[0, 1], bg_xy, sal_label[bg_idx], muted)
    scatter(axes[0, 2], bg_xy, slices[bg_idx])
    scatter(axes[1, 0], sal_xy, bg_label[sal_idx])
    scatter(axes[1, 1], sal_xy, sal_label[sal_idx], muted)
    scatter(axes[1, 2], sal_xy, slices[sal_idx])
    axes[0, 0].set_title(f"Background CellCharter — {bg_name} ({bg_arm})")
    axes[0, 1].set_title(f"Same background geometry — {sal_name}")
    axes[0, 2].set_title("Same background geometry — slice_id")
    axes[1, 0].set_title(f"Salient CellCharter — {bg_name} (target only)")
    axes[1, 1].set_title(f"Same salient geometry — {sal_name}")
    axes[1, 2].set_title("Same salient geometry — slice_id")
    fig.suptitle(f"{title}: Haruka {lambda_label}, slice-bounded CellCharter", fontsize=15)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)

    bg_table = pd.DataFrame({
        "obs_name": obs.index.to_numpy()[bg_idx],
        "representation": "background_cellcharter",
        "included_arm": "target_only" if args.background_target_only else "all",
        "background_label": bg_label[bg_idx],
        "condition_label": sal_label[bg_idx],
        "slice_id": slices[bg_idx],
        "UMAP1": bg_xy[:, 0], "UMAP2": bg_xy[:, 1],
    })
    sal_table = pd.DataFrame({
        "obs_name": obs.index.to_numpy()[sal_idx],
        "representation": "salient_cellcharter",
        "included_arm": "target_only",
        "background_label": bg_label[sal_idx],
        "condition_label": sal_label[sal_idx],
        "slice_id": slices[sal_idx],
        "UMAP1": sal_xy[:, 0], "UMAP2": sal_xy[:, 1],
    })
    coords = out.with_name(f"{out.stem}_coordinates.csv")
    pd.concat([bg_table, sal_table], ignore_index=True).to_csv(coords, index=False)
    print(f"[save] {out}")
    print(f"[save] {coords}")
    print(
        f"[audit] background={len(bg_idx)} {bg_arm}; "
        f"salient={len(sal_idx)} target cells; graph boundary=slice_id"
    )


if __name__ == "__main__":
    main()
