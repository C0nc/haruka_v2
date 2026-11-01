"""UMAP panels for symmetric local Haruka CellCharter readouts."""

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


def embed(x: np.ndarray, seed: int) -> np.ndarray:
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float32))
    return UMAP(
        n_neighbors=30,
        min_dist=0.25,
        metric="cosine",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(x)


def scatter(axis, xy: np.ndarray, labels: np.ndarray, title: str, mute: str | None = None):
    labels = np.asarray(labels).astype(str)
    levels = sorted(pd.unique(labels))
    palette = dict(zip(levels, sns.color_palette("husl", len(levels))))
    if mute in levels:
        palette[mute] = (0.78, 0.78, 0.78)
        levels.remove(mute)
        levels.insert(0, mute)
    for level in levels:
        take = labels == level
        is_muted = level == mute
        axis.scatter(
            xy[take, 0], xy[take, 1],
            s=2.0 if is_muted else 4.0,
            alpha=0.25 if is_muted else 0.70,
            linewidths=0,
            rasterized=True,
            color=palette[level],
            label=f"{level} (n={take.sum():,})",
        )
    axis.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
    axis.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
        fontsize=6.2, markerscale=2.2, ncol=1,
    )


def aligned_obs(data: Path, emb: Path) -> pd.DataFrame:
    backed = ad.read_h5ad(data, backed="r")
    names = np.load(emb / "obs_names.npy", allow_pickle=True).astype(str)
    missing = pd.Index(names).difference(backed.obs_names.astype(str))
    if len(missing):
        raise ValueError(f"Embedding cells absent from data: {missing[:5].tolist()}")
    return backed.obs.loc[names].copy()


def plot_spacseq(args: argparse.Namespace) -> None:
    obs = aligned_obs(args.data, args.emb)
    target = obs["condition"].astype(str).to_numpy() == args.target_label
    label_key = "perturbation" if "perturbation" in obs else "guide_name"
    labels = obs[label_key].astype(str).to_numpy()[target]
    z = np.load(args.emb / "spatial.npy")[target]
    s = np.load(args.emb / "salient_spatial.npy")[target]
    z_xy, s_xy = embed(z, args.seed), embed(s, args.seed + 1)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), constrained_layout=True)
    scatter(axes[0], z_xy, labels, "Background z-CC — targeting sgRNA")
    scatter(axes[1], s_xy, labels, "Salient s-CC — targeting sgRNA")
    fig.suptitle(
        "SPARC-seq condition cells: symmetric local 15-neighbor CellCharter",
        fontsize=14,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    pd.concat([
        pd.DataFrame({"obs_name": obs.index[target], "representation": "z_cc", "label": labels, "UMAP1": z_xy[:, 0], "UMAP2": z_xy[:, 1]}),
        pd.DataFrame({"obs_name": obs.index[target], "representation": "s_cc", "label": labels, "UMAP1": s_xy[:, 0], "UMAP2": s_xy[:, 1]}),
    ]).to_csv(args.out.with_name(f"{args.out.stem}_coordinates.csv"), index=False)


def plot_petracer(args: argparse.Namespace) -> None:
    if len(args.data) != len(args.emb):
        raise ValueError("PEtracer requires matching --data and --emb lists")
    rows = len(args.data)
    sns.set_theme(style="white")
    fig, axes = plt.subplots(rows, 4, figsize=(25, 5.6 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, 4)
    coordinate_tables = []
    for row, (data_path, emb_path) in enumerate(zip(args.data, args.emb)):
        obs = aligned_obs(data_path, emb_path)
        condition = obs["condition"].astype(str).to_numpy() == args.target_label
        if "lineage_effect_eligible" in obs:
            condition &= obs["lineage_effect_eligible"].astype(bool).to_numpy()
        family = obs["gt_lineage_family"].astype(str).to_numpy()[condition]
        clade = obs["gt_lineage_clade"].astype(str).to_numpy()[condition]
        section = str(obs["section_id"].iloc[0])
        z_xy = embed(np.load(emb_path / "spatial.npy")[condition], args.seed + row * 2)
        s_xy = embed(np.load(emb_path / "salient_spatial.npy")[condition], args.seed + row * 2 + 1)
        scatter(axes[row, 0], z_xy, family, f"{section}: background z-CC — lineage family")
        scatter(axes[row, 1], z_xy, clade, f"{section}: same z-CC — lineage clade")
        scatter(axes[row, 2], s_xy, family, f"{section}: salient s-CC — lineage family")
        scatter(axes[row, 3], s_xy, clade, f"{section}: same s-CC — lineage clade")
        names = obs.index.to_numpy()[condition]
        for representation, xy in (("z_cc", z_xy), ("s_cc", s_xy)):
            coordinate_tables.append(pd.DataFrame({
                "obs_name": names, "section": section,
                "representation": representation,
                "lineage_family": family, "lineage_clade": clade,
                "UMAP1": xy[:, 0], "UMAP2": xy[:, 1],
            }))
    fig.suptitle(
        "PEtracer lineage-eligible condition cells: symmetric local 15-neighbor CellCharter",
        fontsize=15,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.concat(coordinate_tables, ignore_index=True).to_csv(
        args.out.with_name(f"{args.out.stem}_coordinates.csv"), index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["spacseq", "petracer"], required=True)
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--emb", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-label", default="disease")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.study == "spacseq":
        if len(args.data) != 1 or len(args.emb) != 1:
            raise ValueError("SPARC-seq accepts exactly one --data and --emb")
        args.data, args.emb = args.data[0], args.emb[0]
        plot_spacseq(args)
    else:
        plot_petracer(args)
    print(f"[save] {args.out}", flush=True)
    print(f"[save] {args.out.with_name(f'{args.out.stem}_coordinates.csv')}", flush=True)


if __name__ == "__main__":
    main()
