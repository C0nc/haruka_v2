"""Discover what condition-only PEtracer s-CC clusters encode."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


CATEGORICAL = {
    "lineage_family": "gt_lineage_family",
    "lineage_clade": "gt_lineage_clade",
    "lineage_eligible": "lineage_effect_eligible",
    "cell_subtype": "gt_background_celltype",
    "tissue_niche": "gt_background_niche",
    "boundary_bin": "gt_background_boundary_bin",
    "within_tumor": "within_tumor",
    "cell_type_coarse": "type",
    "fov": "fov",
}

CONTINUOUS = [
    "total_counts",
    "n_genes_by_counts",
    "volume",
    "total_density",
    "tumor_boundary_dist",
    "lung_boundary_dist",
    "detection_rate",
    "fitness",
    "character_dist_of_relatives",
    "local_character_diversity",
    "paired_change_fraction",
    "s_latent_norm",
]


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if len(values) < 2 or np.var(values) <= 1e-12:
        return float("nan")
    grand = values.mean()
    between = sum(np.sum(groups == group) * (values[groups == group].mean() - grand) ** 2 for group in np.unique(groups))
    return float(between / np.sum((values - grand) ** 2))


def paired_change(adata: ad.AnnData, order: np.ndarray, target: np.ndarray) -> np.ndarray:
    obs = adata.obs.iloc[order]
    pair = obs["pair_id"].astype(str).to_numpy()
    control_rows = np.flatnonzero(~target)
    target_rows = np.flatnonzero(target)
    control_by_pair = pd.Series(control_rows, index=pair[control_rows])
    matched_control = control_by_pair.loc[pair[target_rows]].to_numpy()
    counts = adata.layers["counts"][order]
    delta = counts[target_rows] - counts[matched_control]
    l1 = np.asarray(np.abs(delta).sum(axis=1)).ravel() if sp.issparse(delta) else np.abs(delta).sum(axis=1)
    total = np.asarray(counts[target_rows].sum(axis=1)).ravel()
    return l1 / np.maximum(total, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--emb-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    names = np.load(args.emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Index(adata.obs_names.astype(str)).get_indexer(names)
    if (order < 0).any():
        raise ValueError("Embedding observations do not align to AnnData")
    obs = adata.obs.iloc[order].copy()
    target = obs["condition"].astype(str).to_numpy() == "disease"
    target_obs = obs.loc[target].copy()
    s_latent = np.load(args.emb_dir / "salient.npy")[target]
    s_cc = np.load(args.emb_dir / "salient_spatial.npy")[target]
    target_obs["paired_change_fraction"] = paired_change(adata, order, target)
    target_obs["s_latent_norm"] = np.linalg.norm(s_latent, axis=1)

    d = s_latent.shape[1]
    representations = {"s_cc": s_cc, "s_latent": s_latent}
    for hop in range(s_cc.shape[1] // d):
        representations[f"s_cc_hop{hop + 1}"] = s_cc[:, hop * d : (hop + 1) * d]

    association_rows = []
    predictions = {}
    for rep_name, values in representations.items():
        variable = np.var(values, axis=0) > 1e-10
        pred = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed).fit_predict(values[:, variable])
        predictions[rep_name] = pred
        for label, column in CATEGORICAL.items():
            truth = target_obs[column].astype(str).to_numpy()
            association_rows.append({
                "section": str(target_obs["section_id"].iloc[0]),
                "representation": rep_name,
                "annotation": label,
                "association": "nmi",
                "score": normalized_mutual_info_score(truth, pred),
                "ari": adjusted_rand_score(truth, pred),
            })
        for column in CONTINUOUS:
            association_rows.append({
                "section": str(target_obs["section_id"].iloc[0]),
                "representation": rep_name,
                "annotation": column,
                "association": "eta_squared",
                "score": eta_squared(target_obs[column].to_numpy(), pred),
                "ari": np.nan,
            })

    cluster = predictions["s_cc"]
    target_obs["s_cc_cluster"] = cluster.astype(str)
    assignments = target_obs[[
        "section_id", "pair_id", "s_cc_cluster", "gt_lineage_family",
        "gt_lineage_clade", "gt_background_celltype", "gt_background_niche",
        "lineage_effect_eligible", "paired_change_fraction", "s_latent_norm",
    ]].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.out_dir / "s_cc_cluster_assignments.csv")
    pd.DataFrame(association_rows).to_csv(args.out_dir / "salient_associations.csv", index=False)

    profiles = []
    for group in np.unique(cluster):
        take = cluster == group
        row = {
            "section": str(target_obs["section_id"].iloc[0]),
            "cluster": int(group),
            "n_cells": int(take.sum()),
            "lineage_eligible_fraction": float(target_obs.loc[take, "lineage_effect_eligible"].astype(bool).mean()),
            "paired_change_fraction_mean": float(target_obs.loc[take, "paired_change_fraction"].mean()),
            "s_latent_norm_mean": float(target_obs.loc[take, "s_latent_norm"].mean()),
        }
        for label, column in (("celltype", "gt_background_celltype"), ("niche", "gt_background_niche"), ("family", "gt_lineage_family")):
            counts = target_obs.loc[take, column].astype(str).value_counts(normalize=True)
            row[f"top_{label}"] = counts.index[0]
            row[f"top_{label}_fraction"] = float(counts.iloc[0])
        profiles.append(row)
    pd.DataFrame(profiles).to_csv(args.out_dir / "s_cc_cluster_profiles.csv", index=False)

    x = StandardScaler().fit_transform(s_cc)
    try:
        import umap
        xy = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=args.seed).fit_transform(x)
    except Exception:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=args.seed).fit_transform(x)
    coords = np.asarray(adata.obsm["spatial"])[order][target]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    panels = [
        (xy, cluster.astype(str), "s-CC clusters"),
        (xy, target_obs["gt_lineage_family"].astype(str).to_numpy(), "same UMAP — lineage family"),
        (xy, target_obs["gt_background_niche"].astype(str).to_numpy(), "same UMAP — tissue niche"),
        (coords, cluster.astype(str), "physical tissue — s-CC clusters"),
    ]
    for axis, (positions, labels, title) in zip(axes, panels):
        for value in np.unique(labels):
            take = labels == value
            if value == "not_lineage_target":
                axis.scatter(positions[take, 0], positions[take, 1], s=1, c="lightgray", alpha=0.25, rasterized=True)
            else:
                axis.scatter(positions[take, 0], positions[take, 1], s=2.5, alpha=0.65, label=value, rasterized=True)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if len(np.unique(labels)) <= 20:
            axis.legend(fontsize=6, markerscale=3, frameon=False, ncol=2)
    fig.suptitle(f"PEtracer {target_obs['section_id'].iloc[0]}: what s-CC clusters encode")
    fig.tight_layout()
    fig.savefig(args.out_dir / "s_cc_cluster_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
