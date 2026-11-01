#!/usr/bin/env python
"""Biological discovery and condition-only visualization for OSKM Haruka."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def log_normalize(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32).copy()
    library = np.asarray(matrix.sum(1)).ravel()
    matrix = matrix.multiply((10_000 / np.maximum(library, 1))[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def stratum(obs: pd.DataFrame) -> np.ndarray:
    return (
        obs["cell_type"].astype(str) + "::" + obs["subregion"].astype(str)
    ).to_numpy()


def rejuvenation_scores(train, young, target):
    x = log_normalize(train.layers["counts"])
    y = log_normalize(young.layers["counts"])
    train_group = stratum(train.obs)
    young_group = stratum(young.obs)
    control = ~target
    score = np.full(train.n_obs, np.nan, dtype=np.float32)
    delta = np.zeros((target.sum(), train.n_vars), dtype=np.float32)
    target_rows = np.flatnonzero(target)
    target_lookup = np.full(train.n_obs, -1, dtype=int)
    target_lookup[target_rows] = np.arange(len(target_rows))
    used = []
    for group in np.unique(train_group[target]):
        ci = np.flatnonzero(control & (train_group == group))
        ti = np.flatnonzero(target & (train_group == group))
        yi = np.flatnonzero(young_group == group)
        if len(ci) < 20 or len(ti) < 20 or len(yi) < 20:
            continue
        old_mean = np.asarray(x[ci].mean(0)).ravel()
        young_mean = np.asarray(y[yi].mean(0)).ravel()
        direction = young_mean - old_mean
        denom = float(direction @ direction)
        if denom <= 1e-8:
            continue
        score[ti] = (
            np.asarray(x[ti].dot(direction)).ravel() - old_mean @ direction
        ) / denom
        delta[target_lookup[ti]] = x[ti].toarray() - old_mean
        used.append(group)
    # Rare NSC/neuroblast subregion strata can fall below the threshold after
    # proportional sampling. Score only still-missing cells against a cell-type
    # old/young direction; populated strata keep the stricter spatial score.
    train_celltype = train.obs["cell_type"].astype(str).to_numpy()
    young_celltype = young.obs["cell_type"].astype(str).to_numpy()
    for celltype in np.unique(train_celltype[target & ~np.isfinite(score)]):
        ci = np.flatnonzero(control & (train_celltype == celltype))
        ti = np.flatnonzero(
            target & (train_celltype == celltype) & ~np.isfinite(score)
        )
        yi = np.flatnonzero(young_celltype == celltype)
        if len(ci) < 10 or len(ti) < 5 or len(yi) < 10:
            continue
        old_mean = np.asarray(x[ci].mean(0)).ravel()
        young_mean = np.asarray(y[yi].mean(0)).ravel()
        direction = young_mean - old_mean
        denom = float(direction @ direction)
        if denom <= 1e-8:
            continue
        score[ti] = (
            np.asarray(x[ti].dot(direction)).ravel() - old_mean @ direction
        ) / denom
        delta[target_lookup[ti]] = x[ti].toarray() - old_mean
        used.append(f"celltype_fallback::{celltype}")
    return score, delta, used


def spatial_average(values, coords, graph, k=15):
    output = np.full(len(values), np.nan, dtype=np.float32)
    for sample in np.unique(graph):
        rows = np.flatnonzero(graph == sample)
        valid = rows[np.isfinite(values[rows])]
        if not len(valid):
            continue
        neighbors = min(k, len(valid))
        _, index = cKDTree(coords[valid]).query(coords[valid], k=neighbors)
        if neighbors == 1:
            index = index[:, None]
        output[valid] = np.nanmean(values[valid][index], axis=1)
    return output


def residualize(values, groups):
    output = values.copy()
    for group in np.unique(groups):
        rows = (groups == group) & np.isfinite(values)
        if rows.any():
            output[rows] -= np.nanmean(values[rows])
    return output


def cv_ridge(rep, response, mouse):
    valid = np.isfinite(response)
    rep, response, mouse = rep[valid], response[valid], mouse[valid]
    fold = GroupKFold(n_splits=min(4, len(np.unique(mouse))))
    predicted = np.full(len(response), np.nan)
    for training, testing in fold.split(rep, response, groups=mouse):
        model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 8))
        )
        model.fit(rep[training], response[training])
        predicted[testing] = model.predict(rep[testing])
    return {
        "r2": float(r2_score(response, predicted)),
        "spearman": float(stats.spearmanr(response, predicted).statistic),
        "n": int(len(response)),
    }


def umap_coordinates(rep, rows, seed, max_cells=0):
    import umap

    rng = np.random.default_rng(seed)
    if max_cells > 0 and len(rows) > max_cells:
        rows = np.sort(rng.choice(rows, max_cells, replace=False))
    values = StandardScaler().fit_transform(rep[rows])
    if values.shape[1] > 50:
        values = PCA(50, random_state=seed).fit_transform(values)
    xy = umap.UMAP(
        n_neighbors=30, min_dist=0.25, metric="cosine", random_state=seed
    ).fit_transform(values)
    return rows, xy


def scatter_panel(axis, xy, labels, title, continuous=False):
    if continuous:
        points = axis.scatter(
            xy[:, 0], xy[:, 1], c=labels, s=2, cmap="coolwarm",
            alpha=0.7, rasterized=True,
        )
        plt.colorbar(points, ax=axis, fraction=0.045, pad=0.02)
    else:
        values = pd.Categorical(labels)
        palette = sns.color_palette("husl", len(values.categories))
        for i, category in enumerate(values.categories):
            keep = values.codes == i
            axis.scatter(
                xy[keep, 0], xy[keep, 1], s=2, color=palette[i],
                label=str(category), alpha=0.7, rasterized=True,
            )
        axis.legend(markerscale=4, fontsize=6, frameon=False, loc="best")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--young", required=True, type=Path)
    parser.add_argument("--emb", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-umap-cells", type=int, default=0,
        help="Maximum plotted UMAP cells; 0 retains every condition cell.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)

    data = ad.read_h5ad(args.data)
    young = ad.read_h5ad(args.young)
    names = np.load(args.emb / "obs_names.npy", allow_pickle=True).astype(str)
    source_rows_path = args.emb / "source_rows.npy"
    source_rows = (
        np.load(source_rows_path).astype(np.int64)
        if source_rows_path.exists()
        else pd.Index(data.obs_names.astype(str)).get_indexer(names)
    )
    if (source_rows < 0).any() or not np.array_equal(
        data.obs_names[source_rows].astype(str).to_numpy(), names
    ):
        raise ValueError("Embedding observations do not align")
    z_cc = np.load(args.emb / "spatial.npy").astype(np.float32)
    s_cc = np.load(args.emb / "salient_spatial.npy").astype(np.float32)
    target_full = data.obs["condition"].astype(str).to_numpy() == "oskm"
    if not np.all(target_full[source_rows]):
        raise ValueError("Condition-only CellCharter input contains non-OSKM cells")
    mouse = data.obs.iloc[source_rows]["graph_id"].astype(str).to_numpy()
    groups = stratum(data.obs.iloc[source_rows])

    cell_score_full, expression_delta_full, used = rejuvenation_scores(
        data, young, target_full
    )
    cell_score = cell_score_full[source_rows]
    target_position = np.full(data.n_obs, -1, dtype=np.int64)
    target_position[np.flatnonzero(target_full)] = np.arange(target_full.sum())
    expression_delta = expression_delta_full[target_position[source_rows]]
    local_score = spatial_average(
        cell_score, np.asarray(data.obsm["spatial"])[source_rows], mouse, k=15
    )
    heterogeneity = residualize(local_score, groups)
    valid_target = np.isfinite(heterogeneity)

    scaled_s = StandardScaler().fit_transform(s_cc[valid_target])
    salient_cluster = MiniBatchKMeans(
        args.clusters, batch_size=4096, n_init=20, random_state=args.seed
    ).fit_predict(scaled_s)
    cluster_all = np.full(len(source_rows), -1, dtype=int)
    cluster_all[valid_target] = salient_cluster
    z_cluster = MiniBatchKMeans(
        args.clusters, batch_size=4096, n_init=20, random_state=args.seed
    ).fit_predict(StandardScaler().fit_transform(z_cc))

    target_obs = data.obs.iloc[source_rows].copy()
    target_obs["rejuvenation_score"] = cell_score
    target_obs["local_rejuvenation_score"] = local_score
    target_obs["response_heterogeneity"] = heterogeneity
    target_obs["s_cluster"] = cluster_all
    target_obs["z_cluster"] = z_cluster
    target_obs.to_csv(args.out / "oskm_cell_scores.csv.gz", compression="gzip")

    valid = np.isfinite(heterogeneity)
    metrics = {
        "design": {
            "training": "old_notreatment versus old_treatment",
            "young_reference": "evaluation only",
            "condition_only_readout": True,
            "response_strata": "cell_type x subregion",
            "response_neighborhood_k": 15,
        },
        "n_target": int(len(source_rows)),
        "n_response_scored": int(valid.sum()),
        "n_response_strata": len(used),
        "response_prediction_leave_mouse_out": {
            "z_cc": cv_ridge(z_cc[valid], heterogeneity[valid], mouse[valid]),
            "s_cc": cv_ridge(s_cc[valid], heterogeneity[valid], mouse[valid]),
        },
        "clustering": {
            "z_cc_celltype_ari": float(adjusted_rand_score(target_obs["cell_type"], z_cluster)),
            "z_cc_celltype_nmi": float(normalized_mutual_info_score(target_obs["cell_type"], z_cluster)),
            "z_cc_subregion_ari": float(adjusted_rand_score(target_obs["subregion"], z_cluster)),
            "z_cc_subregion_nmi": float(normalized_mutual_info_score(target_obs["subregion"], z_cluster)),
            "s_cc_celltype_ari": float(adjusted_rand_score(target_obs.loc[valid, "cell_type"], salient_cluster)),
            "s_cc_celltype_nmi": float(normalized_mutual_info_score(target_obs.loc[valid, "cell_type"], salient_cluster)),
            "s_cc_subregion_ari": float(adjusted_rand_score(target_obs.loc[valid, "subregion"], salient_cluster)),
            "s_cc_subregion_nmi": float(normalized_mutual_info_score(target_obs.loc[valid, "subregion"], salient_cluster)),
            "z_cc_slide_nmi": float(normalized_mutual_info_score(target_obs["paired_batch"], z_cluster)),
            "s_cc_slide_nmi": float(normalized_mutual_info_score(target_obs.loc[valid, "paired_batch"], salient_cluster)),
        },
    }

    summary = (
        target_obs[target_obs["s_cluster"] >= 0]
        .groupby("s_cluster", observed=True)
        .agg(
            n=("s_cluster", "size"),
            rejuvenation_mean=("local_rejuvenation_score", "mean"),
            heterogeneity_mean=("response_heterogeneity", "mean"),
            heterogeneity_sd=("response_heterogeneity", "std"),
        )
        .reset_index()
    )
    summary.to_csv(args.out / "salient_cluster_response_summary.csv", index=False)
    celltype_summary = (
        target_obs[np.isfinite(target_obs["local_rejuvenation_score"])]
        .groupby("cell_type", observed=True)
        .agg(
            n=("cell_type", "size"),
            rejuvenation_mean=("local_rejuvenation_score", "mean"),
            rejuvenation_median=("local_rejuvenation_score", "median"),
            heterogeneity_sd=("response_heterogeneity", "std"),
        )
        .sort_values("rejuvenation_median", ascending=False)
    )
    celltype_summary.to_csv(args.out / "celltype_rejuvenation_summary.csv")

    # Mean treatment-minus-old expression within response clusters.
    valid_target_positions = np.flatnonzero(valid)
    delta_valid = expression_delta[valid_target_positions]
    gene_rows = []
    for cluster in range(args.clusters):
        keep = salient_cluster == cluster
        effect = delta_valid[keep].mean(0)
        top_up = np.argsort(effect)[-10:][::-1]
        top_down = np.argsort(effect)[:10]
        for rank, gene_i in enumerate(top_up, 1):
            gene_rows.append((cluster, "up", rank, data.var_names[gene_i], effect[gene_i]))
        for rank, gene_i in enumerate(top_down, 1):
            gene_rows.append((cluster, "down", rank, data.var_names[gene_i], effect[gene_i]))
    pd.DataFrame(
        gene_rows, columns=["s_cluster", "direction", "rank", "gene", "mean_log_delta"]
    ).to_csv(args.out / "salient_cluster_genes.csv", index=False)

    # Categorical enrichments for interpretation and replicate-leakage diagnosis.
    for label in ("cell_type", "subregion", "paired_batch"):
        table = pd.crosstab(
            target_obs.loc[target_obs["s_cluster"] >= 0, "s_cluster"],
            target_obs.loc[target_obs["s_cluster"] >= 0, label],
        )
        expected = np.outer(table.sum(1), table.sum(0)) / table.values.sum()
        enrichment = np.log2((table + 1) / (expected + 1))
        enrichment.to_csv(args.out / f"salient_cluster_{label}_log2_enrichment.csv")

    with (args.out / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    # Condition-only cross-label UMAPs: the same labels are shown on both spaces.
    cc_rows = np.arange(len(source_rows), dtype=np.int64)
    plot_rows, z_xy = umap_coordinates(
        z_cc, cc_rows, args.seed, args.max_umap_cells
    )
    _, s_xy = umap_coordinates(
        s_cc, plot_rows, args.seed, args.max_umap_cells
    )
    plot_obs = target_obs.iloc[plot_rows]
    plot_response = heterogeneity[plot_rows]
    plot_cluster = cluster_all[plot_rows]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for row, xy, prefix in ((0, z_xy, "Background z-CC"), (1, s_xy, "Salient s-CC")):
        scatter_panel(axes[row, 0], xy, plot_obs["cell_type"], f"{prefix} — cell type")
        scatter_panel(axes[row, 1], xy, plot_obs["subregion"], f"{prefix} — brain subregion")
        scatter_panel(axes[row, 2], xy, plot_response, f"{prefix} — response heterogeneity", True)
        scatter_panel(axes[row, 3], xy, plot_obs["paired_batch"], f"{prefix} — paired slide")
    fig.suptitle("OSKM condition cells only (symmetric three-hop CellCharter)")
    fig.tight_layout()
    fig.savefig(figures / "oskm_condition_only_zcc_scc_umap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(
        data=target_obs[target_obs["s_cluster"] >= 0], x="s_cluster",
        y="response_heterogeneity", ax=axes[0], color="#6ab7a8", fliersize=0,
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Youngward response heterogeneity by s-CC cluster")
    ordered_types = [
        celltype for celltype in (
            "NSC", "Neuroblast", "Ependymal", "Endothelial", "Pericyte",
            "VSMC", "VLMC", "Microglia", "Macrophage", "Astrocyte", "OPC",
            "Oligodendrocyte", "Neuron-Excitatory", "Neuron-Inhibitory",
            "Neuron-MSN",
        ) if celltype in set(target_obs["cell_type"].astype(str))
    ]
    view = target_obs[target_obs["cell_type"].isin(ordered_types)].copy()
    view["cell_type"] = pd.Categorical(
        view["cell_type"].astype(str), categories=ordered_types, ordered=True
    )
    sns.boxplot(
        data=view, y="cell_type", x="local_rejuvenation_score", ax=axes[1],
        color="#8da0cb", fliersize=0, order=ordered_types,
    )
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("OSKM youngward score by cell lineage")
    fig.tight_layout()
    fig.savefig(figures / "oskm_salient_response_summary.png", dpi=220)
    plt.close(fig)

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
