#!/usr/bin/env python3
"""Condition-only macrophage analysis for the corrected KRAS salient graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

MACROPHAGE_STATES = [
    "Monocyte", "moMacrophage", "Macrophage_interstitial",
    "Macrophage_alveolar", "Macrophage_alveolar_prolif",
]


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=object)


def categorical(handle, key):
    group = handle["obs"][key]
    categories = decode(group["categories"][:])
    codes = group["codes"][:]
    result = np.full(len(codes), "NA", dtype=object)
    valid = codes >= 0
    result[valid] = categories[codes[valid]]
    return result


def prepare(values, seed):
    values = StandardScaler().fit_transform(np.asarray(values, dtype=np.float32))
    return PCA(n_components=min(30, values.shape[1]), random_state=seed).fit_transform(values).astype(np.float32)


def manifold(values, obs, seed):
    data = ad.AnnData(values, obs=obs.reset_index(drop=True).copy())
    sc.pp.neighbors(data, n_neighbors=30, use_rep="X", random_state=seed)
    sc.tl.leiden(data, resolution=0.45, key_added="cluster", random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)
    sc.tl.paga(data, groups="cluster")
    sc.pl.paga(data, show=False)
    plt.close()
    sc.tl.umap(data, init_pos="paga", min_dist=0.30, random_state=seed)
    return data


def enrichment(labels, annotation):
    table = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(annotation, name="celltype"))
    observed = table.div(table.sum(1), axis=0)
    expected = table.sum(0) / table.to_numpy().sum()
    return np.log2((observed + 1e-4).div(expected + 1e-4, axis=1))


def scatter(ax, xy, values, title):
    values = np.asarray(values).astype(str)
    categories = sorted(np.unique(values))
    for category, color in zip(categories, sns.color_palette("husl", len(categories))):
        take = values == category
        ax.scatter(xy[take, 0], xy[take, 1], s=1.2, color=color, linewidths=0,
                   rasterized=True, label=category)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6, markerscale=5, ncol=2, loc="best")


def paga_plot(ax, data, title):
    conn = data.uns["paga"]["connectivities"]
    if sparse.issparse(conn):
        conn = conn.toarray()
    categories = list(data.obs["cluster"].cat.categories.astype(str))
    graph = nx.Graph()
    graph.add_nodes_from(categories)
    positive = conn[np.triu_indices_from(conn, 1)]
    threshold = max(0.03, float(np.quantile(positive[positive > 0], 0.35))) if np.any(positive > 0) else 1
    for i, source in enumerate(categories):
        for j in range(i + 1, len(categories)):
            if conn[i, j] >= threshold:
                graph.add_edge(source, categories[j], weight=float(conn[i, j]))
    position = nx.spring_layout(graph, seed=4, weight="weight", k=1.2)
    counts = data.obs["cluster"].astype(str).value_counts().reindex(categories)
    sizes = 350 + 2600 * counts.to_numpy() / counts.max()
    colors = sns.color_palette("tab10", len(categories))
    widths = [0.5 + 5 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, position, ax=ax, width=widths, alpha=.45, edge_color="#64748b")
    nx.draw_networkx_nodes(graph, position, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors="white", linewidths=1.2)
    dominant = data.obs.groupby("cluster", observed=True)["celltype"].agg(lambda x: x.value_counts().index[0])
    labels = {cluster: f"{cluster}\n{dominant.loc[cluster].replace('Macrophage_', 'Mac_')}" for cluster in categories}
    nx.draw_networkx_labels(graph, position, labels=labels, ax=ax, font_size=7)
    ax.set_title(title); ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad"))
    parser.add_argument("--cc-dir", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/results_cellcharter/haruka_z_s_cc3_condition_only"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/analysis/condition_only_salient/macrophage"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    figures = args.out_dir / "figures"; tables = args.out_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    source_rows = np.load(args.cc_dir / "source_rows.npy")
    with h5py.File(args.data, "r") as handle:
        obs = pd.DataFrame({key: categorical(handle, key)[source_rows]
                            for key in ["celltype", "slice_id", "condition_label"]})
        names = decode(handle["obs"]["_index"][:])[source_rows]
        spatial = np.asarray(handle["obsm"]["spatial"][:], dtype=np.float32)[source_rows]
    if not np.all(obs["condition_label"].to_numpy() == "KRASi_treated"):
        raise ValueError("Condition-only representation contains a control cell")
    local = np.flatnonzero(obs["celltype"].isin(MACROPHAGE_STATES).to_numpy())
    subset = obs.iloc[local].reset_index(drop=True)
    z = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    s = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    z_data = manifold(prepare(z[local], args.seed), subset, args.seed)
    s_data = manifold(prepare(s[local], args.seed), subset, args.seed)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, (label, data) in enumerate([("Background z-CC", z_data), ("Salient s-CC", s_data)]):
        xy = data.obsm["X_umap"]
        scatter(axes[row, 0], xy, data.obs["celltype"], f"{label} — cell type")
        scatter(axes[row, 1], xy, data.obs["slice_id"], f"{label} — treated section")
        scatter(axes[row, 2], xy, data.obs["cluster"], f"{label} — cluster")
    fig.suptitle("KRASi-treated macrophages only: background and condition-only salient niches", fontsize=16)
    fig.tight_layout(); fig.savefig(figures / "treated_only_macrophage_zcc_scc_umap.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    paga_plot(axes[0], z_data, "Treated macrophage background z-CC PAGA")
    paga_plot(axes[1], s_data, "Treated macrophage salient s-CC PAGA")
    fig.tight_layout(); fig.savefig(figures / "treated_only_macrophage_zcc_scc_paga.png", dpi=250, bbox_inches="tight"); plt.close(fig)

    metrics = []
    assignments = subset.copy()
    assignments.insert(0, "cell_id", names[source_rows * 0 + source_rows][0:0] if False else names[local])
    assignments.insert(1, "source_row", source_rows[local])
    for rep, data in [("z_cc", z_data), ("s_cc", s_data)]:
        labels = data.obs["cluster"].astype(str).to_numpy()
        assignments[f"{rep}_cluster"] = labels
        for key in ["celltype", "slice_id"]:
            truth = subset[key].astype(str).to_numpy()
            metrics.append({"representation": rep, "label": key,
                            "ari": adjusted_rand_score(truth, labels),
                            "nmi": normalized_mutual_info_score(truth, labels)})
        matrix = enrichment(labels, subset["celltype"].astype(str).to_numpy())
        matrix.to_csv(tables / f"{rep}_macrophage_celltype_log2_enrichment.csv")
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.heatmap(matrix, cmap="vlag", center=0, ax=ax,
                    cbar_kws={"label": "log2 observed / expected"})
        ax.set_title(f"{rep} macrophage cluster cell-type enrichment — KRASi treated only")
        fig.tight_layout(); fig.savefig(figures / f"{rep}_macrophage_celltype_enrichment.png", dpi=230, bbox_inches="tight"); plt.close(fig)
    assignments.to_csv(tables / "treated_macrophage_cluster_assignments.csv.gz", index=False, compression="gzip")
    pd.DataFrame(metrics).to_csv(tables / "treated_macrophage_cluster_association.csv", index=False)

    sections = sorted(subset["slice_id"].unique())
    fig, axes = plt.subplots(len(sections), 1, figsize=(7, 4.2 * len(sections)), squeeze=False)
    s_labels = s_data.obs["cluster"].cat.codes.to_numpy()
    for i, section in enumerate(sections):
        take = subset["slice_id"].to_numpy() == section
        xy = spatial[local[take]]
        axes[i, 0].scatter(xy[:, 0], xy[:, 1], c=s_labels[take], cmap="tab20", s=3,
                           linewidths=0, rasterized=True)
        axes[i, 0].set_title(f"{section}: condition-only salient macrophage clusters")
        axes[i, 0].set_aspect("equal"); axes[i, 0].axis("off")
    fig.tight_layout(); fig.savefig(figures / "treated_only_salient_macrophage_clusters_spatial.png", dpi=250, bbox_inches="tight"); plt.close(fig)

    metadata = {"subset": "KRASi_treated macrophage/monocyte cells only",
                "n_cells": int(len(local)), "control_cells_in_salient": 0,
                "representation": str(args.cc_dir), "neighbors": 30,
                "celltype_enrichment": "post-clustering log2 observed/expected; cell types also contributed to Haruka niche supervision",
                "slat": "not run"}
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
