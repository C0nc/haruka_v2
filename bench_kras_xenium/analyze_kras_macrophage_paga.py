#!/usr/bin/env python3
"""Macrophage-focused Haruka/PAGA analysis for the KRAS Xenium experiment.

This is the downstream analysis corresponding to the KRAS application rather
than the pulmonary-fibrosis/SLAT application.  It uses Haruka's three-hop
CellCharter aggregates (z-CC and s-CC), restricts the graph to macrophage and
monocyte states, and constructs both annotation-level and unsupervised PAGA
graphs.  PAGA is descriptive here: control and treated sections are unpaired,
so the graph is not interpreted as a measured longitudinal lineage.
"""

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


MACROPHAGE_STATES = [
    "Monocyte",
    "moMacrophage",
    "Macrophage_interstitial",
    "Macrophage_alveolar",
    "Macrophage_alveolar_prolif",
]


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=object)


def read_categorical(group: h5py.Group) -> np.ndarray:
    categories = _decode(group["categories"][:])
    codes = group["codes"][:]
    values = np.full(len(codes), "NA", dtype=object)
    valid = codes >= 0
    values[valid] = categories[codes[valid]]
    return values


def read_obs_and_spatial(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        columns = ["celltype", "condition_label", "slice_id", "subcluster", "leiden_final"]
        obs = pd.DataFrame({name: read_categorical(handle["obs"][name]) for name in columns})
        names = _decode(handle["obs"]["_index"][:])
        spatial = np.asarray(handle["obsm"]["spatial"][:], dtype=np.float32)
    obs.index = names
    return obs, spatial, names


def prepare_representation(rep: np.ndarray, rows: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(rep[rows], dtype=np.float32)
    values = StandardScaler().fit_transform(values).astype(np.float32)
    n_components = min(30, values.shape[1] - 1, values.shape[0] - 1)
    return PCA(n_components=n_components, random_state=seed).fit_transform(values).astype(np.float32)


def run_unsupervised_paga(
    matrix: np.ndarray,
    obs: pd.DataFrame,
    resolution: float,
    neighbors: int,
    seed: int,
) -> ad.AnnData:
    data = ad.AnnData(matrix)
    data.obs = obs.reset_index(drop=True).copy()
    sc.pp.neighbors(data, n_neighbors=neighbors, use_rep="X", random_state=seed)
    sc.tl.leiden(
        data,
        resolution=resolution,
        key_added="paga_cluster",
        random_state=seed,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    sc.tl.paga(data, groups="paga_cluster")
    # Keep manifold initialization independent of a preceding plotting call.
    sc.tl.umap(data, init_pos="spectral", min_dist=0.30, random_state=seed)

    cluster = data.obs["paga_cluster"].astype(str).to_numpy()
    condition = data.obs["condition_label"].astype(str).to_numpy()
    tab = pd.crosstab(cluster, condition).reindex(data.obs["paga_cluster"].cat.categories, fill_value=0)
    treated_fraction = tab.get("KRASi_treated", pd.Series(0, index=tab.index)) / tab.sum(1).clip(lower=1)
    root_cluster = treated_fraction.idxmin()
    root_candidates = np.flatnonzero(cluster == str(root_cluster))
    data.uns["iroot"] = int(root_candidates[0])
    sc.tl.diffmap(data)
    sc.tl.dpt(data)
    return data


def annotation_connectivity(matrix: np.ndarray, states: np.ndarray, neighbors: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cats = [x for x in MACROPHAGE_STATES if x in set(states)]
    model = NearestNeighbors(n_neighbors=min(neighbors + 1, len(matrix)), metric="euclidean", n_jobs=-1)
    indices = model.fit(matrix).kneighbors(return_distance=False)[:, 1:]
    state_to_int = {state: i for i, state in enumerate(cats)}
    codes = np.asarray([state_to_int[x] for x in states], dtype=np.int16)
    counts = np.zeros((len(cats), len(cats)), dtype=np.float64)
    for i in range(len(matrix)):
        np.add.at(counts[codes[i]], codes[indices[i]], 1.0)
    counts = counts + counts.T
    size = np.bincount(codes, minlength=len(cats)).astype(float)
    expected = np.outer(size, size)
    score = counts / np.maximum(expected, 1.0)
    np.fill_diagonal(score, 0.0)
    if score.max() > 0:
        score /= score.max()
    return pd.DataFrame(score, index=cats, columns=cats), pd.DataFrame(counts, index=cats, columns=cats)


def cluster_summary(data: ad.AnnData, representation: str) -> pd.DataFrame:
    rows = []
    for cluster in data.obs["paga_cluster"].cat.categories:
        take = data.obs["paga_cluster"].astype(str).to_numpy() == str(cluster)
        state_counts = data.obs.loc[take, "celltype"].value_counts()
        section_counts = data.obs.loc[take, "slice_id"].value_counts()
        treated = (data.obs.loc[take, "condition_label"].astype(str) == "KRASi_treated").mean()
        rows.append(
            {
                "representation": representation,
                "cluster": str(cluster),
                "n_cells": int(take.sum()),
                "treated_fraction": float(treated),
                "dominant_cell_state": str(state_counts.index[0]),
                "dominant_cell_state_fraction": float(state_counts.iloc[0] / take.sum()),
                "dominant_section": str(section_counts.index[0]),
                "dominant_section_fraction": float(section_counts.iloc[0] / take.sum()),
                "median_dpt": float(np.median(data.obs.loc[take, "dpt_pseudotime"])),
            }
        )
    return pd.DataFrame(rows)


def paga_edges(data: ad.AnnData, representation: str) -> pd.DataFrame:
    conn = data.uns["paga"]["connectivities"]
    if sparse.issparse(conn):
        conn = conn.toarray()
    cats = list(data.obs["paga_cluster"].cat.categories.astype(str))
    rows = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            rows.append({"representation": representation, "source": cats[i], "target": cats[j], "connectivity": float(conn[i, j])})
    return pd.DataFrame(rows).sort_values("connectivity", ascending=False)


def draw_paga(ax: plt.Axes, data: ad.AnnData, summary: pd.DataFrame, title: str) -> None:
    conn = data.uns["paga"]["connectivities"]
    if sparse.issparse(conn):
        conn = conn.toarray()
    labels = list(data.obs["paga_cluster"].cat.categories.astype(str))
    graph = nx.Graph()
    for label in labels:
        graph.add_node(label)
    nonzero = conn[np.triu_indices_from(conn, 1)]
    threshold = max(0.03, float(np.quantile(nonzero[nonzero > 0], 0.35))) if np.any(nonzero > 0) else 1.0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if conn[i, j] >= threshold:
                graph.add_edge(labels[i], labels[j], weight=float(conn[i, j]))
    pos = nx.spring_layout(graph, seed=4, weight="weight", k=1.25)
    lookup = summary.set_index("cluster").loc[labels]
    node_size = 350 + 2600 * lookup["n_cells"].to_numpy() / lookup["n_cells"].max()
    node_color = lookup["treated_fraction"].to_numpy()
    widths = [0.5 + 5.0 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, alpha=0.45, edge_color="#64748b")
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        edgecolors="white",
        linewidths=1.2,
    )
    labels_text = {c: f"{c}\n{lookup.loc[c, 'dominant_cell_state'].replace('Macrophage_', 'Mac_')}" for c in labels}
    nx.draw_networkx_labels(graph, pos, labels=labels_text, ax=ax, font_size=7)
    plt.colorbar(nodes, ax=ax, fraction=0.045, pad=0.02, label="KRASi-treated fraction")
    ax.set_title(title)
    ax.axis("off")


def plot_umaps(z_data: ad.AnnData, s_data: ad.AnnData, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, (name, data) in enumerate([("Background z-CC", z_data), ("Salient s-CC", s_data)]):
        xy = data.obsm["X_umap"]
        for col, key in enumerate(["celltype", "condition_label", "paga_cluster"]):
            values = data.obs[key].astype(str).to_numpy()
            cats = sorted(np.unique(values))
            palette = sns.color_palette("husl", len(cats))
            for cat, color in zip(cats, palette):
                take = values == cat
                axes[row, col].scatter(xy[take, 0], xy[take, 1], s=1.0, color=color, linewidths=0, rasterized=True, label=cat)
            axes[row, col].set_title(f"{name} — {key}")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            axes[row, col].legend(markerscale=5, fontsize=6, frameon=False, ncol=2, loc="best")
    fig.suptitle("KRAS macrophage-state manifold: background versus treatment-salient spatial embedding", fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_spatial(obs: pd.DataFrame, spatial: np.ndarray, s_data: ad.AnnData, rows: np.ndarray, out: Path) -> None:
    treated = obs.iloc[rows]["condition_label"].astype(str).to_numpy() == "KRASi_treated"
    slices = sorted(obs.iloc[rows].loc[treated, "slice_id"].astype(str).unique())
    clusters = s_data.obs["paga_cluster"].astype(str).to_numpy()
    cluster_categories = list(s_data.obs["paga_cluster"].cat.categories.astype(str))
    cluster_to_code = {cluster: code for code, cluster in enumerate(cluster_categories)}
    cluster_codes = np.asarray([cluster_to_code[x] for x in clusters], dtype=np.int16)
    dpt = s_data.obs["dpt_pseudotime"].to_numpy()
    fig, axes = plt.subplots(len(slices), 2, figsize=(12, 4.4 * len(slices)), squeeze=False)
    row_slice = obs.iloc[rows]["slice_id"].astype(str).to_numpy()
    for i, sid in enumerate(slices):
        take = treated & (row_slice == sid)
        xy = spatial[rows[take]]
        axes[i, 0].scatter(
            xy[:, 0], xy[:, 1], c=cluster_codes[take], cmap="tab20",
            vmin=-0.5, vmax=max(0.5, len(cluster_categories) - 0.5),
            s=3, linewidths=0, rasterized=True,
        )
        im = axes[i, 1].scatter(xy[:, 0], xy[:, 1], c=dpt[take], cmap="viridis", vmin=0, vmax=1, s=3, linewidths=0, rasterized=True)
        axes[i, 0].set_title(f"{sid}: salient macrophage PAGA clusters")
        axes[i, 1].set_title(f"{sid}: salient macrophage DPT ordering")
        for ax in axes[i]:
            ax.set_aspect("equal"); ax.axis("off")
    fig.colorbar(im, ax=axes[:, 1].tolist(), fraction=0.015, pad=0.01, label="DPT (descriptive, not longitudinal)")
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad"))
    parser.add_argument("--cc-dir", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/results_cellcharter/haruka_z_s_cc3"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/yancui/haruka_v2/bench_kras_xenium/analysis/fig6_macrophage_paga"))
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    figure_dir = args.out_dir / "figures"
    table_dir = args.out_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    obs, spatial, names = read_obs_and_spatial(args.data)
    cc_names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(names.astype(str), cc_names):
        raise ValueError("CellCharter arrays do not align with the source AnnData rows")
    z_cc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    s_cc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    rows = np.flatnonzero(obs["celltype"].isin(MACROPHAGE_STATES).to_numpy())
    subset = obs.iloc[rows].copy()

    z_matrix = prepare_representation(z_cc, rows, args.seed)
    s_matrix = prepare_representation(s_cc, rows, args.seed)
    z_data = run_unsupervised_paga(z_matrix, subset, args.resolution, args.neighbors, args.seed)
    s_data = run_unsupervised_paga(s_matrix, subset, args.resolution, args.neighbors, args.seed)

    summaries = []
    edges = []
    for name, data in [("z_cc", z_data), ("s_cc", s_data)]:
        summary = cluster_summary(data, name)
        summaries.append(summary)
        edges.append(paga_edges(data, name))
    summary_table = pd.concat(summaries, ignore_index=True)
    edge_table = pd.concat(edges, ignore_index=True)
    summary_table.to_csv(table_dir / "unsupervised_paga_node_summary.csv", index=False)
    edge_table.to_csv(table_dir / "unsupervised_paga_edges.csv", index=False)

    association_records = []
    for name, data in [("z_cc", z_data), ("s_cc", s_data)]:
        clusters = data.obs["paga_cluster"].astype(str).to_numpy()
        for label in ["condition_label", "celltype", "slice_id"]:
            truth = data.obs[label].astype(str).to_numpy()
            association_records.append({
                "representation": name, "label": label,
                "ari": adjusted_rand_score(truth, clusters),
                "nmi": normalized_mutual_info_score(truth, clusters),
            })
    pd.DataFrame(association_records).to_csv(
        table_dir / "paga_cluster_association_metrics.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw_paga(axes[0], z_data, summaries[0], "Background z-CC macrophage PAGA")
    draw_paga(axes[1], s_data, summaries[1], "Salient s-CC macrophage PAGA")
    fig.suptitle("KRAS inhibitor response: macrophage-state connectivity", fontsize=16)
    fig.tight_layout()
    fig.savefig(figure_dir / "macrophage_zcc_scc_paga.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    state_tables = []
    for rep_name, matrix in [("z_cc", z_matrix), ("s_cc", s_matrix)]:
        for condition in ["Control", "KRASi_treated"]:
            take = subset["condition_label"].astype(str).to_numpy() == condition
            score, raw = annotation_connectivity(matrix[take], subset.loc[take, "celltype"].astype(str).to_numpy(), args.neighbors)
            score.to_csv(table_dir / f"{rep_name}_{condition}_state_connectivity.csv")
            long = score.rename_axis("source").reset_index().melt("source", var_name="target", value_name="normalized_connectivity")
            long["representation"] = rep_name; long["condition"] = condition
            state_tables.append(long)
    pd.concat(state_tables, ignore_index=True).to_csv(table_dir / "macrophage_state_connectivity_long.csv", index=False)

    plot_umaps(z_data, s_data, figure_dir / "macrophage_zcc_scc_umap.png")
    plot_spatial(obs, spatial, s_data, rows, figure_dir / "treated_salient_macrophage_paga_spatial.png")

    rng = np.random.default_rng(args.seed)
    metric_rows = rng.choice(len(rows), min(10_000, len(rows)), replace=False)
    metric_records = []
    for name, matrix in [("z_cc", z_matrix), ("s_cc", s_matrix)]:
        for label in ["celltype", "condition_label", "slice_id"]:
            values = subset.iloc[metric_rows][label].astype(str).to_numpy()
            metric_records.append(
                {
                    "representation": name,
                    "label": label,
                    "silhouette": float(silhouette_score(matrix[metric_rows], values, sample_size=min(5000, len(metric_rows)), random_state=args.seed)),
                }
            )
    pd.DataFrame(metric_records).to_csv(table_dir / "embedding_silhouette_metrics.csv", index=False)

    assignments = subset.copy()
    assignments["z_cc_paga_cluster"] = z_data.obs["paga_cluster"].astype(str).to_numpy()
    assignments["s_cc_paga_cluster"] = s_data.obs["paga_cluster"].astype(str).to_numpy()
    assignments["z_cc_dpt"] = z_data.obs["dpt_pseudotime"].to_numpy()
    assignments["s_cc_dpt"] = s_data.obs["dpt_pseudotime"].to_numpy()
    assignments.to_csv(table_dir / "macrophage_paga_assignments.csv.gz", compression="gzip")

    metadata = {
        "analysis": "KRAS Figure-6-style macrophage PAGA on Haruka z-CC and s-CC",
        "n_cells": int(len(rows)),
        "cell_states": MACROPHAGE_STATES,
        "control_sections": sorted(subset.loc[subset.condition_label == "Control", "slice_id"].unique().tolist()),
        "treated_sections": sorted(subset.loc[subset.condition_label == "KRASi_treated", "slice_id"].unique().tolist()),
        "representation": {"z_cc": "spatial.npy", "s_cc": "salient_spatial.npy"},
        "neighbors": args.neighbors,
        "leiden_resolution": args.resolution,
        "interpretation_caveat": "PAGA/DPT describe connectivity and ordering in unpaired endpoint sections; they do not prove longitudinal lineage transitions.",
        "correction": "The fibrosis/SLAT workflow belongs to the pulmonary-fibrosis application and is not used as the KRAS headline analysis.",
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
