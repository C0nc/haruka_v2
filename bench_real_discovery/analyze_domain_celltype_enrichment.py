#!/usr/bin/env python3
"""Annotate condition-only z-CC/s-CC domains and spatial cell-cell enrichments."""

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
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def balanced_rows(groups: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(groups) <= limit:
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    values = np.unique(groups)
    quota = max(1, limit // len(values))
    parts = []
    for value in values:
        rows = np.flatnonzero(groups == value)
        parts.append(rng.choice(rows, min(quota, len(rows)), replace=False))
    return np.sort(np.concatenate(parts))


def fit_domains(
    matrix: np.ndarray, groups: np.ndarray, n_clusters: int, max_fit: int, seed: int
) -> np.ndarray:
    rows = balanced_rows(groups, max_fit, seed)
    scaler = StandardScaler().fit(np.asarray(matrix[rows], dtype=np.float32))
    model = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, n_init=20, batch_size=8192
    ).fit(scaler.transform(np.asarray(matrix[rows], dtype=np.float32)))
    labels = np.empty(len(matrix), dtype=np.int16)
    for left in range(0, len(matrix), 100_000):
        right = min(left + 100_000, len(matrix))
        labels[left:right] = model.predict(
            scaler.transform(np.asarray(matrix[left:right], dtype=np.float32))
        )
    return labels


def type_enrichment(
    labels: np.ndarray, celltypes: np.ndarray, n_clusters: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        pd.Series(labels, name="domain"), pd.Series(celltypes, name="cell_type")
    ).reindex(range(n_clusters), fill_value=0)
    composition = counts.div(counts.sum(1).replace(0, np.nan), axis=0)
    baseline = counts.sum(0) / counts.to_numpy().sum()
    enrichment = np.log2((composition + 1e-4).div(baseline + 1e-4, axis=1))
    return composition, enrichment


def cellcell_enrichment(
    labels: np.ndarray,
    celltypes: np.ndarray,
    samples: np.ndarray,
    spatial: np.ndarray,
    n_clusters: int,
    k: int,
) -> pd.DataFrame:
    categories = np.unique(celltypes)
    lookup = {value: i for i, value in enumerate(categories)}
    codes = np.asarray([lookup[x] for x in celltypes], dtype=np.int16)
    observed = np.zeros((n_clusters, len(categories), len(categories)), dtype=np.int64)
    domain_type_counts = np.zeros((n_clusters, len(categories)), dtype=np.int64)
    for domain in range(n_clusters):
        np.add.at(domain_type_counts[domain], codes[labels == domain], 1)
    for sample in np.unique(samples):
        rows = np.flatnonzero(samples == sample)
        if len(rows) < 2:
            continue
        take_k = min(k + 1, len(rows))
        _, local_neighbors = cKDTree(spatial[rows]).query(
            spatial[rows], k=take_k, workers=1
        )
        if local_neighbors.ndim == 1:
            local_neighbors = local_neighbors[:, None]
        neighbor_rows = rows[local_neighbors[:, 1:].reshape(-1)]
        source_rows = np.repeat(rows, local_neighbors.shape[1] - 1)
        same_domain = labels[source_rows] == labels[neighbor_rows]
        source_rows = source_rows[same_domain]
        neighbor_rows = neighbor_rows[same_domain]
        flat = (
            labels[source_rows].astype(np.int64) * len(categories) * len(categories)
            + codes[source_rows].astype(np.int64) * len(categories)
            + codes[neighbor_rows].astype(np.int64)
        )
        observed += np.bincount(
            flat, minlength=n_clusters * len(categories) * len(categories)
        ).reshape(observed.shape)

    records = []
    for domain in range(n_clusters):
        total_edges = observed[domain].sum()
        abundance = domain_type_counts[domain].astype(float)
        probability = abundance / max(abundance.sum(), 1.0)
        expected = total_edges * probability[:, None] * probability[None, :]
        score = np.log2((observed[domain] + 0.5) / (expected + 0.5))
        for i, source in enumerate(categories):
            for j, neighbor in enumerate(categories):
                records.append(
                    {
                        "domain": domain,
                        "source_cell_type": source,
                        "neighbor_cell_type": neighbor,
                        "observed_edges": int(observed[domain, i, j]),
                        "expected_edges": float(expected[i, j]),
                        "log2_enrichment": float(score[i, j]),
                    }
                )
    return pd.DataFrame(records)


def plot_heatmaps(
    composition: pd.DataFrame,
    enrichment: pd.DataFrame,
    interactions: pd.DataFrame,
    prefix: str,
    out: Path,
) -> None:
    abundant = composition.sum(0).nlargest(min(20, composition.shape[1])).index
    eligible = interactions[interactions["observed_edges"] >= 25].copy()
    eligible["pair"] = (
        eligible["source_cell_type"].astype(str)
        + " → "
        + eligible["neighbor_cell_type"].astype(str)
    )
    top_pairs = (
        eligible.groupby("pair", observed=True)["log2_enrichment"]
        .apply(lambda x: float(np.nanmax(np.abs(x))))
        .nlargest(30).index
    )
    pair_matrix = (
        eligible[eligible["pair"].isin(top_pairs)]
        .pivot_table(index="domain", columns="pair", values="log2_enrichment")
        .reindex(range(len(composition)))
    )
    fig, axes = plt.subplots(1, 3, figsize=(25, 7), constrained_layout=True)
    sns.heatmap(composition[abundant], cmap="mako", ax=axes[0], cbar_kws={"label": "fraction"})
    axes[0].set_title(f"{prefix}: domain cell-type composition")
    sns.heatmap(enrichment[abundant], cmap="vlag", center=0, ax=axes[1], cbar_kws={"label": "log2 observed/expected"})
    axes[1].set_title(f"{prefix}: cell-type enrichment")
    if pair_matrix.shape[1]:
        sns.heatmap(pair_matrix, cmap="vlag", center=0, ax=axes[2], cbar_kws={"label": "log2 observed/expected"})
    axes[2].set_title(f"{prefix}: within-domain spatial cell-cell enrichment")
    for ax in axes:
        ax.set_ylabel("domain")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cc-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--cell-type-key", required=True)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--max-fit", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(args.data, backed="r")
    source_rows = np.load(args.cc_dir / "source_rows.npy").astype(np.int64)
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    obs = data.obs.iloc[source_rows].copy()
    if not np.array_equal(names, obs.index.astype(str).to_numpy()):
        raise ValueError("CellCharter rows do not align to source AnnData")
    spatial = np.asarray(data.obsm["spatial"], dtype=np.float32)[source_rows]
    samples = obs[args.sample_key].astype(str).to_numpy()
    celltypes = obs[args.cell_type_key].astype(str).to_numpy()

    assignments = pd.DataFrame(
        {"cell_id": names, args.sample_key: samples, "cell_type": celltypes}
    )
    summaries = []
    for prefix, filename in (("z_cc", "spatial.npy"), ("s_cc", "salient_spatial.npy")):
        matrix = np.load(args.cc_dir / filename, mmap_mode="r")
        labels = fit_domains(matrix, samples, args.clusters, args.max_fit, args.seed)
        assignments[f"{prefix}_domain"] = labels
        composition, enrichment = type_enrichment(labels, celltypes, args.clusters)
        interactions = cellcell_enrichment(
            labels, celltypes, samples, spatial, args.clusters, args.neighbors
        )
        composition.to_csv(args.out_dir / f"{prefix}_domain_celltype_composition.csv")
        enrichment.to_csv(args.out_dir / f"{prefix}_domain_celltype_log2_enrichment.csv")
        interactions.to_csv(args.out_dir / f"{prefix}_domain_cellcell_log2_enrichment.csv", index=False)
        plot_heatmaps(
            composition, enrichment, interactions, prefix,
            args.out_dir / f"{prefix}_domain_celltype_and_cellcell_enrichment.png",
        )
        for domain in range(args.clusters):
            comp = composition.loc[domain].sort_values(ascending=False)
            pair = interactions[
                (interactions["domain"] == domain)
                & (interactions["observed_edges"] >= 25)
            ].sort_values("log2_enrichment", ascending=False)
            summaries.append(
                {
                    "representation": prefix,
                    "domain": domain,
                    "n_cells": int((labels == domain).sum()),
                    "dominant_cell_type": comp.index[0],
                    "dominant_cell_type_fraction": float(comp.iloc[0]),
                    "top_enriched_cell_type": enrichment.loc[domain].idxmax(),
                    "top_cellcell_pair": (
                        f"{pair.iloc[0].source_cell_type} → {pair.iloc[0].neighbor_cell_type}"
                        if len(pair) else "insufficient_edges"
                    ),
                    "top_cellcell_log2_enrichment": (
                        float(pair.iloc[0].log2_enrichment) if len(pair) else np.nan
                    ),
                }
            )
    assignments.to_csv(args.out_dir / "domain_assignments.csv.gz", index=False, compression="gzip")
    pd.DataFrame(summaries).to_csv(args.out_dir / "domain_annotations.csv", index=False)
    metadata = {
        "population": "condition cells encoded by condition-only CellCharter",
        "clustering": f"MiniBatchKMeans(k={args.clusters}), seed={args.seed}",
        "spatial_graph": f"{args.neighbors}-nearest neighbors, bounded by {args.sample_key}",
        "cellcell_statistic": "directed source-to-neighbor edge log2 observed/expected within each domain",
        "annotation_role": "post-fit interpretation; cell type also contributes to Haruka niche reconstruction",
        "n_cells": int(len(obs)),
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
