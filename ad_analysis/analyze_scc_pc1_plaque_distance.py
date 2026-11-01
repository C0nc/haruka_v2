#!/usr/bin/env python
"""Relate latest z-CC/s-CC PC1 to plaque distances within samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_embedding(path: Path, distance: pd.DataFrame, condition_ids: set[str]):
    names = np.load(path / "obs_names.npy", allow_pickle=True).astype(str)
    keep = np.fromiter((name in condition_ids for name in names), dtype=bool)
    names = names[keep]
    order = pd.Index(distance["cell_id"].astype(str)).get_indexer(names)
    if (order < 0).any():
        raise ValueError(f"Distance table does not align with {path}")
    frame = distance.iloc[order].reset_index(drop=True)
    output = {}
    for label, filename in {"zCC": "spatial.npy", "sCC": "salient_spatial.npy"}.items():
        rep = np.load(path / filename).astype(np.float32)[keep]
        scaled = StandardScaler().fit_transform(rep)
        pca = PCA(n_components=1, random_state=0).fit(scaled)
        output[label] = (pca.transform(scaled)[:, 0], float(pca.explained_variance_ratio_[0]))
    return frame, output


def within_group_rank_correlation(x, y, group):
    rank_x = np.full(len(x), np.nan)
    rank_y = np.full(len(y), np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    for value in np.unique(group[valid]):
        rows = np.flatnonzero(valid & (group == value))
        if len(rows) < 3:
            continue
        # Percentile ranks put groups with different ranges on the same scale.
        rank_x[rows] = stats.rankdata(x[rows]) / len(rows)
        rank_y[rows] = stats.rankdata(y[rows]) / len(rows)
    keep = np.isfinite(rank_x) & np.isfinite(rank_y)
    result = stats.pearsonr(rank_x[keep], rank_y[keep])
    return float(result.statistic), float(result.pvalue), int(keep.sum())


def evaluate(dataset, frame, embeddings, group_key, endpoints):
    groups = frame[group_key].astype(str).to_numpy()
    summary = []
    by_group = []
    for representation, (pc1, variance) in embeddings.items():
        for endpoint, column in endpoints.items():
            distance = frame[column].to_numpy(float)
            valid = np.isfinite(pc1) & np.isfinite(distance)
            pooled = stats.spearmanr(pc1[valid], distance[valid])
            within_rho, within_p, n = within_group_rank_correlation(pc1, distance, groups)
            group_rhos = []
            for group in np.unique(groups[valid]):
                rows = valid & (groups == group)
                result = stats.spearmanr(pc1[rows], distance[rows])
                rho = float(result.statistic)
                group_rhos.append(rho)
                by_group.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "endpoint": endpoint,
                        "group": group,
                        "n": int(rows.sum()),
                        "spearman_rho": rho,
                        "pvalue": float(result.pvalue),
                    }
                )
            signs = np.sign(group_rhos) == np.sign(within_rho)
            summary.append(
                {
                    "dataset": dataset,
                    "representation": representation,
                    "endpoint": endpoint,
                    "n": n,
                    "n_groups": len(group_rhos),
                    "pc1_explained_variance": variance,
                    "pooled_spearman_rho": float(pooled.statistic),
                    "pooled_pvalue": float(pooled.pvalue),
                    "within_group_spearman_rho": within_rho,
                    "within_group_pvalue": within_p,
                    "median_group_rho": float(np.median(group_rhos)),
                    "min_group_rho": float(np.min(group_rhos)),
                    "max_group_rho": float(np.max(group_rhos)),
                    "groups_same_sign": int(np.sum(signs)),
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(by_group)


def binned_curves(ax, pc1, distance, group, title, rho):
    valid = np.isfinite(pc1) & np.isfinite(distance)
    curves = []
    for value in np.unique(group[valid]):
        rows = np.flatnonzero(valid & (group == value))
        if len(rows) < 20:
            continue
        order = np.argsort(distance[rows])
        bins = np.array_split(rows[order], 20)
        center = np.array([np.median(distance[idx]) for idx in bins])
        rank_center = np.arange(len(bins)) / (len(bins) - 1)
        normalized_pc1 = (pc1[rows] - np.mean(pc1[rows])) / max(np.std(pc1[rows]), 1e-8)
        lookup = pd.Series(normalized_pc1, index=rows)
        response = np.array([lookup.loc[idx].mean() for idx in bins])
        ax.plot(rank_center, response, linewidth=1, alpha=0.35, label=value)
        curves.append(response)
    if curves:
        mean_curve = np.mean(curves, axis=0)
        ax.plot(np.linspace(0, 1, len(mean_curve)), mean_curve, color="black", linewidth=2.5, label="mean")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_title(f"{title}\nwithin-sample Spearman ρ={rho:.3f}", fontsize=9)
    ax.set_xlabel("within-sample distance percentile (near → far)")
    ax.set_ylabel("within-sample standardized s-CC PC1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starmap-emb", required=True, type=Path)
    parser.add_argument("--codex-emb", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    starmap_distance = pd.read_csv(args.out / "starmap_plaque_distance_cells.csv.gz")
    codex_distance = pd.read_csv(args.out / "codex_plaque_distance_cells.csv.gz")
    starmap, starmap_rep = load_embedding(
        args.starmap_emb, starmap_distance, set(starmap_distance["cell_id"].astype(str))
    )
    codex, codex_rep = load_embedding(
        args.codex_emb, codex_distance, set(codex_distance["cell_id"].astype(str))
    )
    starmap_endpoints = {
        "Aβ-associated-cell distance": "distance_to_abeta_associated_cell_spatial_units",
        "p-tau-associated-cell distance": "distance_to_ptau_associated_cell_spatial_units",
        "either-plaque distance": "distance_to_either_associated_cell_spatial_units",
    }
    codex_endpoints = {
        "dense-Aβ centroid distance": "distance_to_dense_abeta_centroid_um",
        "diffuse-Aβ centroid distance": "distance_to_diffuse_abeta_centroid_um",
        "native any-plaque distance": "native_distance_to_any_plaque_um",
    }
    starmap_summary, starmap_groups = evaluate(
        "STARmap", starmap, starmap_rep, "slice_id", starmap_endpoints
    )
    codex_summary, codex_groups = evaluate(
        "CODEX", codex, codex_rep, "donor_id", codex_endpoints
    )
    summary = pd.concat([starmap_summary, codex_summary], ignore_index=True)
    groups = pd.concat([starmap_groups, codex_groups], ignore_index=True)
    summary.to_csv(args.out / "cc_pc1_plaque_distance_association.csv", index=False)
    groups.to_csv(args.out / "cc_pc1_plaque_distance_by_sample.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, (dataset, frame, rep, group_key, endpoints) in enumerate(
        [
            ("STARmap", starmap, starmap_rep, "slice_id", starmap_endpoints),
            ("CODEX", codex, codex_rep, "donor_id", codex_endpoints),
        ]
    ):
        pc1 = rep["sCC"][0]
        group = frame[group_key].astype(str).to_numpy()
        for col, (endpoint, column) in enumerate(endpoints.items()):
            rho = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["representation"].eq("sCC")
                & summary["endpoint"].eq(endpoint),
                "within_group_spearman_rho",
            ].iloc[0]
            binned_curves(
                axes[row, col], pc1, frame[column].to_numpy(float), group,
                f"{dataset}: {endpoint}", rho,
            )
    fig.suptitle(
        "s-CC PC1 versus plaque distance (condition cells only)\n"
        "Each colored line is one slice/donor; black is their mean; PC1 sign is arbitrary",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out / "scc_pc1_vs_plaque_distance.png", dpi=240)
    plt.close(fig)

    # Compact z-CC/s-CC comparison using the confounder-resistant statistic.
    fig, ax = plt.subplots(figsize=(13, 5))
    plot = summary.copy()
    plot["label"] = plot["dataset"] + ": " + plot["endpoint"]
    sns.barplot(
        data=plot, x="label", y="within_group_spearman_rho",
        hue="representation", ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("within-slice/donor Spearman ρ")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Plaque-distance association: z-CC PC1 versus s-CC PC1")
    fig.tight_layout()
    fig.savefig(args.out / "zcc_scc_pc1_plaque_distance_comparison.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
