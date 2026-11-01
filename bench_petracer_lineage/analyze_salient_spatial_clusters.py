"""Biological interpretation of asymmetric PEtracer salient spatial clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, ttest_ind
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


BROAD_PROGRAMS = {
    "cytotoxic_T_NK": ["Gzma", "Gzmb", "Ifng", "Fasl", "Ncr1", "Klrk1", "Cd8a", "Cd8b1"],
    "T_cell_identity": ["Cd3g", "Cd247", "Cd4", "Cd8a", "Cd8b1", "Itk", "Themis", "Skap1"],
    "B_cell_identity": ["Cd79a", "Ms4a1", "Cd22", "Cr2", "Ebf1", "Pax5", "Fcmr"],
    "myeloid_phagocytic": ["C1qa", "C1qb", "C1qc", "Mrc1", "Ms4a7", "Dab2", "Mmp12", "Arg1", "Alox15"],
    "endothelial_vascular": ["Cldn5", "Clec14a", "Gpihbp1", "Ptprb", "Robo4", "Sox17"],
    "epithelial_malignant": ["Cldn4", "Fxyd3", "Tmprss11e", "Krt79", "Fermt1", "Azgp1"],
    "dendritic_APC": ["Batf3", "Flt3", "Fscn1", "Itgax", "Xcr1", "Fcer1a"],
    "ECM_fibroblast": ["Col5a2", "Col6a6", "Chst3", "Vcan", "Cxcl14"],
}


def dense_log_cpm(matrix) -> np.ndarray:
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    values = values.astype(np.float32)
    total = values.sum(1, keepdims=True)
    np.divide(values * 1e4, total, out=values, where=total > 0)
    np.log1p(values, out=values)
    return values


def zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(0, keepdims=True)
    std = values.std(0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-5)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if len(values) < 2 or np.var(values) <= 1e-12:
        return float("nan")
    grand = values.mean()
    between = sum(np.sum(groups == group) * (values[groups == group].mean() - grand) ** 2 for group in np.unique(groups))
    return float(between / np.sum((values - grand) ** 2))


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full_like(pvalues, np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if not len(finite):
        return qvalues
    order = finite[np.argsort(pvalues[finite])]
    ranked = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qvalues[order] = np.minimum(ranked, 1.0)
    return qvalues


def scatter_categories(axis, xy, labels, title, point_size=4.0):
    labels = np.asarray(labels).astype(str)
    for value in np.unique(labels):
        take = labels == value
        axis.scatter(xy[take, 0], xy[take, 1], s=point_size, alpha=0.7, label=value, rasterized=True)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    if len(np.unique(labels)) <= 20:
        axis.legend(fontsize=5.5, markerscale=2.5, frameon=False, ncol=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--emb-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--program-json", required=True, type=Path)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(args.data)
    names = np.load(args.emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Index(a.obs_names.astype(str)).get_indexer(names)
    if (order < 0).any():
        raise ValueError("Embedding observations do not align to AnnData")
    a = a[order].copy()
    obs = a.obs.copy()
    target = obs["condition"].astype(str).to_numpy() == "disease"
    eligible = target & obs["lineage_effect_eligible"].astype(bool).to_numpy()
    focal_rows = np.flatnonzero(eligible)
    target_rows = np.flatnonzero(target)
    focal_target_rows = pd.Index(target_rows).get_indexer(focal_rows)
    if (focal_target_rows < 0).any():
        raise RuntimeError("Eligible focal rows are not in the condition arm")

    s_cc = np.load(args.emb_dir / "salient_spatial.npy")
    focal_rep = s_cc[focal_rows]
    variable = np.var(focal_rep, axis=0) > 1e-10
    cluster = KMeans(n_clusters=args.clusters, n_init=20, random_state=args.seed).fit_predict(focal_rep[:, variable])
    section = str(obs["section_id"].iloc[0])

    pair = obs["pair_id"].astype(str).to_numpy()
    control_rows = np.flatnonzero(~target)
    control_by_pair = pd.Series(control_rows, index=pair[control_rows])
    matched_control = control_by_pair.loc[pair[focal_rows]].to_numpy()
    counts = a.layers["counts"]
    condition_log = dense_log_cpm(counts[focal_rows])
    control_log = dense_log_cpm(counts[matched_control])
    delta = condition_log - control_log
    genes = pd.Index(a.var_names.astype(str))
    control_z = zscore(control_log)
    delta_z = zscore(delta)

    with open(args.program_json) as handle:
        simulation = json.load(handle)
    effect_genes = simulation["effect_genes"]
    family_programs = {
        f"synthetic_family_{family}": effect_genes[2 * family : 2 * family + 2]
        for family in range(6)
    }
    state_genes = simulation["state_interaction_genes"]

    cell_scores = pd.DataFrame(index=obs.index[focal_rows])
    cell_scores["paired_change_l1"] = np.mean(np.abs(delta), axis=1)
    for name, members in BROAD_PROGRAMS.items():
        indices = genes.get_indexer([gene for gene in members if gene in genes])
        indices = indices[indices >= 0]
        if len(indices):
            cell_scores[f"native_{name}"] = control_z[:, indices].mean(1)
            cell_scores[f"response_{name}"] = delta_z[:, indices].mean(1)
    for name, members in family_programs.items():
        indices = genes.get_indexer([gene for gene in members if gene in genes])
        indices = indices[indices >= 0]
        if len(indices):
            cell_scores[f"response_{name}"] = delta_z[:, indices].mean(1)
    state_indices = genes.get_indexer([gene for gene in state_genes if gene in genes])
    state_indices = state_indices[state_indices >= 0]
    cell_scores["native_simulation_state"] = control_z[:, state_indices].mean(1)
    cell_scores["cluster"] = cluster
    cell_scores["section"] = section
    cell_scores["lineage_family"] = obs.iloc[focal_rows]["gt_lineage_family"].astype(str).to_numpy()
    cell_scores["lineage_clade"] = obs.iloc[focal_rows]["gt_lineage_clade"].astype(str).to_numpy()
    cell_scores.to_csv(args.out_dir / "cell_program_scores.csv")

    program_columns = [column for column in cell_scores if column.startswith(("native_", "response_"))] + ["paired_change_l1"]
    program_rows = []
    for group in range(args.clusters):
        take = cluster == group
        for column in program_columns:
            values = cell_scores[column].to_numpy(float)
            statistic, pvalue = mannwhitneyu(values[take], values[~take], alternative="two-sided")
            program_rows.append({
                "section": section, "cluster": group, "program": column,
                "mean": float(np.mean(values[take])), "rest_mean": float(np.mean(values[~take])),
                "difference": float(np.mean(values[take]) - np.mean(values[~take])),
                "pvalue": float(pvalue), "n_cells": int(take.sum()),
            })
    program_frame = pd.DataFrame(program_rows)
    program_frame["fdr"] = fdr_bh(program_frame["pvalue"].to_numpy())
    program_frame.to_csv(args.out_dir / "program_enrichment.csv", index=False)

    gene_rows = []
    for group in range(args.clusters):
        take = cluster == group
        statistic, pvalue = ttest_ind(delta[take], delta[~take], axis=0, equal_var=False, nan_policy="omit")
        difference = delta[take].mean(0) - delta[~take].mean(0)
        qvalue = fdr_bh(pvalue)
        order_gene = np.argsort(-np.abs(difference))[:30]
        for rank, gene_index in enumerate(order_gene, start=1):
            gene_rows.append({
                "section": section, "cluster": group, "rank": rank,
                "gene": genes[gene_index], "delta_difference": float(difference[gene_index]),
                "cluster_mean_delta": float(delta[take, gene_index].mean()),
                "rest_mean_delta": float(delta[~take, gene_index].mean()),
                "pvalue": float(pvalue[gene_index]), "fdr": float(qvalue[gene_index]),
                "is_simulated_effect_gene": bool(genes[gene_index] in effect_genes),
            })
    pd.DataFrame(gene_rows).to_csv(args.out_dir / "top_paired_delta_genes.csv", index=False)

    continuous = {
        "tumor_boundary_dist": pd.to_numeric(obs.iloc[focal_rows]["tumor_boundary_dist"], errors="coerce").to_numpy(),
        "lung_boundary_dist": pd.to_numeric(obs.iloc[focal_rows]["lung_boundary_dist"], errors="coerce").to_numpy(),
        "total_density": pd.to_numeric(obs.iloc[focal_rows]["total_density"], errors="coerce").to_numpy(),
        "fitness": pd.to_numeric(obs.iloc[focal_rows]["fitness"], errors="coerce").to_numpy(),
        "local_character_diversity": pd.to_numeric(obs.iloc[focal_rows]["local_character_diversity"], errors="coerce").to_numpy(),
        "character_dist_of_relatives": pd.to_numeric(obs.iloc[focal_rows]["character_dist_of_relatives"], errors="coerce").to_numpy(),
        "paired_change_l1": cell_scores["paired_change_l1"].to_numpy(),
        "native_simulation_state": cell_scores["native_simulation_state"].to_numpy(),
    }
    continuous_rows = []
    for name, values in continuous.items():
        groups = [values[cluster == group] for group in range(args.clusters)]
        statistic, pvalue = kruskal(*groups, nan_policy="omit")
        for group in range(args.clusters):
            take = cluster == group
            _, one_p = mannwhitneyu(values[take], values[~take], alternative="two-sided", nan_policy="omit")
            continuous_rows.append({
                "section": section, "variable": name, "cluster": group,
                "mean": float(np.nanmean(values[take])), "median": float(np.nanmedian(values[take])),
                "rest_mean": float(np.nanmean(values[~take])),
                "difference": float(np.nanmean(values[take]) - np.nanmean(values[~take])),
                "eta_squared_all_clusters": eta_squared(values, cluster),
                "kruskal_pvalue": float(pvalue), "one_vs_rest_pvalue": float(one_p),
            })
    continuous_frame = pd.DataFrame(continuous_rows)
    continuous_frame["one_vs_rest_fdr"] = fdr_bh(continuous_frame["one_vs_rest_pvalue"].to_numpy())
    continuous_frame.to_csv(args.out_dir / "continuous_state_distance_enrichment.csv", index=False)

    distance_rows = []
    within = obs.iloc[focal_rows]["within_tumor"].astype(bool).to_numpy()
    distance = continuous["tumor_boundary_dist"]
    for group in range(args.clusters):
        take = cluster == group
        distance_rows.append({
            "section": section, "cluster": group, "n_cells": int(take.sum()),
            "tumor_distance_mean": float(np.nanmean(distance[take])),
            "tumor_distance_median": float(np.nanmedian(distance[take])),
            "tumor_distance_q25": float(np.nanquantile(distance[take], 0.25)),
            "tumor_distance_q75": float(np.nanquantile(distance[take], 0.75)),
            "within_tumor_fraction": float(np.mean(within[take])),
            "lineage_family_nmi": float(normalized_mutual_info_score(obs.iloc[focal_rows]["gt_lineage_family"].astype(str), cluster)),
            "lineage_family_ari": float(adjusted_rand_score(obs.iloc[focal_rows]["gt_lineage_family"].astype(str), cluster)),
        })
    pd.DataFrame(distance_rows).to_csv(args.out_dir / "distance_to_tumor_by_cluster.csv", index=False)

    import cellcharter as cc
    import squidpy as sq
    target_data = a[target_rows].copy()
    sq.gr.spatial_neighbors(target_data, coord_type="generic", delaunay=True, spatial_key="spatial")
    cc.gr.remove_long_links(target_data)
    adjacency = target_data.obsp["spatial_connectivities"].tocsr()
    neighbor_types = target_data.obs["gt_background_celltype"].astype(str).to_numpy()
    all_type_counts = {}
    for row in focal_target_rows:
        for neighbor in adjacency.indices[adjacency.indptr[row] : adjacency.indptr[row + 1]]:
            value = neighbor_types[neighbor]
            all_type_counts[value] = all_type_counts.get(value, 0) + 1
    total_edges = sum(all_type_counts.values())
    contact_rows = []
    for group in range(args.clusters):
        group_focal = focal_target_rows[cluster == group]
        group_counts = {}
        for row in group_focal:
            for neighbor in adjacency.indices[adjacency.indptr[row] : adjacency.indptr[row + 1]]:
                value = neighbor_types[neighbor]
                group_counts[value] = group_counts.get(value, 0) + 1
        group_edges = sum(group_counts.values())
        for celltype in sorted(all_type_counts):
            observed = group_counts.get(celltype, 0)
            expected_fraction = all_type_counts[celltype] / max(total_edges, 1)
            observed_fraction = observed / max(group_edges, 1)
            table = [
                [observed, group_edges - observed],
                [all_type_counts[celltype] - observed, (total_edges - group_edges) - (all_type_counts[celltype] - observed)],
            ]
            _, pvalue = fisher_exact(table)
            contact_rows.append({
                "section": section, "cluster": group, "neighbor_celltype": celltype,
                "observed_edges": observed, "observed_fraction": observed_fraction,
                "expected_fraction": expected_fraction,
                "log2_enrichment": float(np.log2((observed_fraction + 1e-5) / (expected_fraction + 1e-5))),
                "pvalue": float(pvalue),
            })
    contact_frame = pd.DataFrame(contact_rows)
    contact_frame["fdr"] = fdr_bh(contact_frame["pvalue"].to_numpy())
    contact_frame.to_csv(args.out_dir / "neighbor_celltype_contact_enrichment.csv", index=False)

    assignment = obs.iloc[focal_rows][[
        "section_id", "pair_id", "gt_lineage_family", "gt_lineage_clade",
        "tumor_boundary_dist", "within_tumor", "fitness", "local_character_diversity",
    ]].copy()
    assignment["salient_spatial_cluster"] = cluster.astype(str)
    assignment.to_csv(args.out_dir / "salient_spatial_cluster_assignments.csv")

    scaled_rep = StandardScaler().fit_transform(focal_rep[:, variable])
    try:
        import umap
        xy = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=args.seed).fit_transform(scaled_rep)
    except Exception:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=args.seed).fit_transform(scaled_rep)
    coords = np.asarray(a.obsm["spatial"])[focal_rows]

    program_matrix = program_frame.pivot(index="cluster", columns="program", values="difference")
    top_programs = program_matrix.var(0).sort_values(ascending=False).head(10).index
    contact_matrix = contact_frame.pivot(index="cluster", columns="neighbor_celltype", values="log2_enrichment")
    top_contacts = contact_matrix.abs().max(0).sort_values(ascending=False).head(10).index

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    scatter_categories(axes[0, 0], xy, cluster.astype(str), "s-CC UMAP — clusters")
    scatter_categories(axes[0, 1], xy, obs.iloc[focal_rows]["gt_lineage_family"].astype(str), "same UMAP — lineage family")
    scatter_categories(axes[0, 2], coords, cluster.astype(str), "physical tissue — clusters")
    axes[1, 0].boxplot([distance[cluster == group] for group in range(args.clusters)], labels=[str(group) for group in range(args.clusters)], showfliers=False)
    axes[1, 0].set_title("Distance to tumor boundary")
    axes[1, 0].set_xlabel("s-CC cluster")
    axes[1, 0].set_ylabel("tumor_boundary_dist")
    image = axes[1, 1].imshow(program_matrix[top_programs], aspect="auto", cmap="coolwarm")
    axes[1, 1].set_title("Cluster program enrichment")
    axes[1, 1].set_yticks(range(args.clusters), range(args.clusters))
    axes[1, 1].set_xticks(range(len(top_programs)), top_programs, rotation=70, ha="right", fontsize=7)
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046)
    image = axes[1, 2].imshow(contact_matrix[top_contacts], aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    axes[1, 2].set_title("Neighbor cell-type contact enrichment")
    axes[1, 2].set_yticks(range(args.clusters), range(args.clusters))
    axes[1, 2].set_xticks(range(len(top_contacts)), top_contacts, rotation=70, ha="right", fontsize=7)
    fig.colorbar(image, ax=axes[1, 2], fraction=0.046, label="log2 enrichment")
    fig.suptitle(f"PEtracer {section}: salient spatial cluster interpretation", fontsize=16)
    fig.tight_layout()
    fig.savefig(args.out_dir / "salient_spatial_cluster_enrichment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = {
        "section": section,
        "n_focal_cells": int(len(focal_rows)),
        "n_clusters": args.clusters,
        "lineage_family_ari": float(adjusted_rand_score(obs.iloc[focal_rows]["gt_lineage_family"].astype(str), cluster)),
        "lineage_family_nmi": float(normalized_mutual_info_score(obs.iloc[focal_rows]["gt_lineage_family"].astype(str), cluster)),
        "tumor_distance_eta_squared": eta_squared(distance, cluster),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
