"""Find spatial biology captured by salient CellCharter but not background.

The analysis uses every condition cell for spatial-domain discovery.  Direct
sgRNA identities are used only on anchor cells as a validation annotation.
"Unique" means association remaining after conditioning on the other
CellCharter clustering and slice, or expression variance added after the same
nuisance variables.  This is an exploratory association analysis: the RNA
matrix also contributed to the learned representations.
"""

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
from scipy import sparse
from scipy.stats import f as f_distribution
from scipy.stats import hypergeom, ttest_ind
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def conditional_mi(y: np.ndarray, x: np.ndarray, strata: np.ndarray) -> float:
    """Empirical conditional mutual information in nats."""
    y = np.asarray(y).astype(str)
    x = np.asarray(x).astype(str)
    strata = np.asarray(strata).astype(str)
    total = len(y)
    value = 0.0
    for level in np.unique(strata):
        take = strata == level
        if take.sum() > 1:
            value += take.mean() * mutual_info_score(y[take], x[take])
    return float(value)


def permuted_conditional_mi(
    y: np.ndarray,
    x: np.ndarray,
    strata: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    observed = conditional_mi(y, x, strata)
    strata = np.asarray(strata).astype(str)
    groups = [np.flatnonzero(strata == level) for level in np.unique(strata)]
    null = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        shuffled = np.asarray(x).copy()
        for group in groups:
            shuffled[group] = rng.permutation(shuffled[group])
        null[iteration] = conditional_mi(y, shuffled, strata)
    pvalue = (1 + np.sum(null >= observed)) / (n_permutations + 1)
    return observed, float(null.mean()), float(pvalue)


def design_matrix(*columns: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({f"x{i}": np.asarray(x).astype(str) for i, x in enumerate(columns)})
    encoded = pd.get_dummies(frame, drop_first=True, dtype=float)
    return np.column_stack([np.ones(len(frame)), encoded.to_numpy(float)])


def nested_partial_r2(
    expression: np.ndarray,
    base: np.ndarray,
    full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_fit = base @ np.linalg.lstsq(base, expression, rcond=None)[0]
    full_fit = full @ np.linalg.lstsq(full, expression, rcond=None)[0]
    sse_base = np.square(expression - base_fit).sum(0)
    sse_full = np.square(expression - full_fit).sum(0)
    gain = np.maximum(sse_base - sse_full, 0)
    partial = np.divide(gain, sse_base, out=np.zeros_like(gain), where=sse_base > 0)
    df1 = max(np.linalg.matrix_rank(full) - np.linalg.matrix_rank(base), 1)
    df2 = max(len(expression) - np.linalg.matrix_rank(full), 1)
    statistic = (gain / df1) / np.maximum(sse_full / df2, 1e-12)
    pvalue = f_distribution.sf(statistic, df1, df2)
    return partial, pvalue, expression - base_fit


def read_gmt(path: Path, universe: set[str]) -> dict[str, set[str]]:
    terms = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            genes = {gene.upper() for gene in fields[2:] if gene.upper() in universe}
            if 10 <= len(genes) <= 500:
                terms[fields[0]] = genes
    return terms


def enrich_markers(
    markers: pd.DataFrame,
    terms: dict[str, set[str]],
    universe: set[str],
) -> pd.DataFrame:
    rows = []
    for cluster, part in markers.groupby("cluster"):
        selected = set(part.query("qvalue < 0.05 and residual_delta > 0").gene.str.upper().head(250))
        if len(selected) < 3:
            continue
        for term, members in terms.items():
            overlap = selected & members
            if len(overlap) < 3:
                continue
            pvalue = hypergeom.sf(
                len(overlap) - 1, len(universe), len(members), len(selected)
            )
            rows.append({
                "cluster": int(cluster), "term": term,
                "overlap_n": len(overlap), "selected_n": len(selected),
                "term_n": len(members), "pvalue": float(pvalue),
                "genes": ";".join(sorted(overlap)),
            })
    result = pd.DataFrame(rows)
    if len(result):
        result["qvalue"] = result.groupby("cluster").pvalue.transform(
            lambda x: multipletests(x, method="fdr_bh")[1]
        )
        result["minus_log10_q"] = -np.log10(result.qvalue.clip(lower=1e-300))
        result = result.sort_values(["cluster", "qvalue", "pvalue"])
    return result


def neighborhood_agreement(
    xy: np.ndarray, clusters: np.ndarray, slices: np.ndarray, k: int
) -> dict[str, float]:
    same = []
    expected = []
    for slice_id in np.unique(slices):
        take = np.flatnonzero(slices == slice_id)
        n_neighbors = min(k + 1, len(take))
        index = NearestNeighbors(n_neighbors=n_neighbors).fit(xy[take])
        neighbors = index.kneighbors(return_distance=False)[:, 1:]
        labels = clusters[take]
        same.append((labels[:, None] == labels[neighbors]).mean(1))
        probabilities = pd.Series(labels).value_counts(normalize=True).to_numpy()
        expected.extend([float(np.square(probabilities).sum())] * len(take))
    observed = float(np.concatenate(same).mean())
    baseline = float(np.mean(expected))
    return {
        "neighbor_agreement": observed,
        "frequency_expected": baseline,
        "excess_agreement": observed - baseline,
    }


def residual_markers(
    residual: np.ndarray, clusters: np.ndarray, genes: np.ndarray
) -> pd.DataFrame:
    rows = []
    for cluster in sorted(np.unique(clusters)):
        inside = clusters == cluster
        statistic, pvalue = ttest_ind(
            residual[inside], residual[~inside], axis=0, equal_var=False,
            nan_policy="omit",
        )
        delta = residual[inside].mean(0) - residual[~inside].mean(0)
        qvalue = multipletests(np.nan_to_num(pvalue, nan=1.0), method="fdr_bh")[1]
        order = np.argsort(-delta)
        for index in order:
            rows.append({
                "cluster": int(cluster), "gene": str(genes[index]),
                "residual_delta": float(delta[index]),
                "t_statistic": float(statistic[index]),
                "qvalue": float(qvalue[index]),
            })
    return pd.DataFrame(rows)


def spatial_cluster_figure(
    frame: pd.DataFrame, out: Path, point_size: float = 3.0
) -> None:
    slices = sorted(frame.slice_id.unique())
    fig, axes = plt.subplots(2, len(slices), figsize=(5.2 * len(slices), 9.0), constrained_layout=True)
    for column, slice_id in enumerate(slices):
        part = frame[frame.slice_id == slice_id]
        for row, key in enumerate(("z_cluster", "s_cluster")):
            ax = axes[row, column]
            ax.scatter(
                part.array_col, -part.array_row, c=part[key], cmap="tab20",
                s=point_size, linewidths=0, rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set(xticks=[], yticks=[])
            ax.set_title(f"{slice_id}: {'background' if row == 0 else 'salient'} CC clusters")
    fig.suptitle("SPARC-seq condition cells: physical spatial domains", fontsize=15)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--gmt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.emb_dir)
    adata = ad.read_h5ad(args.data)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    condition = adata.obs.condition.astype(str).to_numpy() == "disease"
    adata = adata[condition].copy()
    obs = adata.obs.copy()

    z = StandardScaler().fit_transform(np.load(root / "spatial.npy")[condition])
    s = StandardScaler().fit_transform(np.load(root / "salient_spatial.npy")[condition])
    z_cluster = KMeans(args.clusters, n_init=100, random_state=args.seed).fit_predict(z)
    s_cluster = KMeans(args.clusters, n_init=100, random_state=args.seed).fit_predict(s)
    slices = obs.slice_id.astype(str).to_numpy()
    z_strata = np.char.add(np.char.add(slices, "|z"), z_cluster.astype(str))
    s_strata = np.char.add(np.char.add(slices, "|s"), s_cluster.astype(str))

    frame = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "slice_id": slices,
        "array_row": obs.array_row.to_numpy(float),
        "array_col": obs.array_col.to_numpy(float),
        "z_cluster": z_cluster,
        "s_cluster": s_cluster,
        "cell_type": obs.cell_type.astype(str).to_numpy(),
        "niche": obs.niche.astype(str).to_numpy(),
        "perturbation_neighborhood": obs.function.astype(str).to_numpy(),
        "is_anchor": obs.is_anchor.to_numpy(bool),
        "guide_identity": obs.guide_identity.astype(str).to_numpy(),
    })
    frame.to_csv(out / "zcc_scc_all_condition_cluster_assignments.csv", index=False)

    rng = np.random.default_rng(args.seed)
    tests = {
        "cell_type": np.ones(len(frame), bool),
        "niche": frame.niche.to_numpy() != "Unknown",
        "perturbation_neighborhood": np.ones(len(frame), bool),
        "direct_sgRNA": frame.is_anchor.to_numpy(bool) & (frame.guide_identity.to_numpy() != ""),
    }
    conditional_rows = []
    for annotation, use in tests.items():
        values = (
            frame.guide_identity.to_numpy()[use]
            if annotation == "direct_sgRNA"
            else frame[annotation].to_numpy()[use]
        )
        for representation, clusters, strata in (
            ("salient_unique_after_background_and_slice", s_cluster[use], z_strata[use]),
            ("background_unique_after_salient_and_slice", z_cluster[use], s_strata[use]),
        ):
            observed, null_mean, pvalue = permuted_conditional_mi(
                values, clusters, strata, args.permutations, rng
            )
            conditional_rows.append({
                "annotation": annotation, "representation": representation,
                "n": int(use.sum()), "conditional_mi_nats": observed,
                "permutation_null_mean": null_mean, "permutation_pvalue": pvalue,
                "excess_conditional_mi_nats": observed - null_mean,
            })
    conditional = pd.DataFrame(conditional_rows)
    conditional["qvalue"] = multipletests(conditional.permutation_pvalue, method="fdr_bh")[1]
    conditional.to_csv(out / "conditional_annotation_information.csv", index=False)

    counts = adata.layers["counts"]
    totals = np.asarray(counts.sum(1)).ravel().clip(min=1)
    expression = counts.multiply(1e4 / totals[:, None])
    if sparse.issparse(expression):
        expression.data = np.log1p(expression.data)
        expression = expression.toarray()
    else:
        expression = np.log1p(expression)
    genes = np.asarray(adata.var_names.astype(str))
    full = design_matrix(slices, z_cluster, s_cluster)
    z_base = design_matrix(slices, z_cluster)
    s_base = design_matrix(slices, s_cluster)
    s_partial, s_pvalue, z_residual = nested_partial_r2(expression, z_base, full)
    z_partial, z_pvalue, s_residual = nested_partial_r2(expression, s_base, full)
    gene_table = pd.DataFrame({
        "gene": genes,
        "salient_partial_r2_after_background_slice": s_partial,
        "salient_qvalue": multipletests(s_pvalue, method="fdr_bh")[1],
        "background_partial_r2_after_salient_slice": z_partial,
        "background_qvalue": multipletests(z_pvalue, method="fdr_bh")[1],
    })
    gene_table["salient_minus_background_partial_r2"] = s_partial - z_partial
    gene_table = gene_table.sort_values("salient_minus_background_partial_r2", ascending=False)
    gene_table.to_csv(out / "unique_expression_variance_by_gene.csv", index=False)

    markers = residual_markers(z_residual, s_cluster, genes)
    markers.to_csv(out / "scc_markers_after_zcc_and_slice.csv", index=False)
    universe = {gene.upper() for gene in genes}
    enrichments = enrich_markers(markers, read_gmt(Path(args.gmt), universe), universe)
    enrichments.to_csv(out / "scc_unique_hallmark_enrichment.csv", index=False)

    cluster_rows = []
    for cluster in sorted(frame.s_cluster.unique()):
        part = frame[frame.s_cluster == cluster]
        anchors = part[part.is_anchor]

        def dominant(column: str, source: pd.DataFrame = part) -> tuple[str, float]:
            frequency = source[column].value_counts(normalize=True)
            if not len(frequency):
                return "NA", 0.0
            return str(frequency.index[0]), float(frequency.iloc[0])

        perturbation, perturbation_fraction = dominant("perturbation_neighborhood")
        cell_type, cell_type_fraction = dominant("cell_type")
        niche, niche_fraction = dominant("niche")
        slice_id, slice_fraction = dominant("slice_id")
        guide, guide_fraction = dominant("guide_identity", anchors)
        top_genes = markers.query(
            "cluster == @cluster and qvalue < 0.05 and residual_delta > 0"
        ).head(12).gene
        top_pathways = enrichments.query(
            "cluster == @cluster and qvalue < 0.1"
        ).head(5).term if len(enrichments) else []
        cluster_rows.append({
            "s_cluster": int(cluster), "n": int(len(part)),
            "dominant_perturbation_neighborhood": perturbation,
            "dominant_perturbation_fraction": perturbation_fraction,
            "dominant_direct_guide": guide,
            "dominant_direct_guide_fraction": guide_fraction,
            "dominant_cell_type": cell_type,
            "dominant_cell_type_fraction": cell_type_fraction,
            "dominant_niche": niche, "dominant_niche_fraction": niche_fraction,
            "dominant_slice": slice_id, "dominant_slice_fraction": slice_fraction,
            "top_genes_after_zcc_slice": ";".join(top_genes),
            "top_hallmarks_after_zcc_slice": ";".join(top_pathways),
        })
    pd.DataFrame(cluster_rows).to_csv(out / "scc_unique_cluster_summary.csv", index=False)

    xy = obs[["array_row", "array_col"]].to_numpy(float)
    spatial = {
        "background": neighborhood_agreement(xy, z_cluster, slices, args.neighbors),
        "salient": neighborhood_agreement(xy, s_cluster, slices, args.neighbors),
    }
    metrics = {
        "population": "all condition cells",
        "n_cells": int(len(frame)),
        "n_clusters_each": int(args.clusters),
        "definition_of_unique": (
            "conditional MI after the other clustering and slice; expression partial R2 "
            "after the other clustering and slice"
        ),
        "spatial_neighbor_agreement": spatial,
        "n_salient_unique_genes_fdr_0_05": int(np.sum(gene_table.salient_qvalue < 0.05)),
        "n_background_unique_genes_fdr_0_05": int(np.sum(gene_table.background_qvalue < 0.05)),
        "median_salient_partial_r2": float(np.median(s_partial)),
        "median_background_partial_r2": float(np.median(z_partial)),
        "conditional_information_excess_nats": {
            row.annotation: {
                "salient_unique": float(row.excess_conditional_mi_nats),
                "background_unique": float(
                    conditional.loc[
                        (conditional.annotation == row.annotation)
                        & (conditional.representation == "background_unique_after_salient_and_slice"),
                        "excess_conditional_mi_nats",
                    ].iloc[0]
                ),
            }
            for row in conditional.query(
                "representation == 'salient_unique_after_background_and_slice'"
            ).itertuples()
        },
        "circularity_note": (
            "Exploratory: expression contributed to both learned embeddings; direct sgRNA "
            "identity and physical spatial coherence are the less circular validations."
        ),
        "gmt_note": "Mouse symbols were uppercased for approximate human Hallmark mapping.",
    }
    (out / "unique_salient_spatial_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    spatial_cluster_figure(frame, out / "physical_zcc_scc_clusters.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    plot_cmi = conditional.copy()
    plot_cmi["representation"] = plot_cmi.representation.map({
        "salient_unique_after_background_and_slice": "s-CC unique | z-CC, slice",
        "background_unique_after_salient_and_slice": "z-CC unique | s-CC, slice",
    })
    sns.barplot(
        data=plot_cmi, x="annotation", y="excess_conditional_mi_nats",
        hue="representation", palette=["#E45756", "#4C78A8"], ax=axes[0, 0],
    )
    axes[0, 0].tick_params(axis="x", rotation=18)
    axes[0, 0].set(
        title="Information unique to each spatial clustering",
        ylabel="Conditional MI above permutation null (nats)", xlabel="",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    top = gene_table.head(20).sort_values("salient_minus_background_partial_r2")
    axes[0, 1].barh(top.gene, top.salient_minus_background_partial_r2, color="#E45756")
    axes[0, 1].axvline(0, color="black", lw=0.8)
    axes[0, 1].set(title="Top s-CC-specific expression associations", xlabel="Partial R²(s | z,slice) − Partial R²(z | s,slice)")

    function_table = pd.crosstab(
        frame.s_cluster, frame.perturbation_neighborhood, normalize="index"
    )
    sns.heatmap(function_table, cmap="mako", ax=axes[1, 0])
    axes[1, 0].set(title="Perturbation-neighborhood composition", xlabel="Neighborhood target", ylabel="s-CC cluster")

    if len(enrichments):
        top_terms = enrichments.query("qvalue < 0.1").groupby("cluster", group_keys=False).head(2)
        if len(top_terms):
            pathway = top_terms.pivot_table(
                index="term", columns="cluster", values="minus_log10_q", fill_value=0, aggfunc="max"
            )
            sns.heatmap(pathway, cmap="rocket_r", ax=axes[1, 1])
            axes[1, 1].set(title="s-CC programs remaining after z-CC + slice", xlabel="s-CC cluster", ylabel="Hallmark")
        else:
            axes[1, 1].text(0.5, 0.5, "No Hallmark term at q<0.1", ha="center", va="center")
            axes[1, 1].set_axis_off()
    else:
        axes[1, 1].text(0.5, 0.5, "No eligible enrichment", ha="center", va="center")
        axes[1, 1].set_axis_off()
    fig.suptitle("SPARC-seq: biological properties uniquely revealed by salient CellCharter", fontsize=15)
    fig.savefig(out / "unique_salient_spatial_biology.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
