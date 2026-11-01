"""Characterize what unsupervised salient-CellCharter clusters represent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import fisher_exact, kruskal, ttest_ind
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


PROGRAMS = {
    "cell_cycle": ["Mki67", "Top2a", "Pcna", "Ccnb1", "Ccnb2", "Cdk1", "Ube2c"],
    "stress_hypoxia": ["Hif1a", "Vegfa", "Bnip3", "Ddit4", "Atf3", "Fos", "Jun", "Hspa1a", "Hspa1b"],
    "ecm_emt": ["Col1a1", "Col1a2", "Col3a1", "Vim", "Fn1", "Tnc", "Mmp2", "Dpt", "Thbs2"],
    "epithelial": ["Epcam", "Krt8", "Krt18", "Krt19", "Krt7"],
    "interferon": ["Stat1", "Isg15", "Ifit1", "Ifit3", "Cxcl10", "B2m"],
    "lysosome": ["Lamp1", "Lamp2", "Ctsb", "Ctsd", "Lgals3", "Laptm5"],
}


def entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values.astype(str), return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probability = counts / counts.sum()
    return float(-(probability * np.log(probability)).sum() / np.log(len(counts)))


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, float)
    groups = np.asarray(groups).astype(str)
    total = np.square(values - values.mean()).sum()
    if total == 0:
        return 0.0
    between = sum(
        (groups == level).sum() * (values[groups == level].mean() - values.mean()) ** 2
        for level in np.unique(groups)
    )
    return float(between / total)


def enrichment_table(clusters: np.ndarray, values: np.ndarray, label: str) -> pd.DataFrame:
    rows = []
    values = np.asarray(values).astype(str)
    for cluster in sorted(np.unique(clusters)):
        inside = clusters == cluster
        for level in sorted(np.unique(values)):
            positive = values == level
            table = np.array([
                [(inside & positive).sum(), (inside & ~positive).sum()],
                [(~inside & positive).sum(), (~inside & ~positive).sum()],
            ])
            odds, pvalue = fisher_exact(table, alternative="greater")
            rows.append({
                "cluster": int(cluster), "annotation": label, "level": level,
                "inside_n": int(inside.sum()), "inside_positive": int((inside & positive).sum()),
                "inside_fraction": float((inside & positive).sum() / inside.sum()),
                "global_fraction": float(positive.mean()), "odds_ratio": float(odds),
                "pvalue": float(pvalue),
            })
    frame = pd.DataFrame(rows)
    frame["qvalue"] = multipletests(frame.pvalue, method="fdr_bh")[1]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--response-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.emb_dir)
    adata = ad.read_h5ad(args.data)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    obs = adata.obs
    function = obs.function.astype(str).to_numpy()
    guide = obs.guide_identity.astype(str).to_numpy()
    take = obs.is_anchor.to_numpy(bool) & (function != "NT") & (guide != "")
    names = adata.obs_names.to_numpy()[take].astype(str)

    scc = StandardScaler().fit_transform(np.load(root / "salient_spatial.npy")[take])
    cluster = KMeans(args.clusters, n_init=100, random_state=args.seed).fit_predict(scc)
    response = pd.read_csv(args.response_csv).set_index("obs_name").loc[names]
    response_percentile_column = (
        "response_percentile"
        if "response_percentile" in response.columns
        else "intrinsic_response_percentile"
    )
    frame = pd.DataFrame({
        "obs_name": names,
        "cluster": cluster,
        "guide_identity": guide[take],
        "target_gene": function[take],
        "slice_id": obs.slice_id.astype(str).to_numpy()[take],
        "niche": obs.niche.astype(str).to_numpy()[take],
        "responder": response.responder.astype(bool).to_numpy(),
        "response_percentile": response[response_percentile_column].to_numpy(float),
    })
    frame.to_csv(out / "scc_cluster_assignments.csv", index=False)

    enrichments = pd.concat([
        enrichment_table(cluster, frame.guide_identity.to_numpy(), "guide_identity"),
        enrichment_table(cluster, frame.target_gene.to_numpy(), "target_gene"),
        enrichment_table(cluster, frame.slice_id.to_numpy(), "slice_id"),
        enrichment_table(cluster, frame.niche.to_numpy(), "niche"),
        enrichment_table(cluster, frame.responder.astype(str).to_numpy(), "responder"),
    ], ignore_index=True)
    enrichments.to_csv(out / "scc_cluster_enrichments.csv", index=False)

    counts = adata.layers["counts"][take]
    totals = np.asarray(counts.sum(1)).ravel().clip(min=1)
    expression = counts.multiply(1e4 / totals[:, None])
    if sparse.issparse(expression):
        expression.data = np.log1p(expression.data)
        expression = expression.toarray()
    else:
        expression = np.log1p(expression)
    genes = np.asarray(adata.var_names.astype(str))
    gene_index = {gene: index for index, gene in enumerate(genes)}
    expression_z = StandardScaler().fit_transform(expression)

    program_scores = {}
    program_genes = {}
    for program, requested in PROGRAMS.items():
        present = [gene for gene in requested if gene in gene_index]
        program_genes[program] = present
        program_scores[program] = (
            expression_z[:, [gene_index[gene] for gene in present]].mean(1)
            if present else np.zeros(len(frame))
        )
        frame[program] = program_scores[program]
    frame.to_csv(out / "scc_cluster_assignments.csv", index=False)

    marker_rows = []
    for label in sorted(np.unique(cluster)):
        inside = cluster == label
        statistic, pvalue = ttest_ind(
            expression[inside], expression[~inside], axis=0, equal_var=False,
            nan_policy="omit",
        )
        delta = expression[inside].mean(0) - expression[~inside].mean(0)
        qvalue = multipletests(np.nan_to_num(pvalue, nan=1.0), method="fdr_bh")[1]
        for index in np.argsort(-delta)[:30]:
            marker_rows.append({
                "cluster": int(label), "gene": str(genes[index]),
                "mean_log_expression_delta": float(delta[index]),
                "t_statistic": float(statistic[index]), "qvalue": float(qvalue[index]),
            })
    markers = pd.DataFrame(marker_rows)
    markers.to_csv(out / "scc_cluster_top_markers.csv", index=False)

    summaries = []
    for label in sorted(np.unique(cluster)):
        part = frame[frame.cluster == label]
        row = {
            "cluster": int(label), "n": int(len(part)),
            "dominant_guide": part.guide_identity.value_counts().index[0],
            "dominant_guide_fraction": float(part.guide_identity.value_counts(normalize=True).iloc[0]),
            "guide_entropy": entropy(part.guide_identity.to_numpy()),
            "dominant_target": part.target_gene.value_counts().index[0],
            "dominant_target_fraction": float(part.target_gene.value_counts(normalize=True).iloc[0]),
            "target_entropy": entropy(part.target_gene.to_numpy()),
            "dominant_slice": part.slice_id.value_counts().index[0],
            "dominant_slice_fraction": float(part.slice_id.value_counts(normalize=True).iloc[0]),
            "slice_entropy": entropy(part.slice_id.to_numpy()),
            "dominant_niche": part.niche.value_counts().index[0],
            "dominant_niche_fraction": float(part.niche.value_counts(normalize=True).iloc[0]),
            "responder_fraction": float(part.responder.mean()),
            "median_response_percentile": float(part.response_percentile.median()),
            "top_markers": ";".join(markers[markers.cluster == label].gene.head(8)),
        }
        row.update({f"{program}_mean": float(part[program].mean()) for program in PROGRAMS})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "scc_cluster_summary.csv", index=False)

    categorical = {}
    for column in ("guide_identity", "target_gene", "slice_id", "niche", "responder"):
        use = np.ones(len(frame), bool)
        if column == "niche":
            use = frame.niche.to_numpy() != "Unknown"
        categorical[column] = {
            "n": int(use.sum()),
            "ari": float(adjusted_rand_score(frame.loc[use, column].astype(str), cluster[use])),
            "nmi": float(normalized_mutual_info_score(frame.loc[use, column].astype(str), cluster[use])),
        }
    continuous = {
        "response_percentile_eta_squared": eta_squared(frame.response_percentile, cluster),
        "response_percentile_kruskal_p": float(kruskal(*[
            frame.loc[frame.cluster == label, "response_percentile"]
            for label in sorted(frame.cluster.unique())
        ]).pvalue),
        "program_eta_squared": {
            program: eta_squared(frame[program], cluster) for program in PROGRAMS
        },
    }
    result = {
        "population": "targeting sgRNA anchors only",
        "n_cells": int(len(frame)), "n_clusters": int(args.clusters),
        "categorical_association": categorical,
        "continuous_association": continuous,
        "program_genes_present": program_genes,
    }
    (out / "scc_cluster_property_metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    target_table = pd.crosstab(frame.cluster, frame.target_gene, normalize="index")
    slice_table = pd.crosstab(frame.cluster, frame.slice_id, normalize="index")
    program_table = frame.groupby("cluster")[list(PROGRAMS)].mean()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    target_table.plot(kind="bar", stacked=True, ax=axes[0, 0], colormap="tab20")
    axes[0, 0].set(title="Target-gene composition", ylabel="Cluster fraction")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    slice_table.plot(kind="bar", stacked=True, ax=axes[0, 1], colormap="Set2")
    axes[0, 1].set(title="Slice composition", ylabel="Cluster fraction")
    axes[0, 1].legend(fontsize=8, frameon=False)
    sns.barplot(data=summary, x="cluster", y="responder_fraction", color="#E45756", ax=axes[1, 0])
    axes[1, 0].axhline(frame.responder.mean(), color="black", ls="--", lw=1)
    axes[1, 0].set(title="Independent responder enrichment", ylabel="Responder fraction")
    sns.heatmap(program_table.T, cmap="vlag", center=0, ax=axes[1, 1])
    axes[1, 1].set(title="Expression programs", xlabel="s-CC cluster", ylabel="")
    fig.suptitle("SPARC decoder-only Haruka: salient-CellCharter cluster properties")
    fig.savefig(out / "scc_cluster_properties.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
