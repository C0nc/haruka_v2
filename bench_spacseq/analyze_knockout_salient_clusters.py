"""Interpret each SPARC-seq knockout through salient CellCharter clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def enrichments(
    frame: pd.DataFrame, cluster_key: str, population: str
) -> pd.DataFrame:
    rows = []
    targets = sorted(frame.target_gene.unique())
    clusters = sorted(frame[cluster_key].unique())
    for target in targets:
        positive = frame.target_gene.to_numpy() == target
        for cluster in clusters:
            inside = frame[cluster_key].to_numpy() == cluster
            table = np.array([
                [(inside & positive).sum(), (inside & ~positive).sum()],
                [(~inside & positive).sum(), (~inside & ~positive).sum()],
            ])
            odds, pvalue = fisher_exact(table, alternative="greater")
            rows.append({
                "population": population, "representation": cluster_key[:1],
                "target_gene": target, "cluster": int(cluster),
                "inside_n": int(inside.sum()),
                "target_inside_n": int((inside & positive).sum()),
                "target_cluster_fraction": float((inside & positive).sum() / positive.sum()),
                "cluster_target_purity": float((inside & positive).sum() / inside.sum()),
                "odds_ratio": float(odds), "pvalue": float(pvalue),
            })
    result = pd.DataFrame(rows)
    result["qvalue"] = multipletests(result.pvalue, method="fdr_bh")[1]
    return result


def guide_consistency(frame: pd.DataFrame, cluster_key: str, n_clusters: int) -> pd.DataFrame:
    rows = []
    for target, part in frame.groupby("target_gene"):
        guides = sorted(part.guide_identity.unique())
        if len(guides) < 2:
            continue
        distributions = []
        for guide in guides:
            counts = np.bincount(
                part.loc[part.guide_identity == guide, cluster_key], minlength=n_clusters
            ).astype(float) + 0.5
            distributions.append(counts / counts.sum())
        pairwise = [
            float(jensenshannon(distributions[i], distributions[j], base=2))
            for i in range(len(guides)) for j in range(i + 1, len(guides))
        ]
        within_slice = []
        for _, slice_part in part.groupby("slice_id"):
            supported = [
                guide for guide in guides
                if (slice_part.guide_identity == guide).sum() >= 3
            ]
            slice_distributions = []
            for guide in supported:
                counts = np.bincount(
                    slice_part.loc[
                        slice_part.guide_identity == guide, cluster_key
                    ], minlength=n_clusters
                ).astype(float) + 0.5
                slice_distributions.append(counts / counts.sum())
            within_slice.extend(
                float(jensenshannon(slice_distributions[i], slice_distributions[j], base=2))
                for i in range(len(slice_distributions))
                for j in range(i + 1, len(slice_distributions))
            )
        rows.append({
            "representation": cluster_key[:1], "target_gene": target,
            "guides": ";".join(guides), "n_guides": len(guides),
            "mean_jensen_shannon_distance": float(np.mean(pairwise)),
            "guide_consistency_one_minus_js": float(1 - np.mean(pairwise)),
            "n_within_slice_guide_comparisons": len(within_slice),
            "within_slice_guide_consistency_one_minus_js": (
                float(1 - np.mean(within_slice)) if within_slice else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--cluster-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_cells = pd.read_csv(args.assignments)
    response = pd.read_csv(args.response)
    cluster_summary = pd.read_csv(args.cluster_summary).set_index("s_cluster")

    anchor = all_cells[all_cells.is_anchor].copy()
    anchor = anchor.rename(columns={"perturbation_neighborhood": "target_gene"})
    anchor = anchor.merge(
        response[["obs_name", "response_percentile", "responder"]],
        on="obs_name", how="inner", validate="one_to_one",
    )
    neighborhood = all_cells.rename(
        columns={"perturbation_neighborhood": "target_gene"}
    ).copy()

    enrichment = pd.concat([
        enrichments(anchor, "s_cluster", "direct_sgRNA_anchors"),
        enrichments(anchor, "z_cluster", "direct_sgRNA_anchors"),
        enrichments(neighborhood, "s_cluster", "assigned_spatial_neighborhoods"),
        enrichments(neighborhood, "z_cluster", "assigned_spatial_neighborhoods"),
    ], ignore_index=True)
    enrichment.to_csv(out / "knockout_cluster_enrichment_scc_zcc.csv", index=False)

    consistency = pd.concat([
        guide_consistency(anchor, "s_cluster", anchor.s_cluster.nunique()),
        guide_consistency(anchor, "z_cluster", anchor.z_cluster.nunique()),
    ], ignore_index=True)
    consistency.to_csv(out / "knockout_guide_consistency_scc_zcc.csv", index=False)

    s_anchor = enrichment.query(
        "population == 'direct_sgRNA_anchors' and representation == 's'"
    )
    s_neighborhood = enrichment.query(
        "population == 'assigned_spatial_neighborhoods' and representation == 's'"
    )
    rows = []
    for target, part in anchor.groupby("target_gene"):
        direct = s_anchor[s_anchor.target_gene == target].sort_values(
            ["qvalue", "odds_ratio"], ascending=[True, False]
        )
        spatial = s_neighborhood[s_neighborhood.target_gene == target].sort_values(
            ["qvalue", "odds_ratio"], ascending=[True, False]
        )
        validated = direct.query("qvalue < 0.05 and odds_ratio > 1").cluster.astype(int).tolist()
        neighborhood_hits = spatial.query("qvalue < 0.05 and odds_ratio > 1").cluster.astype(int).tolist()
        shared = [cluster for cluster in validated if cluster in neighborhood_hits]
        use = shared or validated or neighborhood_hits or [int(direct.iloc[0].cluster)]
        summaries = cluster_summary.loc[cluster_summary.index.intersection(use)]
        genes = []
        pathways = []
        for value in summaries.top_genes_after_zcc_slice.dropna():
            genes.extend(str(value).split(";"))
        for value in summaries.top_hallmarks_after_zcc_slice.dropna():
            pathways.extend(str(value).split(";"))
        top_direct = direct.iloc[0]
        top_spatial = spatial.iloc[0]
        guide_score = consistency.query(
            "representation == 's' and target_gene == @target"
        ).within_slice_guide_consistency_one_minus_js
        z_guide_score = consistency.query(
            "representation == 'z' and target_gene == @target"
        ).within_slice_guide_consistency_one_minus_js
        rows.append({
            "target_gene": target, "n_direct_anchors": int(len(part)),
            "mean_response_percentile": float(part.response_percentile.mean()),
            "responder_fraction": float(part.responder.mean()),
            "scc_within_slice_guide_consistency": float(guide_score.iloc[0]) if len(guide_score) else np.nan,
            "zcc_within_slice_guide_consistency": float(z_guide_score.iloc[0]) if len(z_guide_score) else np.nan,
            "top_direct_s_cluster": int(top_direct.cluster),
            "top_direct_odds_ratio": float(top_direct.odds_ratio),
            "top_direct_qvalue": float(top_direct.qvalue),
            "top_neighborhood_s_cluster": int(top_spatial.cluster),
            "top_neighborhood_odds_ratio": float(top_spatial.odds_ratio),
            "top_neighborhood_qvalue": float(top_spatial.qvalue),
            "direct_significant_s_clusters": ";".join(map(str, validated)),
            "neighborhood_significant_s_clusters": ";".join(map(str, neighborhood_hits)),
            "replicated_direct_and_neighborhood_clusters": ";".join(map(str, shared)),
            "cluster_residual_genes": ";".join(dict.fromkeys(genes)),
            "cluster_hallmark_programs": ";".join(dict.fromkeys(pathways)),
        })
    summary = pd.DataFrame(rows).sort_values("target_gene")
    summary.to_csv(out / "knockout_salient_cluster_biology_summary.csv", index=False)

    cluster_order = sorted(anchor.s_cluster.unique())
    direct_profile = pd.crosstab(
        anchor.target_gene, anchor.s_cluster, normalize="index"
    ).reindex(columns=cluster_order, fill_value=0)
    neighborhood_profile = pd.crosstab(
        neighborhood.target_gene, neighborhood.s_cluster, normalize="index"
    ).reindex(index=direct_profile.index, columns=cluster_order, fill_value=0)
    direct_profile.to_csv(out / "knockout_direct_anchor_scc_profiles.csv")
    neighborhood_profile.to_csv(out / "knockout_neighborhood_scc_profiles.csv")

    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    sns.heatmap(direct_profile, cmap="rocket", ax=axes[0, 0], cbar_kws={"label": "Within-KO fraction"})
    axes[0, 0].set(title="Direct sgRNA anchors across salient clusters", xlabel="s-CC cluster", ylabel="Knockout")
    sns.heatmap(neighborhood_profile, cmap="mako", ax=axes[0, 1], cbar_kws={"label": "Within-KO fraction"})
    axes[0, 1].set(title="Assigned perturbation neighborhoods", xlabel="s-CC cluster", ylabel="Knockout")

    plot_enrichment = s_anchor.pivot(index="target_gene", columns="cluster", values="odds_ratio")
    plot_enrichment = np.log2(plot_enrichment.replace([np.inf, 0], np.nan)).clip(-3, 3)
    sns.heatmap(plot_enrichment, cmap="vlag", center=0, ax=axes[1, 0], cbar_kws={"label": "log2 odds ratio"})
    axes[1, 0].set(title="Direct-anchor enrichment", xlabel="s-CC cluster", ylabel="Knockout")

    response_plot = summary.sort_values("mean_response_percentile")
    axes[1, 1].barh(response_plot.target_gene, response_plot.mean_response_percentile, color="#E45756")
    axes[1, 1].scatter(response_plot.responder_fraction, response_plot.target_gene, color="black", label="Responder fraction")
    axes[1, 1].set(title="Expression-defined knockout response", xlabel="Fraction / mean percentile", xlim=(0, 1))
    axes[1, 1].legend(frameon=False)
    fig.suptitle("SPARC-seq knockout biology resolved by salient CellCharter clusters", fontsize=15)
    fig.savefig(out / "knockout_salient_cluster_analysis.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    result = {
        "n_direct_anchors": int(len(anchor)),
        "n_knockouts": int(anchor.target_gene.nunique()),
        "interpretation": (
            "Knockout-specific salient states require direct-anchor enrichment; concordant "
            "neighborhood enrichment is supportive spatial evidence."
        ),
        "summary_file": str(out / "knockout_salient_cluster_biology_summary.csv"),
    }
    (out / "knockout_salient_cluster_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
