"""Create a compact cross-section PEtracer salient spatial discovery report."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.environ.get(
    "HARUKA_PETRACER_RESULTS",
    Path(__file__).resolve().parent / "results_lambda2_22k",
))
OUT = ROOT / "summary" / "salient_spatial_enrichment"
OUT.mkdir(parents=True, exist_ok=True)
RUN = os.environ.get("HARUKA_PETRACER_RUN", "haruka_cellcharter_asym_seed0")

rows = []
effects = []
for directory in sorted(ROOT.glob(f"*/{RUN}/salient_spatial_enrichment")):
    assignments = pd.read_csv(directory / "salient_spatial_cluster_assignments.csv")
    programs = pd.read_csv(directory / "program_enrichment.csv")
    contacts = pd.read_csv(directory / "neighbor_celltype_contact_enrichment.csv")
    distance = pd.read_csv(directory / "distance_to_tumor_by_cluster.csv")
    genes = pd.read_csv(directory / "top_paired_delta_genes.csv")
    continuous = pd.read_csv(directory / "continuous_state_distance_enrichment.csv")
    section = str(distance["section"].iloc[0])
    effects.append(continuous.drop_duplicates(["section", "variable"])[[
        "section", "variable", "eta_squared_all_clusters", "kruskal_pvalue"
    ]])
    for cluster in sorted(distance["cluster"].unique()):
        take = assignments[assignments["salient_spatial_cluster"].astype(int).eq(cluster)]
        families = take["gt_lineage_family"].astype(str).value_counts(normalize=True)
        program_take = programs[programs["cluster"].eq(cluster) & programs["program"].str.startswith("response_")]
        significant_programs = program_take[program_take["fdr"].lt(0.05)]
        if significant_programs.empty:
            significant_programs = program_take
        top_program = significant_programs.sort_values("difference", ascending=False).iloc[0]
        native_take = programs[programs["cluster"].eq(cluster) & programs["program"].str.startswith("native_")]
        significant_native = native_take[native_take["fdr"].lt(0.05)]
        if significant_native.empty:
            significant_native = native_take
        top_native = significant_native.sort_values("difference", ascending=False).iloc[0]
        contact_take = contacts[
            contacts["cluster"].eq(cluster)
            & contacts["fdr"].lt(0.05)
            & contacts["observed_edges"].ge(10)
        ]
        if contact_take.empty:
            contact_take = contacts[contacts["cluster"].eq(cluster) & contacts["observed_edges"].ge(10)]
        top_contact = contact_take.sort_values("log2_enrichment", ascending=False).iloc[0]
        gene_take = genes[genes["cluster"].eq(cluster) & genes["fdr"].lt(0.05)]
        if gene_take.empty:
            gene_take = genes[genes["cluster"].eq(cluster)]
        top_gene = gene_take.sort_values("delta_difference", ascending=False).iloc[0]
        dist = distance[distance["cluster"].eq(cluster)].iloc[0]
        rows.append({
            "section": section,
            "cluster": int(cluster),
            "n_cells": int(len(take)),
            "dominant_lineage_family": families.index[0],
            "lineage_family_purity": float(families.iloc[0]),
            "top_response_program": str(top_program["program"]),
            "response_difference": float(top_program["difference"]),
            "response_fdr": float(top_program["fdr"]),
            "top_native_state": str(top_native["program"]),
            "native_state_difference": float(top_native["difference"]),
            "native_state_fdr": float(top_native["fdr"]),
            "top_paired_delta_gene": str(top_gene["gene"]),
            "gene_delta_difference": float(top_gene["delta_difference"]),
            "gene_fdr": float(top_gene["fdr"]),
            "enriched_neighbor_celltype": str(top_contact["neighbor_celltype"]),
            "neighbor_log2_enrichment": float(top_contact["log2_enrichment"]),
            "neighbor_fdr": float(top_contact["fdr"]),
            "tumor_distance_median": float(dist["tumor_distance_median"]),
            "tumor_distance_q25": float(dist["tumor_distance_q25"]),
            "tumor_distance_q75": float(dist["tumor_distance_q75"]),
        })

discovery = pd.DataFrame(rows).sort_values(["section", "cluster"])
discovery.to_csv(OUT / "cluster_discovery_summary.csv", index=False)
effect_frame = pd.concat(effects, ignore_index=True)
effect_frame.to_csv(OUT / "continuous_effect_sizes_by_section.csv", index=False)
effect_summary = effect_frame.groupby("variable", as_index=False).agg(
    eta_squared_mean=("eta_squared_all_clusters", "mean"),
    eta_squared_std=("eta_squared_all_clusters", "std"),
)
effect_summary = effect_summary.sort_values("eta_squared_mean", ascending=False)
effect_summary.to_csv(OUT / "continuous_effect_sizes_mean_std.csv", index=False)

fig, axis = plt.subplots(figsize=(10, 5.5))
axis.barh(effect_summary["variable"], effect_summary["eta_squared_mean"], xerr=effect_summary["eta_squared_std"], color="#7469B6", alpha=0.85)
axis.invert_yaxis()
axis.set_xlabel("Variance explained by salient spatial cluster, η² (mean ± SD)")
axis.set_title("PEtracer salient spatial cluster state/distance associations")
fig.tight_layout()
fig.savefig(OUT / "continuous_effect_sizes.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(effect_summary.to_string(index=False))
print("\nCLUSTER DISCOVERY\n")
print(discovery.to_string(index=False))
