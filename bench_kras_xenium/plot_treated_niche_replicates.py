#!/usr/bin/env python3
"""Add annotated UMAP and treated-section niche-proportion panels."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(os.environ.get(
    "HARUKA_KRAS_ANALYSIS_ROOT",
    "/home/yancui/haruka_v2/bench_kras_xenium/analysis/condition_only_salient",
))
GLOB = ROOT / "global" / "tables"
OUT = ROOT / "responder_niche_interpretation"
TAB, FIG = OUT / "tables", OUT / "figures"

assign = pd.read_csv(GLOB / "cell_cluster_assignments.csv.gz", index_col=0)
treated = assign.query("condition_label == 'KRASi_treated'").copy()
summary = pd.read_csv(TAB / "zcc_scc_niche_annotations_and_response.csv")

records = []
for rep, col in [("z_cc", "z_cc_cluster"), ("s_cc", "s_cc_cluster")]:
    temp = treated[treated[col] >= 0].groupby(["slice_id", col], observed=True).size().rename("n").reset_index()
    temp["fraction_within_section"] = temp["n"] / temp.groupby("slice_id")["n"].transform("sum")
    for row in temp.itertuples(index=False):
        records.append({"representation": rep, "slice_id": row.slice_id,
                        "cluster": int(getattr(row, col)), "n": int(row.n),
                        "fraction_within_section": float(row.fraction_within_section)})
abundance = pd.DataFrame(records)
abundance.to_csv(TAB / "zcc_scc_niche_fraction_by_treated_section.csv", index=False)

sns.set_theme(style="white", font_scale=0.9)
fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
for ax, rep, title in [(axes[0], "z_cc", "z-CC background niches"),
                       (axes[1], "s_cc", "s-CC salient niches")]:
    p = abundance[abundance.representation == rep].pivot(index="cluster", columns="slice_id", values="fraction_within_section").fillna(0)
    sns.heatmap(p, cmap="mako", annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "Fraction of all cells in section"})
    ax.set_title(title); ax.set_xlabel("Treated section (JB_M*)"); ax.set_ylabel("Niche cluster")
fig.suptitle("Treated-section niche proportions: anatomy/section dependence", fontsize=15)
fig.savefig(FIG / "treated_only_niche_proportion_by_replicate.png", dpi=220, bbox_inches="tight")
plt.close(fig)

coords = pd.read_csv(GLOB / "treated_umap_coordinates.csv.gz")
rows = coords["row"].astype(int).to_numpy()
sample = assign.iloc[rows].copy()
fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
for ax, rep, col, xcol, ycol, prefix in [
    (axes[0], "z_cc", "z_cc_cluster", "z_umap1", "z_umap2", "Z"),
    (axes[1], "s_cc", "s_cc_cluster", "s_umap1", "s_umap2", "S")]:
    labels = sample[col].astype(int).to_numpy()
    ax.scatter(coords[xcol], coords[ycol], c=labels, cmap="tab20", s=1.0,
               linewidths=0, alpha=0.72, rasterized=True)
    ss = summary[summary.representation == rep].set_index("cluster")
    for cluster in sorted(np.unique(labels)):
        take = labels == cluster
        cx = np.median(coords.loc[take, xcol]); cy = np.median(coords.loc[take, ycol])
        row = ss.loc[cluster]
        symbol = {"responder-enriched": "R+", "nonresponder-enriched": "NR+",
                  "mixed / not significant": "mixed",
                  "insufficient tumor cells": "TME"}[row.response_niche_call]
        ax.text(cx, cy, f"{prefix}{cluster}\n{row.top_celltype_1}\n{symbol}",
                ha="center", va="center", fontsize=7, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))
    ax.set_title(("Background z-CC" if rep == "z_cc" else "Salient s-CC") + " — annotated treated niches")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("KRASi-treated cells only: niche identity and relative tumor-response call", fontsize=15)
fig.savefig(FIG / "treated_only_annotated_zcc_scc_niche_umap.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(abundance.to_string(index=False))
print(FIG / "treated_only_annotated_zcc_scc_niche_umap.png")
