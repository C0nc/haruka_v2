#!/usr/bin/env python3
"""Interpret treated-only Haruka z-CC/s-CC niches and relative KRASi response.

Responder labels are never assigned to control cells.  The response score is the
existing within-(treated section, tumor state) relative decoded-effect score.
This script joins those tumor labels to the global treated niche clustering and
tests whether niche associations persist across treated sections.
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(os.environ.get(
    "HARUKA_KRAS_ANALYSIS_ROOT",
    "/home/yancui/haruka_v2/bench_kras_xenium/analysis/condition_only_salient",
))
GLOBAL = ROOT / "global" / "tables" / "cell_cluster_assignments.csv.gz"
RESP = ROOT / "responder" / "tables" / "treated_tumor_relative_responder_assignments.csv.gz"
OUT = ROOT / "responder_niche_interpretation"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1.0)
    return q


def compact_name(celltypes, max_items=3):
    return " + ".join(celltypes[:max_items])


g = pd.read_csv(GLOBAL, index_col=0)
g.index.name = "cell_id"
g = g.query("condition_label == 'KRASi_treated'").copy()
r = pd.read_csv(RESP)
assert set(r["condition_label"]) == {"KRASi_treated"}
assert not r["cell_id"].duplicated().any()

extreme = r[r["response_class"].isin(["relative_responder", "relative_nonresponder"])].copy()
extreme["is_responder"] = (extreme["response_class"] == "relative_responder").astype(int)

all_summaries = []
all_compositions = []
all_by_slice = []

for rep, col in [("z_cc", "z_cc_cluster"), ("s_cc", "s_cc_cluster")]:
    treated = g[g[col] >= 0].copy()
    treated[col] = treated[col].astype(int)

    counts = pd.crosstab(treated[col], treated["celltype"])
    props = counts.div(counts.sum(axis=1), axis=0)
    for cluster in props.index:
        row = props.loc[cluster].sort_values(ascending=False)
        for ct, frac in row.items():
            all_compositions.append({
                "representation": rep, "cluster": int(cluster),
                "celltype": ct, "fraction": float(frac),
                "n_cells": int(counts.loc[cluster, ct]),
            })

    joined = extreme[["cell_id", "slice_id", "celltype", "is_responder", "response_percentile"]].merge(
        treated[[col]], left_on="cell_id", right_index=True, how="inner", validate="one_to_one"
    )
    joined = joined.rename(columns={col: "cluster"})

    for cluster in sorted(treated[col].unique()):
        cluster = int(cluster)
        top = props.loc[cluster].sort_values(ascending=False)
        top3 = top.index[:3].tolist()
        sub = joined[joined["cluster"] == cluster]
        inside_r = int(sub["is_responder"].sum())
        inside_n = int(len(sub) - inside_r)
        outside = joined[joined["cluster"] != cluster]
        outside_r = int(outside["is_responder"].sum())
        outside_n = int(len(outside) - outside_r)
        odds, p = fisher_exact([[inside_r, inside_n], [outside_r, outside_n]])

        slice_fracs = []
        for sid, ss in sub.groupby("slice_id"):
            nr = int(ss["is_responder"].sum())
            nn = int(len(ss) - nr)
            frac = nr / len(ss) if len(ss) else np.nan
            slice_fracs.append(frac)
            all_by_slice.append({
                "representation": rep, "cluster": cluster, "slice_id": sid,
                "n_extreme_tumor_cells": int(len(ss)),
                "responder_fraction": frac,
                "responder_n": nr, "nonresponder_n": nn,
            })

        overall = inside_r / max(inside_r + inside_n, 1)
        all_summaries.append({
            "representation": rep,
            "cluster": cluster,
            "n_all_treated_cells": int((treated[col] == cluster).sum()),
            "n_extreme_tumor_cells": int(len(sub)),
            "responder_n": inside_r,
            "nonresponder_n": inside_n,
            "responder_fraction_extremes": overall,
            "responder_fold_vs_0.5": overall / 0.5,
            "fisher_odds_ratio": odds,
            "p_value": p,
            "slice_fraction_min": np.nanmin(slice_fracs) if slice_fracs else np.nan,
            "slice_fraction_max": np.nanmax(slice_fracs) if slice_fracs else np.nan,
            "slice_fraction_sd": np.nanstd(slice_fracs) if slice_fracs else np.nan,
            "n_slices_with_extreme_cells": len(slice_fracs),
            "top_celltype_1": top3[0], "top_celltype_1_fraction": top.iloc[0],
            "top_celltype_2": top3[1], "top_celltype_2_fraction": top.iloc[1],
            "top_celltype_3": top3[2], "top_celltype_3_fraction": top.iloc[2],
            "niche_annotation": compact_name(top3),
        })

summary = pd.DataFrame(all_summaries)
summary["fdr_bh"] = np.nan
for rep, idx in summary.groupby("representation").groups.items():
    summary.loc[idx, "fdr_bh"] = bh_fdr(summary.loc[idx, "p_value"])

def response_call(row):
    if row["n_extreme_tumor_cells"] < 30:
        return "insufficient tumor cells"
    if row["fdr_bh"] < 0.05 and row["fisher_odds_ratio"] > 1:
        return "responder-enriched"
    if row["fdr_bh"] < 0.05 and row["fisher_odds_ratio"] < 1:
        return "nonresponder-enriched"
    return "mixed / not significant"

summary["response_niche_call"] = summary.apply(response_call, axis=1)
summary = summary.sort_values(["representation", "cluster"])
composition = pd.DataFrame(all_compositions)
by_slice = pd.DataFrame(all_by_slice)

summary.to_csv(TAB / "zcc_scc_niche_annotations_and_response.csv", index=False)
composition.to_csv(TAB / "zcc_scc_cluster_celltype_composition.csv", index=False)
by_slice.to_csv(TAB / "zcc_scc_responder_fraction_by_slice.csv", index=False)

# A compact human-readable interpretation table.
report_cols = [
    "representation", "cluster", "niche_annotation", "n_all_treated_cells",
    "n_extreme_tumor_cells", "responder_fraction_extremes", "fisher_odds_ratio",
    "fdr_bh", "slice_fraction_min", "slice_fraction_max", "response_niche_call",
]
summary[report_cols].to_csv(TAB / "niche_response_interpretation_compact.csv", index=False)

sns.set_theme(style="whitegrid", font_scale=0.9)
fig, axes = plt.subplots(2, 2, figsize=(19, 12), constrained_layout=True)

for ax, rep, title in [(axes[0, 0], "z_cc", "Background z-CC niche composition"),
                       (axes[0, 1], "s_cc", "Salient s-CC niche composition")]:
    comp = composition[composition.representation == rep]
    # Show the globally most abundant 12 cell types; retain remainder as Other.
    totals = comp.groupby("celltype")["n_cells"].sum().nlargest(12).index
    plot = comp.copy()
    plot["display_celltype"] = np.where(plot.celltype.isin(totals), plot.celltype, "Other")
    plot = plot.groupby(["cluster", "display_celltype"], as_index=False)["fraction"].sum()
    piv = plot.pivot(index="cluster", columns="display_celltype", values="fraction").fillna(0)
    # Put Other last.
    cols = [c for c in piv.columns if c != "Other"] + (["Other"] if "Other" in piv else [])
    piv[cols].plot(kind="bar", stacked=True, ax=ax, width=0.85, colormap="tab20", legend=False)
    ax.set(title=title, xlabel=f"{rep} cluster", ylabel="Cell-type proportion", ylim=(0, 1))

# Shared composition legend outside top-right panel.
handles, labels = axes[0, 1].get_legend_handles_labels()
axes[0, 1].legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title="Cell type")

for ax, rep, title in [(axes[1, 0], "z_cc", "z-CC: relative responder enrichment"),
                       (axes[1, 1], "s_cc", "s-CC: relative responder enrichment")]:
    ss = summary[summary.representation == rep].copy()
    colors = ss.response_niche_call.map({
        "responder-enriched": "#d73027",
        "nonresponder-enriched": "#4575b4",
        "mixed / not significant": "#999999",
        "insufficient tumor cells": "#dddddd",
    })
    ax.bar(ss.cluster.astype(str), ss.responder_fraction_extremes, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="balanced extreme-class null")
    for x, (_, row) in enumerate(ss.iterrows()):
        label = f"{row.top_celltype_1}\n(n={int(row.n_extreme_tumor_cells)})"
        ax.text(x, min(0.98, row.responder_fraction_extremes + 0.035), label,
                ha="center", va="bottom", rotation=90, fontsize=6.5)
    ax.set(title=title, xlabel=f"{rep} cluster", ylabel="Responder fraction among responder/nonresponder extremes", ylim=(0, 1.2))

fig.suptitle("KRASi-treated cells only: TME niche composition and relative tumor-response enrichment", fontsize=16)
fig.savefig(FIG / "treated_only_zcc_scc_niche_composition_response.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Per-section response heatmaps expose reproducibility versus replicate dependence.
fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
for ax, rep, title in [(axes[0], "z_cc", "z-CC niches"), (axes[1], "s_cc", "s-CC niches")]:
    p = by_slice[by_slice.representation == rep].pivot(index="cluster", columns="slice_id", values="responder_fraction")
    sns.heatmap(p, vmin=0, vmax=1, center=0.5, cmap="RdBu_r", annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "Responder fraction among extremes"})
    ax.set_title(title + ": section-stratified relative response")
    ax.set_xlabel("Treated section (JB_M*)")
    ax.set_ylabel("Niche cluster")
fig.suptitle("Replicate dependence of niche–response associations (treated tumor cells only)", fontsize=15)
fig.savefig(FIG / "treated_only_niche_response_by_replicate.png", dpi=220, bbox_inches="tight")
plt.close(fig)

meta = {
    "candidate_population": "KRASi_treated tumor cells only",
    "controls_assigned_responder_status": False,
    "niche_population": "all KRASi-treated cells",
    "response_test_population": "relative responders and nonresponders among treated tumor cells",
    "niche_definition": "K-means clusters of three-hop slice-bounded z-CC or corrected condition-only s-CC",
    "celltype_composition": "observed annotated cell-type fraction among all treated cells in each niche",
    "response_call": "two-sided Fisher exact test versus other niches, BH-FDR within representation",
    "replicate_interpretation": "section-specific variation is response heterogeneity only if association is directionally reproducible; otherwise it is replicate/anatomy dependent and cannot be separated from technical effects",
    "important_caveat": "response is an internal Haruka decoded-effect magnitude, not measured tumor shrinkage; cell-type composition was part of niche reconstruction and is not independent validation",
}
(OUT / "analysis_metadata.json").write_text(json.dumps(meta, indent=2))

print(summary[report_cols].to_string(index=False))
print(f"\nOutputs: {OUT}")
