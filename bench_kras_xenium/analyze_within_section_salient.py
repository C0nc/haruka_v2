#!/usr/bin/env python3
"""Find what Haruka s-CC captures within KRASi-treated sections.

The corrected condition-only s-CC is standardized separately inside JB_M0--M3,
then a shared PCA/K-means model is fit on a balanced sample.  This removes
section-specific first and second moments before clustering and focuses the
readout on intra-section salient heterogeneity.
"""
from pathlib import Path
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import umap


ANALYSIS_ROOT = Path(os.environ.get("HARUKA_KRAS_ANALYSIS_ROOT", "/home/yancui/haruka_v2/bench_kras_xenium/analysis/condition_only_salient"))
DATA = Path(os.environ.get("HARUKA_KRAS_DATA", "/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad"))
CC = Path(os.environ.get("HARUKA_KRAS_CC", "/home/yancui/haruka_v2/bench_kras_xenium/results_cellcharter/haruka_z_s_cc3_condition_only"))
MODEL = Path(os.environ.get("HARUKA_KRAS_MODEL", "/home/yancui/haruka_v2/bench_kras_xenium/results_new_haruka/mouselung_kras_xenium_clean_noC3/haruka_shared_full_niche_seed0"))
RESP = ANALYSIS_ROOT / "responder" / "tables" / "treated_tumor_relative_responder_assignments.csv.gz"
OUT = ANALYSIS_ROOT / "within_section_salient"
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

PROGRAMS = {
    "MAPK_AP1": ["Dusp1", "Dusp4", "Egr1", "Fosl1", "Fosl2", "Spry1", "Spry2", "Etv4", "Etv5"],
    "EMT": ["Vim", "Fn1", "Sparc", "Spp1", "Tagln", "Acta2", "Tgfbr1", "Tgfbr2"],
    "epithelial_Itgb6": ["Itgb6", "Epcam", "Krt2", "Krt20", "Sox9", "Muc1", "Klf5", "Yap1", "Tead1"],
    "cycling": ["Mki67", "Top2a", "Cdk1", "Ccnb1", "Pcna", "Mcm2"],
    "interferon": ["Stat1", "Irf1", "Isg15", "Ifit1", "Ifit3", "Cxcl10"],
    "antigen_presentation": ["Tap1", "Ciita", "Psmb9", "H2-DMb2"],
    "hypoxia_stress": ["Hif1a", "Vegfa", "Slc2a1", "Ldha", "Hmox1", "Ndrg1"],
    "myeloid_suppression": ["Arg1", "Trem2", "Mrc1", "Lgals9", "Ccl2", "Ccl7", "Cxcl2", "Tnf"],
}


def bh(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    q0 = p[order] * m / np.arange(1, m + 1)
    q0 = np.minimum.accumulate(q0[::-1])[::-1]
    q = np.empty(m); q[order] = np.minimum(q0, 1)
    return q


def program_scores(data, full_rows):
    genes = pd.Index(data.var_names.astype(str))
    present = {k: [g for g in v if g in genes] for k, v in PROGRAMS.items()}
    chosen = sorted(set(sum(present.values(), [])))
    pos = genes.get_indexer(chosen); gene_col = {g: i for i, g in enumerate(chosen)}
    block = data[full_rows].to_memory()
    x = block.layers["counts"] if "counts" in block.layers else block.X
    if sparse.issparse(x):
        x = x.tocsr().astype(np.float32)
        lib = np.asarray(x.sum(1)).ravel()
        x = x[:, pos].multiply((1e4 / np.maximum(lib, 1))[:, None]).tocsr()
        x.data = np.log1p(x.data); x = x.toarray()
    else:
        x = np.asarray(x, np.float32); lib = x.sum(1)
        x = np.log1p(x[:, pos] * (1e4 / np.maximum(lib, 1))[:, None])
    out = {}
    for name, members in present.items():
        cols = [gene_col[g] for g in members]
        out[name] = x[:, cols].mean(1) if cols else np.zeros(len(full_rows))
    return pd.DataFrame(out), present


rng = np.random.default_rng(7)
data = ad.read_h5ad(DATA, backed="r")
full_obs = data.obs.copy()
source_rows = np.load(CC / "source_rows.npy").astype(int)
names = np.load(CC / "obs_names.npy", allow_pickle=True).astype(str)
obs = full_obs.iloc[source_rows].copy()
assert np.array_equal(names, obs.index.astype(str).to_numpy())
assert set(obs["condition_label"].astype(str)) == {"KRASi_treated"}
scc = np.load(CC / "salient_spatial.npy", mmap_mode="r")
slices = obs["slice_id"].astype(str).to_numpy()

# Estimate section-specific means/SDs on balanced 12k-cell samples.
fit_parts, plot_parts, means, scales = [], [], {}, {}
for sid in sorted(np.unique(slices)):
    rows = np.flatnonzero(slices == sid)
    fit = rng.choice(rows, min(12000, len(rows)), replace=False)
    plot = rng.choice(rows, min(3500, len(rows)), replace=False)
    x = np.asarray(scc[fit], dtype=np.float32)
    means[sid] = x.mean(0)
    sd = x.std(0); sd[sd < 1e-5] = 1.0; scales[sid] = sd
    fit_parts.append(np.sort(fit)); plot_parts.append(np.sort(plot))
fit_rows = np.concatenate(fit_parts); plot_rows = np.concatenate(plot_parts)

def residual_block(rows):
    out = np.empty((len(rows), scc.shape[1]), dtype=np.float32)
    row_slices = slices[rows]
    for sid in np.unique(row_slices):
        take = row_slices == sid
        out[take] = (np.asarray(scc[rows[take]], np.float32) - means[sid]) / scales[sid]
    return np.nan_to_num(out)

xfit = residual_block(fit_rows)
pca = PCA(n_components=30, random_state=7).fit(xfit)
pfit = pca.transform(xfit)
kmeans = MiniBatchKMeans(n_clusters=10, random_state=7, n_init=20, batch_size=8192).fit(pfit)
labels = np.full(len(obs), -1, np.int16)
for left in range(0, len(obs), 25000):
    rows = np.arange(left, min(left + 25000, len(obs)))
    labels[rows] = kmeans.predict(pca.transform(residual_block(rows)))

# Verify section mixing on the balanced fit set.
fit_labels = labels[fit_rows]
section_ari = adjusted_rand_score(slices[fit_rows], fit_labels)
section_nmi = normalized_mutual_info_score(slices[fit_rows], fit_labels)

# Observed cell-type composition and enrichment of residual salient niches.
celltypes = obs["celltype"].astype(str).to_numpy()
ctab = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(celltypes, name="celltype"))
composition = ctab.div(ctab.sum(1), axis=0)
expected = ctab.sum(0) / ctab.to_numpy().sum()
enrichment = np.log2((composition + 1e-4).div(expected + 1e-4, axis=1))
composition.to_csv(TAB / "within_section_scc_celltype_composition.csv")
enrichment.to_csv(TAB / "within_section_scc_celltype_log2_enrichment.csv")

# Response association uses treated tumor cells only; controls do not exist here.
resp = pd.read_csv(RESP)
lookup = pd.Series(labels, index=names, name="within_section_scc_cluster")
resp = resp.join(lookup, on="cell_id", how="left", validate="many_to_one")
extreme = resp[resp.response_class.isin(["relative_responder", "relative_nonresponder"])].copy()
extreme["is_responder"] = (extreme.response_class == "relative_responder").astype(int)
slice_response = (extreme.groupby(["within_section_scc_cluster", "slice_id"], observed=True)
                  .is_responder.agg(["size", "sum", "mean"]).reset_index()
                  .rename(columns={"size": "n_extreme", "sum": "responder_n", "mean": "responder_fraction"}))
slice_response.to_csv(TAB / "within_section_scc_response_by_section.csv", index=False)

summaries = []
for c in range(10):
    inside = extreme[extreme.within_section_scc_cluster == c]
    outside = extreme[extreme.within_section_scc_cluster != c]
    a, b = int(inside.is_responder.sum()), int(len(inside) - inside.is_responder.sum())
    cc, d = int(outside.is_responder.sum()), int(len(outside) - outside.is_responder.sum())
    odds, pval = fisher_exact([[a, b], [cc, d]])
    top = composition.loc[c].sort_values(ascending=False)
    sf = slice_response[slice_response.within_section_scc_cluster == c].responder_fraction
    summaries.append({"cluster": c, "n_all_treated": int((labels == c).sum()),
                      "n_extreme_tumor": len(inside), "responder_fraction": a / max(a + b, 1),
                      "odds_ratio": odds, "p_value": pval,
                      "slice_min": sf.min(), "slice_max": sf.max(), "slice_sd": sf.std(),
                      "top_celltype_1": top.index[0], "top_celltype_1_fraction": top.iloc[0],
                      "top_celltype_2": top.index[1], "top_celltype_2_fraction": top.iloc[1],
                      "top_celltype_3": top.index[2], "top_celltype_3_fraction": top.iloc[2]})
summary = pd.DataFrame(summaries); summary["fdr_bh"] = bh(summary.p_value)
summary["response_call"] = np.where((summary.fdr_bh < .05) & (summary.odds_ratio > 1), "responder-enriched",
                              np.where((summary.fdr_bh < .05) & (summary.odds_ratio < 1), "nonresponder-enriched", "mixed"))

# Molecular programs, residualized within (section, cell type), on balanced sample.
program, present = program_scores(data, source_rows[fit_rows])
program["slice_id"] = slices[fit_rows]; program["celltype"] = celltypes[fit_rows]
program["cluster"] = fit_labels
for col in PROGRAMS:
    group = program.groupby(["slice_id", "celltype"], observed=True)[col]
    mu = group.transform("mean"); sd = group.transform("std").replace(0, np.nan)
    program[col + "_resid_z"] = ((program[col] - mu) / sd).fillna(0).clip(-5, 5)
prog_cols = [x + "_resid_z" for x in PROGRAMS]
prog_mean = program.groupby("cluster", observed=True)[prog_cols].mean()
prog_mean.columns = [x.replace("_resid_z", "") for x in prog_mean.columns]
prog_mean.to_csv(TAB / "within_section_scc_residual_program_means.csv")

# Decoded composition changes on the same balanced sample.
shift = np.load(MODEL / "salient_composition_shift.npy", mmap_mode="r")
cats = json.loads((MODEL / "meta.json").read_text())["cell_type_categories"]
shift_sample = np.asarray(shift[source_rows[fit_rows]], np.float32)
shift_mean = pd.DataFrame(shift_sample, columns=cats).assign(cluster=fit_labels).groupby("cluster").mean()
shift_mean.to_csv(TAB / "within_section_scc_decoded_composition_shift.csv")

for i, row in summary.iterrows():
    pm = prog_mean.loc[row.cluster].sort_values()
    sm = shift_mean.loc[row.cluster].sort_values()
    summary.loc[i, "top_program_high"] = pm.index[-1]
    summary.loc[i, "top_program_low"] = pm.index[0]
    summary.loc[i, "top_composition_increase"] = sm.index[-1]
    summary.loc[i, "top_composition_decrease"] = sm.index[0]
summary.to_csv(TAB / "within_section_scc_cluster_interpretation.csv", index=False)

# Figure: section-residual UMAP, annotation, response, and biological programs.
xplot = pca.transform(residual_block(plot_rows))
xy = umap.UMAP(n_neighbors=30, min_dist=.25, random_state=7).fit_transform(xplot)
plot_labels = labels[plot_rows]
fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
axes[0, 0].scatter(xy[:, 0], xy[:, 1], c=plot_labels, cmap="tab10", s=1, linewidths=0, rasterized=True)
for c in range(10):
    take = plot_labels == c
    axes[0, 0].text(np.median(xy[take, 0]), np.median(xy[take, 1]), f"WS{c}", weight="bold",
                    bbox=dict(fc="white", ec="black", alpha=.8), ha="center")
axes[0, 0].set_title("Within-section residual s-CC clusters")
for sid, color in zip(sorted(np.unique(slices)), sns.color_palette("Set2", 4)):
    take = slices[plot_rows] == sid
    axes[0, 1].scatter(xy[take, 0], xy[take, 1], s=1, color=color, label=sid, linewidths=0, rasterized=True)
axes[0, 1].set_title(f"Section mixing after residualization: ARI={section_ari:.3f}, NMI={section_nmi:.3f}")
axes[0, 1].legend(markerscale=5, frameon=False)

top_ct = ctab.sum(0).nlargest(16).index
sns.heatmap(enrichment[top_ct], cmap="vlag", center=0, ax=axes[1, 0],
            cbar_kws={"label": "log2 observed/expected"})
axes[1, 0].set_title("Cell-type enrichment of within-section s-CC niches")
sns.heatmap(prog_mean, cmap="vlag", center=0, annot=True, fmt=".2f", ax=axes[1, 1],
            cbar_kws={"label": "mean within-section/cell-type residual z"})
axes[1, 1].set_title("Molecular programs beyond section and cell type")
for ax in axes[0]: ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("What salient-CC captures inside KRASi-treated sections", fontsize=17)
fig.savefig(FIG / "within_section_scc_identity_and_programs.png", dpi=220, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
rp = slice_response.pivot(index="within_section_scc_cluster", columns="slice_id", values="responder_fraction")
sns.heatmap(rp, cmap="RdBu_r", center=.5, vmin=0, vmax=1, annot=True, fmt=".2f", ax=axes[0],
            cbar_kws={"label": "relative responder fraction"})
axes[0].set_title("Tumor response association within each treated section")
top_shift = shift_mean.abs().max(0).nlargest(18).index
sns.heatmap(shift_mean[top_shift], cmap="vlag", center=0, ax=axes[1],
            cbar_kws={"label": "decoded D(z,s) - D(z,0)"})
axes[1].set_title("Decoded treatment-associated niche-composition remodeling")
fig.savefig(FIG / "within_section_scc_response_and_decoded_shift.png", dpi=220, bbox_inches="tight")
plt.close(fig)

pd.DataFrame({"cell_id": names, "slice_id": slices, "celltype": celltypes,
              "within_section_scc_cluster": labels}).to_csv(
    TAB / "within_section_scc_assignments.csv.gz", index=False, compression="gzip")

meta = {"population": "KRASi-treated cells only", "control_responder_labels": 0,
        "method": "per-section feature centering/scaling, shared PCA30 + KMeans10",
        "purpose": "remove section first/second moments and characterize intra-section salient heterogeneity",
        "section_ari": section_ari, "section_nmi": section_nmi,
        "response_endpoint": "internal relative decoded-effect score, not measured shrinkage",
        "caveat": "unpaired sections; residualization does not prove causal treatment response",
        "present_program_genes": present}
(OUT / "analysis_metadata.json").write_text(json.dumps(meta, indent=2))
print(summary.to_string(index=False))
print(json.dumps(meta, indent=2))
print(f"Outputs: {OUT}")
