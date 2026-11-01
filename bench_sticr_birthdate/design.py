"""Task 3: early-vs-late arm assignment on MERFISH (paper's cluster labels) + anatomy-confound metrics.

Run: python -I design.py <results_dir> <geo_metadata.tsv>
Inputs (in results_dir): merfish_meta.parquet, birthdate_region_celltype_cluster.tsv
Optional: mapping_confidence.parquet (cellId, our bootstrap agreement with paper label)
"""
import sys, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

res = sys.argv[1]
rng = np.random.default_rng(0)
K = 20
MIN_MC = 20          # min multicellular clones for a cluster birthdate to be trusted
MIN_ARM = 50         # min cells per arm in a (class, region, section) unit

bd = pd.read_csv(f"{res}/birthdate_region_celltype_cluster.tsv", sep="\t", index_col=0)
CLASS = {"Excitatory neurons": "EX", "Inhibitory neurons": "IN", "Astrocytes": "Astro",
         "OPC": "OPC_Oligo", "Cajal Retzius cells": "CR", "Ependymal": "Ependymal",
         "Mitotic": "Mitotic", "UN": "UN", "Microglia": "Microglia", "Endothelial": "Vascular",
         "Vasculature": "Vascular", "Epithelial": "Epithelial"}
bd["cls"] = bd.celltype_broad.map(CLASS)
ARM_CLASSES = ["EX", "IN", "Astro", "OPC_Oligo"]   # neural lineages with a birthdate meaning
# proliferating clusters are excluded: still dividing at P4 => trivially "late-labelled" (state, not birthdate)
prolif = bd.index.str.contains("Mitotic")
ok = bd.cls.isin(ARM_CLASSES) & (bd.n_mclones >= MIN_MC) & (bd.broad_purity >= 0.8) & ~prolif
# Relative call within (dissection region, broad class): Fisher exact test of the cluster's multicellular
# clones labelled at >=E16 vs <E16 against the rest of its class in that region; BH-FDR across clusters.
# (Per-age normalisation cancels to first order because cluster and pool share the same region/age totals.)
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
geo = pd.read_csv(sys.argv[2], sep="\t", usecols=["mouse", "clone_index", "injection_date", "region",
                                                  "region_celltype_cluster"])
geo["clone"] = geo.mouse + "|" + geo.clone_index
geo = geo[geo.clone.map(geo.clone.value_counts()) >= 2]
geo["late"] = geo.injection_date.isin(["E16", "E17", "P0"])
geo["cls"] = geo.region_celltype_cluster.map(bd.cls)
mc = geo.drop_duplicates(["region_celltype_cluster", "clone"])
cnt = mc.groupby(["region_celltype_cluster", "late"]).size().unstack(fill_value=0).reindex(columns=[False, True], fill_value=0)
bd["mc_late"] = cnt[True].reindex(bd.index).fillna(0); bd["mc_early"] = cnt[False].reindex(bd.index).fillna(0)
bd["region"] = [x.split(":")[0] for x in bd.index]
bd["OR"] = np.nan; bd["p"] = np.nan
for (reg, cl), g in bd[ok].groupby(["region", "cls"]):
    # pool = all clones of this class in this region (dedup per clone)
    pool = mc[(mc.region == reg) & (mc.cls == cl)].drop_duplicates("clone")
    for c in g.index:
        inc = set(mc.clone[mc.region_celltype_cluster == c])
        rest = pool[~pool.clone.isin(inc)]
        a, b = bd.at[c, "mc_late"], bd.at[c, "mc_early"]
        cc, d = rest.late.sum(), (~rest.late).sum()
        o, pv = fisher_exact([[a, b], [cc, d]])
        bd.at[c, "OR"] = o; bd.at[c, "p"] = pv
m_ = bd.p.notna()
bd.loc[m_, "q"] = multipletests(bd.p[m_], method="fdr_bh")[1]
bd["arm"] = "mid"
bd.loc[ok & (bd.q < 0.05) & (bd.OR <= 0.8), "arm"] = "early"
bd.loc[ok & (bd.q < 0.05) & (bd.OR >= 1.25), "arm"] = "late"
bd.loc[~ok, "arm"] = "untrusted"
bd.to_csv(f"{res}/cluster_arm_calls.tsv", sep="\t")
print("cluster calls:\n", pd.crosstab(bd.cls, bd.arm))

mf = pd.read_parquet(f"{res}/merfish_meta.parquet")
# forebrain masks (paper's union) -> analysis region groups (non-overlapping CCF)
FB = ["Isocortex", "fiber tracts", "VS", "CTXsp", "OLF", "HIP", "RHP", "TH", "HY",
      "STRd", "STRv", "STR", "LSX", "PAL", "sAMY"]
mf = mf[mf.broad_region.isin(FB)].copy()
mf["region"] = mf.broad_region.replace({"STR": "STRd"})
# Cluster numbers are shared across dissected regions (Cortex:OPC_8 ~ Hippocampus:OPC_8 transcriptomically),
# so the paper label's region prefix is not reliably mappable. Use the global cluster from the label and the
# dissection region implied by the cell's CCF location; fall back to the label prefix where ambiguous.
CCF2DIS = {"Isocortex": "Cortex", "CTXsp": "Cortex", "HIP": "Hippocampus", "RHP": "Hippocampus",
           "OLF": "OB", "STRd": "Striatum", "STRv": "Striatum", "LSX": "Striatum", "PAL": "Striatum",
           "sAMY": "Striatum", "TH": "Thalamus", "HY": "Thalamus"}   # fiber tracts / VS -> label prefix
lab_reg = mf.region_celltype_cluster.str.split(":").str[0]
mf["gcluster"] = mf.region_celltype_cluster.str.split(":").str[1]
dis = mf.region.map(CCF2DIS).fillna(lab_reg)
key = dis + ":" + mf.gcluster
key = key.where(key.isin(bd.index), mf.region_celltype_cluster)
mf["bd_key"] = key
mf["cls"] = key.map(bd.cls)
mf["arm"] = key.map(bd.arm)
mf["arm_labelprefix"] = mf.region_celltype_cluster.map(bd.arm)
both = mf.arm.isin(["early", "late"]) | mf.arm_labelprefix.isin(["early", "late"])
print("arm agreement location-key vs label-prefix-key (cells with either call):",
      (mf.arm[both] == mf.arm_labelprefix[both]).mean())
conf_p = f"{res}/mapping_confidence.parquet"
if os.path.exists(conf_p):
    cf = pd.read_parquet(conf_p).set_index("cellId")
    mf = mf.join(cf, on="cellId")
print("forebrain cells", len(mf))

# arm counts per class x region (paper's labels, no confidence filter)
tab = mf[mf.arm.isin(["early", "late"])].groupby(["cls", "region", "arm"]).size().unstack(fill_value=0)
tab.to_csv(f"{res}/arm_counts_class_region.tsv", sep="\t")
pd.set_option("display.width", 220, "display.max_rows", 400)
print(tab.to_string())


def unit_metrics(xy, y, comp):
    """xy coords, y (1=late), comp: neighbour broad-class composition (all cells)."""
    n1, n0 = y.sum(), (1 - y).sum()
    p = y.mean()
    t = cKDTree(xy)
    _, nn = t.query(xy, K + 1)
    nn = nn[:, 1:]
    opp = (y[nn] != y[:, None]).mean(1)
    exp_opp = np.where(y == 1, n0 / (len(y) - 1), n1 / (len(y) - 1))
    mixing = opp.mean() / exp_opp.mean()          # 1 = random mixing, 0 = segregated
    # histogram intersection on 100 um bins
    b = np.floor(xy / 100).astype(int)
    key = b[:, 0] * 100000 + b[:, 1]
    h1 = pd.Series(key[y == 1]).value_counts(normalize=True)
    h0 = pd.Series(key[y == 0]).value_counts(normalize=True)
    overlap = np.minimum(h1, h0.reindex(h1.index).fillna(0)).sum()
    # permutation null (same counts, labels shuffled) -> overlap ratio, 1 = as overlapping as random
    yp = rng.permutation(y)
    q1 = pd.Series(key[yp == 1]).value_counts(normalize=True)
    q0 = pd.Series(key[yp == 0]).value_counts(normalize=True)
    overlap_null = np.minimum(q1, q0.reindex(q1.index).fillna(0)).sum()
    # separability: 5-fold CV kNN on coordinates, and logistic on neighbour composition
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    s_xy = np.zeros(len(y)); s_cp = np.zeros(len(y))
    for tr, te in skf.split(xy, y):
        s_xy[te] = KNeighborsClassifier(K).fit(xy[tr], y[tr]).predict_proba(xy[te])[:, 1]
        s_cp[te] = LogisticRegression(max_iter=500).fit(comp[tr], y[tr]).predict_proba(comp[te])[:, 1]
    return dict(n_early=int(n0), n_late=int(n1), mixing=mixing, bin_overlap=overlap,
                overlap_ratio=overlap / overlap_null,
                auc_xy=roc_auc_score(y, s_xy), auc_nbcomp=roc_auc_score(y, s_cp))


# neighbour broad-class composition over ALL forebrain cells in the section (Haruka's niche target)
mf["cls_all"] = mf.cls.fillna("NA")
classes = sorted(mf.cls_all.unique())
rows = []
for sec, d in mf.groupby("AP_order"):
    xy_all = d[["center_x", "center_y"]].to_numpy()
    _, nn = cKDTree(xy_all).query(xy_all, K + 1)
    oh = pd.get_dummies(d.cls_all).reindex(columns=classes, fill_value=0).to_numpy(float)
    comp_all = oh[nn[:, 1:]].mean(1)
    d = d.assign(_i=np.arange(len(d)))
    for (cls, reg), u in d[d.arm.isin(["early", "late"])].groupby(["cls", "region"]):
        y = (u.arm == "late").to_numpy(int)
        if y.sum() < MIN_ARM or (1 - y).sum() < MIN_ARM:
            continue
        mt = unit_metrics(u[["center_x", "center_y"]].to_numpy(), y, comp_all[u._i.to_numpy()])
        mt.update(section=sec, cls=cls, region=reg)
        if "boot_agree" in u:
            mt["conf_early"] = u.boot_agree[y == 0].mean(); mt["conf_late"] = u.boot_agree[y == 1].mean()
        rows.append(mt)
um = pd.DataFrame(rows)
um.to_csv(f"{res}/unit_metrics.tsv", sep="\t", index=False)
agg = um.groupby(["cls", "region"]).agg(n_sections=("section", "nunique"), n_early=("n_early", "sum"),
                                        n_late=("n_late", "sum"), mixing=("mixing", "median"),
                                        bin_overlap=("bin_overlap", "median"),
                                        overlap_ratio=("overlap_ratio", "median"), auc_xy=("auc_xy", "median"),
                                        auc_nbcomp=("auc_nbcomp", "median"))
agg.to_csv(f"{res}/unit_metrics_agg.tsv", sep="\t")
print(agg.round(3).sort_values("n_sections", ascending=False).to_string())
mf[["cellId", "AP_order", "region", "cls", "arm", "arm_labelprefix", "bd_key", "gcluster", "region_celltype_cluster"]].to_parquet(f"{res}/merfish_fb_arms.parquet")
