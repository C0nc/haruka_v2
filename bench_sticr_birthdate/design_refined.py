"""Refined candidate designs + anatomy-confound metrics, with and without mapping-consensus filter.
consensus = paper label and our MapMyCells label give the same arm, and the cell passes the 3xMADlow filter.
Run: python -I design_refined.py <results_dir>
"""
import sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

res = sys.argv[1]
K, MIN_ARM = 20, 50
rng = np.random.default_rng(0)
meta = pd.read_parquet(f"{res}/merfish_meta.parquet", columns=["cellId", "center_x", "center_y", "AP_order"])
fb = pd.read_parquet(f"{res}/merfish_fb_arms.parquet")
cf = pd.read_parquet(f"{res}/mapping_confidence.parquet")
bdg = pd.read_csv(f"{res}/cluster_arm_calls.tsv", sep="\t", index_col=0)
m = fb.merge(meta, on=["cellId", "AP_order"]).merge(cf, on="cellId", how="left")
m["cls_all"] = m.cls.fillna("NA")
classes = sorted(m.cls_all.unique())
DIS = {"Isocortex": "Cortex", "fiber tracts": "Cortex", "CTXsp": "Cortex", "VS": "Cortex",
       "HIP": "Hippocampus", "RHP": "Hippocampus", "HY": "Thalamus"}

DESIGNS = {
    # paper: OPC prenatal clusters 4/8(/39) vs postnatal-enriched 5/16; strict = 4,8 vs 16 (drop borderline 5, oligos)
    "OPC_strict": dict(cls="OPC_Oligo", early={"OPC_4", "OPC_8"}, late={"OPC_16"}, regions=["Isocortex", "fiber tracts"]),
    # recommended: OPCs only (no oligodendrocytes = no differentiation-state contrast); paper's prenatal 4/8 vs postnatal 5/16
    "OPC_4_8_vs_5_16": dict(cls="OPC_Oligo", early={"OPC_4", "OPC_8"}, late={"OPC_16", "OPC_5"}, regions=["Isocortex", "fiber tracts"]),
    "OPC_fisher": dict(cls="OPC_Oligo", early=None, late=None, regions=["Isocortex", "fiber tracts"]),
    "Astro_fisher_HPF": dict(cls="Astro", early=None, late=None, regions=["HIP", "RHP"]),
    "Astro_fisher_HY": dict(cls="Astro", early=None, late=None, regions=["HY"]),
}


def arm_of(df, spec, cluster_col):
    if spec["early"] is not None:
        return np.where(df[cluster_col].isin(spec["early"]), "early",
                        np.where(df[cluster_col].isin(spec["late"]), "late", "none"))
    key = df.region.map(DIS).fillna("Cortex") + ":" + df[cluster_col]
    a = key.map(bdg.arm).where(key.map(bdg.cls) == spec["cls"])
    a = a.where(a.isin(["early", "late"]))   # mid / untrusted are not arms
    return a.fillna("none").to_numpy()


def metrics(xy, y, comp):
    n1, n0 = y.sum(), (1 - y).sum()
    _, nn = cKDTree(xy).query(xy, K + 1); nn = nn[:, 1:]
    opp = (y[nn] != y[:, None]).mean(1)
    exp_opp = np.where(y == 1, n0 / (len(y) - 1), n1 / (len(y) - 1))
    b = np.floor(xy / 100).astype(int); key = b[:, 0] * 100000 + b[:, 1]
    def ov(yy):
        h1 = pd.Series(key[yy == 1]).value_counts(normalize=True); h0 = pd.Series(key[yy == 0]).value_counts(normalize=True)
        return np.minimum(h1, h0.reindex(h1.index).fillna(0)).sum()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    s_xy = np.zeros(len(y)); s_cp = np.zeros(len(y))
    for tr, te in skf.split(xy, y):
        s_xy[te] = KNeighborsClassifier(K).fit(xy[tr], y[tr]).predict_proba(xy[te])[:, 1]
        s_cp[te] = LogisticRegression(max_iter=500).fit(comp[tr], y[tr]).predict_proba(comp[te])[:, 1]
    return dict(n_early=int(n0), n_late=int(n1), mixing=opp.mean() / exp_opp.mean(),
                overlap_ratio=ov(y) / ov(rng.permutation(y)),
                auc_xy=roc_auc_score(y, s_xy), auc_nbcomp=roc_auc_score(y, s_cp))


rows, counts = [], []
for sec, d in m.groupby("AP_order"):
    xy_all = d[["center_x", "center_y"]].to_numpy()
    _, nn = cKDTree(xy_all).query(xy_all, K + 1)
    oh = pd.get_dummies(d.cls_all).reindex(columns=classes, fill_value=0).to_numpy(float)
    comp_all = oh[nn[:, 1:]].mean(1)
    d = d.assign(_i=np.arange(len(d)))
    for name, spec in DESIGNS.items():
        u = d[(d.cls == spec["cls"]) & d.region.isin(spec["regions"])]
        if len(u) == 0:
            continue
        a_p = arm_of(u, spec, "gcluster")
        a_m = np.where(u.ctm_cluster.notna(), arm_of(u.assign(ctm_cluster=u.ctm_cluster.fillna("NA")), spec, "ctm_cluster"), "NA")
        cons = (a_p == a_m) & u.ctm_mad_pass.astype("boolean").fillna(False).to_numpy(bool)
        for filt, mask in [("paper_label", a_p != "none"), ("consensus", (a_p != "none") & cons)]:
            uu = u[mask]; yy = (a_p[mask] == "late").astype(int)
            for reg in spec["regions"] + ["all"]:
                sel = np.ones(len(uu), bool) if reg == "all" else (uu.region == reg).to_numpy()
                y = yy[sel]
                counts.append(dict(design=name, filt=filt, region=reg, section=sec, n_early=int((1 - y).sum()),
                                   n_late=int(y.sum())))
                if y.sum() < MIN_ARM or (1 - y).sum() < MIN_ARM:
                    continue
                mt = metrics(uu[["center_x", "center_y"]].to_numpy()[sel], y, comp_all[uu._i.to_numpy()[sel]])
                mt.update(design=name, filt=filt, region=reg, section=sec)
                rows.append(mt)
um = pd.DataFrame(rows); um.to_csv(f"{res}/refined_unit_metrics.tsv", sep="\t", index=False)
ct = pd.DataFrame(counts); ct.to_csv(f"{res}/refined_counts.tsv", sep="\t", index=False)
agg = um.groupby(["design", "filt", "region"]).agg(
    n_sections=("section", "nunique"), n_early=("n_early", "sum"), n_late=("n_late", "sum"),
    mixing=("mixing", "median"), overlap_ratio=("overlap_ratio", "median"),
    auc_xy=("auc_xy", "median"), auc_xy_q90=("auc_xy", lambda x: x.quantile(0.9)), auc_nbcomp=("auc_nbcomp", "median"))
tot = ct.groupby(["design", "filt", "region"])[["n_early", "n_late"]].sum().add_prefix("all_sections_")
agg = agg.join(tot)
agg.to_csv(f"{res}/refined_metrics_agg.tsv", sep="\t")
pd.set_option("display.width", 250)
print(agg.round(3).to_string())
