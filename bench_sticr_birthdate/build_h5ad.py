"""Build (do NOT fit) the Haruka h5ad for one early-vs-late design.
Run: python -I build_h5ad.py <bench_dir> <design_name> <class> <regions comma-sep> <early clusters> <late clusters>
condition: control = early-born subtypes, disease = late-born subtypes (same broad class, same CCF regions),
           halo = every other extracted cell (neighbours only; runner flag --exclude_label halo).
Sections kept = sections with >= MIN_ARM cells in both arms.
"""
import sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

B, name, cls, regions = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
EARLY, LATE = set(sys.argv[5].split(",")), set(sys.argv[6].split(","))
min_conf = 0.0
MIN_ARM = 50
meta = pd.read_parquet(f"{B}/results/merfish_meta.parquet")
fb = pd.read_parquet(f"{B}/results/merfish_fb_arms.parquet")
meta = meta.merge(fb[["cellId", "cls", "arm", "region", "bd_key", "gcluster"]], on="cellId", how="left")
X = sp.load_npz(f"{B}/work/merfish_subset.npz")
cells = np.load(f"{B}/work/merfish_subset.npz.cells.npy")
genes = pd.read_csv(f"{B}/work/merfish_subset.npz.genes.tsv", sep="\t", header=None)[0].to_numpy()
gk = ~pd.Series(genes).str.startswith("Blank").to_numpy()
meta = meta.set_index("cellId").loc[cells]
# broad cell-type key for every cell (paper label -> global cluster -> majority scRNA broad class)
bdg = pd.read_csv(f"{B}/results/cluster_arm_calls.tsv", sep="\t", index_col=0)
lab2cls = bdg.cls
meta["broad"] = meta.region_celltype_cluster.map(lab2cls).fillna("other")
conf = None
try:
    conf = pd.read_parquet(f"{B}/results/mapping_confidence.parquet").set_index("cellId")
    meta = meta.join(conf, how="left")
except FileNotFoundError:
    pass
meta["arm"] = np.where(meta.gcluster.isin(EARLY), "early", np.where(meta.gcluster.isin(LATE), "late", None))
inarm = (meta.cls == cls) & meta.region.isin(regions) & meta.arm.isin(["early", "late"])
# mapping consensus flag (paper label and our MapMyCells label in the same arm, passes 3xMADlow)
if conf is not None:
    ctm_arm = np.where(meta.ctm_cluster.isin(EARLY), "early", np.where(meta.ctm_cluster.isin(LATE), "late", "other"))
    meta["consensus"] = (ctm_arm == meta.arm) & meta.ctm_mad_pass.astype("boolean").fillna(False).to_numpy(bool)
wmm = f"{B}/results/opc_isocortex_wm_matched_cellids.npy"
import os
meta["wm_matched"] = meta.index.isin(np.load(wmm)) if os.path.exists(wmm) else False
cnt = meta[inarm].groupby(["AP_order", "arm"]).size().unstack(fill_value=0)
good = cnt.index[(cnt.get("early", 0) >= MIN_ARM) & (cnt.get("late", 0) >= MIN_ARM)]
keep = meta.AP_order.isin(good).to_numpy()
m = meta[keep].copy()
m["condition"] = "halo"
a = inarm[keep]
m.loc[a & (m.arm == "early"), "condition"] = "control"
m.loc[a & (m.arm == "late"), "condition"] = "disease"
m["section"] = "sec" + m.AP_order.astype(int).astype(str).str.zfill(2)
obs = pd.DataFrame({"condition": m.condition.values, "cell_type": m.broad.values, "section": m.section.values,
                    "ccf_region": m.broad_region.values, "paper_label": m.region_celltype_cluster.values,
                    "gcluster": m.gcluster.values, "arm_call": m.arm.fillna("NA").values,
                    "wm_matched": m.wm_matched.values & (m.condition != "halo").values},
                   index=m.index.astype(str))
if conf is not None:
    for c in ["ctm_cluster", "ctm_boot_prob", "ctm_corr", "ctm_agree_cluster", "ctm_mad_pass", "consensus"]:
        if c in m:
            v = m[c]
            obs[c] = v.astype(str).values if v.dtype == object or c in ("ctm_agree_cluster", "ctm_mad_pass") else v.values
Xk = X[keep][:, gk].astype(np.float32)
A = ad.AnnData(X=Xk.copy(), obs=obs, var=pd.DataFrame(index=genes[gk]))
A.layers["counts"] = Xk
A.obsm["spatial"] = m[["center_x", "center_y"]].to_numpy(np.float32)
for c in ["condition", "cell_type", "section", "ccf_region"]:
    A.obs[c] = A.obs[c].astype("category")
A.uns["design"] = dict(name=name, broad_class=cls, regions=regions, early=sorted(EARLY), late=sorted(LATE),
                       note="control=early-born subtypes, disease=late-born subtypes; halo=neighbours only")
out = f"{B}/sticr_{name}.h5ad"
A.write_h5ad(out, compression="gzip")
print(out, A.shape)
print(pd.crosstab(A.obs.section, A.obs.condition).to_string())
print(pd.crosstab(A.obs.cell_type, A.obs.condition).to_string())
