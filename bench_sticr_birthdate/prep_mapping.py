"""Build reference + query h5ads for MapMyCells (cell_type_mapper) correlation mapping.
reference: scRNA cells x 500 MERFISH genes, obs: cls (majority broad) / cluster (region_celltype_cluster)
query: MERFISH forebrain cells (subset), 500 genes, raw counts
Run: python -I prep_mapping.py <bench_dir> <n_query_random>
"""
import sys, json
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

B = sys.argv[1]; NQ = int(sys.argv[2])
rng = np.random.default_rng(0)
# ---------- reference
X = sp.load_npz(f"{B}/work/rna_counts_500.npz")
g = pd.read_csv(f"{B}/work/rna_counts_500.npz.genes.tsv", sep="\t", header=None)[0].to_numpy()
bc = pd.read_csv(f"{B}/raw/ucsc_rna/counts_barcodes.tsv.gz", sep="\t", header=None)[0].to_numpy()
meta = pd.read_csv(f"{B}/raw/geo/geo_metadata.tsv", sep="\t").set_index("cell")
meta = meta.reindex(bc)
assert meta.celltype_cluster.notna().all()
# global transcriptomic clusters (region prefix is a dissection tag, not separable by expression)
maj = meta.groupby("celltype_cluster").celltype_broad.agg(lambda x: x.value_counts().index[0])
obs = pd.DataFrame({"cls": meta.celltype_cluster.map(maj).str.replace(" ", "_").values,
                    "cluster": meta.celltype_cluster.values}, index=bc)
ref = ad.AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=g))
ref.write_h5ad(f"{B}/work/ref_500.h5ad")
# ---------- query
mf = pd.read_parquet(f"{B}/results/merfish_fb_arms.parquet")
cand = ((mf.cls == "OPC_Oligo") & mf.region.isin(["Isocortex", "fiber tracts"])) | \
       ((mf.cls == "Astro") & mf.region.isin(["HIP", "RHP", "HY"]))
arm = mf[mf.arm.isin(["early", "late"]) & cand].cellId.to_numpy()
Q = sp.load_npz(f"{B}/work/merfish_subset.npz")
qc = np.load(f"{B}/work/merfish_subset.npz.cells.npy")
qg = pd.read_csv(f"{B}/work/merfish_subset.npz.genes.tsv", sep="\t", header=None)[0].to_numpy()
rest = np.setdiff1d(np.intersect1d(qc, mf.cellId.to_numpy()), arm)
pick = np.sort(np.concatenate([arm, rng.choice(rest, NQ, replace=False)]))
rows = np.searchsorted(qc, pick)
assert (qc[rows] == pick).all()
keep = ~pd.Series(qg).str.startswith("Blank").to_numpy()
q = ad.AnnData(X=Q[rows][:, keep].astype(np.float32), obs=pd.DataFrame(index=pick.astype(str)),
               var=pd.DataFrame(index=qg[keep]))
q.write_h5ad(f"{B}/work/query_fb.h5ad")
json.dump({"None": list(qg[keep])}, open(f"{B}/work/markers_all500.json", "w"))
print("ref", ref.shape, "query", q.shape, "arm glia", len(arm))
