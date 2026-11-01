"""Choose the MERFISH cell subset S whose counts we extract: forebrain Astro/OPC_Oligo arm cells (early/late),
their 30 nearest neighbours in the same section (halo candidates), all early/late EX/IN in HIP/RHP (spare),
plus a random 300k forebrain sample for mapping QC.  Run: python -I select_cells.py <results_dir>
"""
import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
res = sys.argv[1]
rng = np.random.default_rng(0)
mf = pd.read_parquet(f"{res}/merfish_meta.parquet", columns=["cellId", "center_x", "center_y", "AP_order"])
fb = pd.read_parquet(f"{res}/merfish_fb_arms.parquet")
mf = mf.merge(fb[["cellId", "cls", "arm", "region"]], on="cellId", how="left")
# candidate designs from design_v4 (most intermixed within-region contrasts)
cand = ((mf.cls == "OPC_Oligo") & mf.region.isin(["Isocortex", "fiber tracts"])) | \
       ((mf.cls == "Astro") & mf.region.isin(["HIP", "RHP", "HY"]))
core = mf.arm.isin(["early", "late"]) & cand
keep = core.copy()
for sec, d in mf.groupby("AP_order"):
    c = d[core.loc[d.index]]
    if len(c) == 0:
        continue
    _, nn = cKDTree(d[["center_x", "center_y"]].to_numpy()).query(c[["center_x", "center_y"]].to_numpy(), 21)
    keep.loc[d.index[np.unique(nn.ravel())]] = True
fbids = mf.cellId[mf.region.notna()].to_numpy()
rand = rng.choice(fbids, 300000, replace=False)
ids = np.union1d(mf.cellId[keep].to_numpy(), rand)
np.save(f"{res}/../work/subset_cellids.npy", ids)
print("core", core.sum(), "with halo", keep.sum(), "total subset", len(ids))
