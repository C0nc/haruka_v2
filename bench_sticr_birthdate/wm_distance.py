"""Distance of Isocortex OPC (early 4/8 vs late 5/16) to the nearest fibre-tract (white-matter) cell, per section.
Run: python -I wm_distance.py <results_dir>"""
import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
res = sys.argv[1]
m = pd.read_parquet(f"{res}/merfish_fb_arms.parquet").merge(
    pd.read_parquet(f"{res}/merfish_meta.parquet", columns=["cellId", "center_x", "center_y"]), on="cellId")
E, L = {"OPC_4", "OPC_8"}, {"OPC_16", "OPC_5"}
rows = []
for sec, d in m.groupby("AP_order"):
    wm = d[d.region == "fiber tracts"][["center_x", "center_y"]].to_numpy()
    o = d[(d.region == "Isocortex") & (d.cls == "OPC_Oligo") & d.gcluster.isin(E | L)]
    if len(wm) < 100 or (o.gcluster.isin(L)).sum() < 50 or (o.gcluster.isin(E)).sum() < 50:
        continue
    dist, _ = cKDTree(wm).query(o[["center_x", "center_y"]].to_numpy())
    late = o.gcluster.isin(L).to_numpy()
    rows.append(dict(section=sec, med_early=np.median(dist[~late]), med_late=np.median(dist[late]),
                     auc=mannwhitneyu(dist[~late], dist[late]).statistic / ((~late).sum() * late.sum())))
r = pd.DataFrame(rows)
print(r.round(1).to_string()); print(r[["med_early", "med_late", "auc"]].median().round(3))
r.to_csv(f"{res}/opc_isocortex_wm_distance.tsv", sep="\t", index=False)
