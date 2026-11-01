"""Isocortex OPC 4/8 vs 5/16: match early to late on distance-to-white-matter (50 um bins, per section, up to 3:1),
then re-measure arm separability from coordinates and from k=20 neighbour broad-class composition.
Writes the matched cell list. Run: python -I wm_matched.py <results_dir>"""
import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
res = sys.argv[1]; rng = np.random.default_rng(0); K = 20
m = pd.read_parquet(f"{res}/merfish_fb_arms.parquet").merge(
    pd.read_parquet(f"{res}/merfish_meta.parquet", columns=["cellId", "center_x", "center_y"]), on="cellId")
m["cls_all"] = m.cls.fillna("NA"); classes = sorted(m.cls_all.unique())
E, L = {"OPC_4", "OPC_8"}, {"OPC_16", "OPC_5"}
rows, keep_ids = [], []
for sec, d in m.groupby("AP_order"):
    wm = d[d.region == "fiber tracts"][["center_x", "center_y"]].to_numpy()
    o = d[(d.region == "Isocortex") & (d.cls == "OPC_Oligo") & d.gcluster.isin(E | L)]
    if len(wm) < 100 or o.gcluster.isin(L).sum() < 50 or o.gcluster.isin(E).sum() < 50:
        continue
    xy = d[["center_x", "center_y"]].to_numpy()
    _, nn = cKDTree(xy).query(xy, K + 1)
    oh = pd.get_dummies(d.cls_all).reindex(columns=classes, fill_value=0).to_numpy(float)
    comp = pd.DataFrame(oh[nn[:, 1:]].mean(1), index=d.index)
    dist, _ = cKDTree(wm).query(o[["center_x", "center_y"]].to_numpy())
    o = o.assign(b=np.minimum(dist // 50, 20), late=o.gcluster.isin(L))
    sel = []
    for b, g in o.groupby("b"):
        lt, er = g[g.late], g[~g.late]
        n = min(len(er), 3 * len(lt))
        sel += list(lt.index) + list(rng.choice(er.index, n, replace=False)) if n > 0 else []
    u = o.loc[sel]
    y = u.late.to_numpy(int)
    if y.sum() < 50 or (1 - y).sum() < 50:
        continue
    keep_ids += list(u.cellId)
    X = u[["center_x", "center_y"]].to_numpy(); C = comp.loc[u.index].to_numpy()
    s1 = np.zeros(len(y)); s2 = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        s1[te] = KNeighborsClassifier(K).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        s2[te] = LogisticRegression(max_iter=500).fit(C[tr], y[tr]).predict_proba(C[te])[:, 1]
    rows.append(dict(section=sec, n_early=int((1 - y).sum()), n_late=int(y.sum()),
                     auc_xy=roc_auc_score(y, s1), auc_nbcomp=roc_auc_score(y, s2)))
r = pd.DataFrame(rows); r.to_csv(f"{res}/opc_isocortex_wm_matched.tsv", sep="\t", index=False)
np.save(f"{res}/opc_isocortex_wm_matched_cellids.npy", np.array(keep_ids))
print(len(r), "sections", r.n_early.sum(), "early", r.n_late.sum(), "late")
print(r[["auc_xy", "auc_nbcomp"]].median().round(3))
