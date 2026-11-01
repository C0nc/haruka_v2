#!/usr/bin/env python3
"""Within-section tumour-vs-non-tumour design on PEtracer M2-1 (read-only on sources).

control = non-malignant cells outside the tumours (beyond a buffer band)
disease = non-malignant cells inside the tumours (beyond a buffer band)
halo    = malignant cells, buffer-band cells, and cell types not present in both
          regions with >=200 cells: neighbours-only (kept in k=20 niche targets,
          excluded from both training arms via --exclude_label halo).
Segmentation: authors' obs['within_tumor'] / tumor_boundary_dist, cross-checked with
k-means on k=20 neighbour cell-type composition.
"""
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

SRC = Path("/home/yancui/Jill/fourteen/petracer_data/M2_tumor_tracing.h5td")
OUT = Path("/home/yancui/haruka_v2/bench_petracer_lineage/reaudit_2026-10-16")
SECTION, K, MIN_CELLS = "M2-1", 20, 200

with h5py.File(SRC, "r") as h:
    obs = ad.io.read_elem(h["obs"])
    var = ad.io.read_elem(h["var"])
    counts = ad.io.read_elem(h["layers/counts"]).tocsr()
    xy = np.asarray(h["obsm/spatial"], dtype=np.float32)
keep = obs["sample"].astype(str).to_numpy() == SECTION
obs, counts, xy = obs.loc[keep].copy(), counts[keep], xy[keep]
n = len(obs)
ct = obs["cell_subtype"].astype(str).to_numpy()
inside = obs["within_tumor"].astype(bool).to_numpy()
dist = obs["tumor_boundary_dist"].to_numpy(float)
signed = np.where(inside, dist, -dist)  # + inside tumour, - outside
mal = ct == "Malignant"

tree = cKDTree(xy)
dd, nb = tree.query(xy, k=K + 1)
nb = nb[:, 1:]
r20 = float(np.median(dd[:, K]))
cats = sorted(np.unique(ct))
code = pd.Categorical(ct, categories=cats).codes
comp = np.zeros((n, len(cats)), np.float32)
for j in range(K):
    np.add.at(comp, (np.arange(n), code[nb[:, j]]), 1.0)
comp /= K
mal_frac = comp[:, cats.index("Malignant")]

# --- unsupervised cross-check: k-means on k=20 composition --------------------
seg = {"median_dist_to_20th_neighbour_um": r20}
for k in (2, 4, 6, 8, 10):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(comp)
    # call a cluster 'tumour' if its mean malignant-neighbour fraction > 0.2 (label-free rule)
    tum_cl = [c for c in range(k) if mal_frac[lab == c].mean() > 0.2]
    pred = np.isin(lab, tum_cl)
    seg[f"kmeans_k{k}"] = dict(
        ARI_vs_within_tumor=float(adjusted_rand_score(inside, pred)),
        agreement=float((pred == inside).mean()),
        n_tumour_clusters=len(tum_cl))
# simple threshold baseline: malignant-neighbour fraction > 0.2
seg["malfrac_gt_0.2"] = dict(ARI_vs_within_tumor=float(adjusted_rand_score(inside, mal_frac > 0.2)),
                             agreement=float(((mal_frac > 0.2) == inside).mean()))

# --- buffer band ----------------------------------------------------------------
BUFFER = float(max(50.0, 2 * r20))
far = np.abs(signed) >= BUFFER
# disagreement between authors' label and unsupervised k=6 call, away from boundary
lab6 = KMeans(6, n_init=10, random_state=0).fit_predict(comp)
pred6 = np.isin(lab6, [c for c in range(6) if mal_frac[lab6 == c].mean() > 0.2])
seg["kmeans_k6_agreement_outside_buffer"] = float((pred6[far] == inside[far]).mean())
seg["buffer_um"] = BUFFER
# outside-arm contamination check: malignant neighbours among control candidates
cand_ctrl = (~inside) & far & ~mal
seg["control_candidates_with_any_malignant_nbr_frac"] = float((mal_frac[cand_ctrl] > 0).mean())
seg["control_candidates_mean_malignant_nbr_frac"] = float(mal_frac[cand_ctrl].mean())
cand_dis = inside & far & ~mal
seg["disease_candidates_mean_malignant_nbr_frac"] = float(mal_frac[cand_dis].mean())

# --- arms -----------------------------------------------------------------------
tab = pd.crosstab(ct[far & ~mal], np.where(inside[far & ~mal], "tumour", "nontumour"))
shared = tab[(tab["tumour"] >= MIN_CELLS) & (tab["nontumour"] >= MIN_CELLS)].index.tolist()
cond = np.full(n, "halo", dtype=object)
ok = far & ~mal & np.isin(ct, shared)
cond[ok & ~inside] = "control"
cond[ok & inside] = "disease"
arm_tab = pd.crosstab(ct, cond)
arm_tab.to_csv(OUT / "m2_arm_counts_by_celltype.csv")
print(arm_tab.to_string())

o = pd.DataFrame(index=pd.Index([f"{SECTION}_{x}" for x in obs.index.astype(str)]))
o["condition"] = pd.Categorical(cond)
o["cell_type"] = pd.Categorical(ct)
o["slice_id"] = SECTION
o["within_tumor"] = inside
o["signed_boundary_dist"] = signed
o["tumor"] = obs["tumor"].astype(str).to_numpy()
o["malignant_nbr_frac_k20"] = mal_frac
o["kmeans6_tumour_call"] = pred6
o["cell_neighborhood"] = obs["cell_neighborhood"].astype(str).to_numpy()
o["total_counts"] = np.asarray(counts.sum(1)).ravel()
a = ad.AnnData(X=counts.astype(np.float32).copy(), obs=o, var=pd.DataFrame(index=var.index.astype(str)))
a.layers["counts"] = counts.astype(np.float32).copy()
a.obsm["spatial"] = xy
a.uns["design"] = dict(section=SECTION, buffer_um=BUFFER, shared_types=shared, min_cells=MIN_CELLS,
                       halo="malignant + buffer band + types not in both regions; neighbours-only")
path = OUT / "petracer_M2-1_tumour_vs_nontumour.h5ad"
a.write_h5ad(path, compression="gzip")
seg["arm_counts"] = pd.Series(cond).value_counts().to_dict()
seg["shared_types"] = shared
json.dump(seg, open(OUT / "m2_segmentation_check.json", "w"), indent=2)
print(json.dumps(seg, indent=2))
print(path)
