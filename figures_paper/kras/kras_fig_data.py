#!/usr/bin/env python3
"""Data for the KRAS-inhibitor figure (make_kras_figure.py).  Fit: results_kras_te_null (the refit with the saved model and
3 shuffled-s nulls; same settings as results_real_compgated_nb_final).  Data: mouselung_kras_xenium_clean_noC3.h5ad,
4 control (JB_C*) and 4 KRASi-treated (JB_M*) sections, separate Xenium runs per arm.
Nodules exactly as bench_kras_xenium/nodule_te/nodule_te.py: tumour cells (Tumor_EMT / Tumor_epithelial /
Tumor_epithelial_Itgb6), radius graph 25 um within a section, connected components with >= 50 cells; nodule TE = mean
over its tumour cells of the salient composition shift D([z,s]) - D([z,0]).
Outputs (figures_paper/kras/data/):
  sections.csv       per section: arm, tumour burden (tumour share of all cells), Itgb6+ share of tumour, responder rate
  subtype_resp.csv   treated: responder rate per tumour subtype (MAPK MPAS < control p25 of the same subtype)
  example.npz        one treated section: xy, cell type, MPAS (tumour cells), nodule id + nodule Itgb6-neighbour TE
  umap.npz           treated cells (<= 40k, stratified by cell type): UMAP of z and s, cell type, MPAS (tumour cells)
"""
import os, json, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, umap
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'figures_paper/kras/data'; OUT.mkdir(parents=True, exist_ok=True)
MD = H / 'results_kras_te_null/mouselung_kras_xenium_clean_noC3/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r'); o = a.obs
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
TUMOR = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']
ct = o.celltype.astype(str).to_numpy(); sl = o.slice_id.astype(str).to_numpy(); treated = (o.condition_label.astype(str) == 'KRASi_treated').to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]; tum = np.isin(ct, TUMOR)
sig = pd.read_csv(H / 'bench_kras_xenium/biological_responder/all_tumor_signature_scores.csv.gz')
assert len(sig) == tum.sum() and (sig.ct.to_numpy() == ct[tum]).all()
mpas = np.full(len(ct), np.nan); mpas[tum] = sig.MAPK_MPAS.to_numpy()
p25 = {t: np.nanpercentile(mpas[tum & ~treated & (ct == t)], 25) for t in TUMOR}
resp = tum & (np.nan_to_num(mpas, nan=np.inf) < np.array([p25.get(c, -np.inf) for c in ct]))
rows = []
for s_ in np.unique(sl):
    m = sl == s_; t_ = m & tum
    rows.append(dict(section=s_, arm='treated' if treated[m][0] else 'control', burden=t_.sum() / m.sum(),
                     itgb6_share=(ct[t_] == 'Tumor_epithelial_Itgb6').mean(), resp_rate=resp[t_].mean()))
pd.DataFrame(rows).to_csv(OUT / 'sections.csv', index=False); print(pd.DataFrame(rows).round(3))
pd.DataFrame([dict(subtype=t, section=s_, resp_rate=resp[(ct == t) & (sl == s_)].mean(), n=((ct == t) & (sl == s_)).sum())
              for t in TUMOR for s_ in np.unique(sl[treated])]).to_csv(OUT / 'subtype_resp.csv', index=False)
# example treated section with its nodules
dp = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r'); jI = types.index('Tumor_epithelial_Itgb6')
cnt = {}
for s_ in np.unique(sl[treated]):
    q = np.flatnonzero((sl == s_) & tum); _, lab = connected_components(radius_neighbors_graph(xy[q], 25.0, mode='connectivity'), directed=False)
    cnt[s_] = (q, lab, (np.bincount(lab) >= 50).sum())
ex = max(cnt, key=lambda k: cnt[k][2]); q, lab, _ = cnt[ex]
nod = np.full(len(q), -1); te = np.full(len(q), np.nan); big = np.flatnonzero(np.bincount(lab) >= 50)
sh = np.asarray(dp[q])[:, jI]
for i, l in enumerate(big): m = lab == l; nod[m] = i; te[m] = sh[m].mean()
mx = sl == ex
np.savez(OUT / 'example.npz', section=ex, xy=xy[mx], cell_type=ct[mx], mpas=mpas[mx], tum_xy=xy[q], tum_ct=ct[q], nodule=nod, nodule_te_itgb6=te)
print('example section', ex, 'nodules', len(big))
# UMAPs of treated cells
rng = np.random.default_rng(0); tri = np.flatnonzero(treated)
pick = np.sort(np.concatenate([g if len(g) <= 2500 else rng.choice(g, 2500, replace=False) for g in pd.Series(tri).groupby(ct[tri]).apply(np.asarray)]))
res = {}
for k, f in [('z', 'background.npy'), ('s', 'salient.npy')]:
    res[k] = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(StandardScaler().fit_transform(np.asarray(np.load(MD / f, mmap_mode='r')[pick])))
np.savez(OUT / 'umap.npz', umap_z=res['z'], umap_s=res['s'], cell_type=ct[pick], mpas=mpas[pick], resp=resp[pick])
print('umap cells', len(pick))
