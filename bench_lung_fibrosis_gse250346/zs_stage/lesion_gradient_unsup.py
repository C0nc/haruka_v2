#!/usr/bin/env python3
"""Unsupervised version of lesion_gradient.py part A: no model is fitted to distance.
Same cells as lesion_gradient.py (fibrosis cores with >= 30 cells of lesion L; non-L cells within 400 um of the nearest
L cell).  Per lesion L x cell type (>= 1000 cells, >= 4 donors; <= 15000 cells subsampled):
  latent (z or s) standardised and centred per core x cell type, then
  PC1, PC2 (PCA on that cell type's centred latent) and DPT (scanpy: 15-NN on the centred latent, diffmap, root = cell
  with the lowest diffusion component 1 -- chosen from the latent alone, never from distance).
Readout: Pearson r(axis, distance in um) inside each core (>= 50 cells), averaged over cores weighted by n while
keeping the sign (an axis must point the same way in every core), then |.|.  z vs s paired over lesion x cell type.
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import os, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = Path(f'{_HR}'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/lesion_gradient'
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
o = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r').obs
log = open(OUT / 'lesion_gradient_unsup.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
pd.set_option('display.width', 250)
stage = o.sample_affect.astype(str).to_numpy(); core = o['sample'].astype(str).to_numpy(); ctA = o.cell_type.astype(str).to_numpy(); donA = o.donor.astype(str).to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]
pos = pd.Series(np.arange(len(o)), index=o.index)
he = pd.read_csv(f'{_HW}/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type']); he['i'] = pos.reindex(he.obs).to_numpy()
Zm = np.load(MD / 'background.npy', mmap_mode='r'); Sm = np.load(MD / 'salient.npy', mmap_mode='r')
def centre(M, g):
    M = np.asarray(M, np.float64).copy()
    for k in pd.Series(np.arange(len(g))).groupby(g).indices.values(): M[k] -= M[k].mean(0)
    return M
def signed_mean_r(ax, d, cc):
    r, w = [], []
    for q in np.unique(cc):
        m = cc == q
        if m.sum() < 50 or ax[m].std() == 0: continue
        r.append(np.corrcoef(ax[m], d[m])[0, 1]); w.append(m.sum())
    return abs(np.average(r, weights=w)) if r else np.nan
def dpt(X, seed=0):
    A = ad.AnnData(X.astype(np.float32)); sc.pp.neighbors(A, n_neighbors=15, use_rep='X', random_state=seed)
    sc.tl.diffmap(A, n_comps=10); A.uns['iroot'] = int(np.argmin(A.obsm['X_diffmap'][:, 1])); sc.tl.dpt(A)
    return A.obs['dpt_pseudotime'].to_numpy()
LES = ['fibroblastic_focus', 'mixed_inflammation', 'TLS', 'remodeled_epithelium', 'microscopic_honeycombing', 'severe_fibrosis']
fibcores = set(core[stage != 'Unaffected']); rng = np.random.default_rng(0); rows = []
for L in LES:
    li = he.i[he.annotation_type == L].to_numpy(); idx_all, dist_all = [], []
    for c in sorted(fibcores):
        lc = li[core[li] == c]
        if len(lc) < 30: continue
        ci = np.flatnonzero(core == c); ci = ci[~np.isin(ci, lc)]
        d, _ = cKDTree(xy[lc]).query(xy[ci]); k = d <= 400; idx_all.append(ci[k]); dist_all.append(d[k])
    idx = np.concatenate(idx_all); dist = np.concatenate(dist_all)
    for t in np.unique(ctA[idx]):
        m = np.flatnonzero(ctA[idx] == t)
        if len(m) < 1000 or len(np.unique(donA[idx[m]])) < 4: continue
        if len(m) > 15000: m = np.sort(rng.choice(m, 15000, replace=False))
        ii = idx[m]; o_ = np.argsort(ii); ii, d = ii[o_], dist[m][o_]; cc = core[ii]
        rec = dict(lesion=L, cell_type=t, n=len(ii), cores=len(np.unique(cc)))
        for k, M in [('z', Zm), ('s', Sm)]:
            X = centre(StandardScaler().fit_transform(np.asarray(M[ii])), cc)
            pc = PCA(2, random_state=0).fit_transform(X)
            rec[f'PC1_{k}'] = signed_mean_r(pc[:, 0], d, cc); rec[f'PC2_{k}'] = signed_mean_r(pc[:, 1], d, cc)
            pt = dpt(X); ok = np.isfinite(pt); rec[f'DPT_{k}'] = signed_mean_r(pt[ok], d[ok], cc[ok])
        rows.append(rec)
    T = pd.DataFrame([r for r in rows if r['lesion'] == L])
    P(f'\n### {L}: {len(T)} cell types, {T.cores.max()} cores')
    for ax in ['PC1', 'PC2', 'DPT']:
        t = T[[f'{ax}_z', f'{ax}_s']].dropna()
        P(f'  {ax}: median |r| z {t[f"{ax}_z"].median():.3f}  s {t[f"{ax}_s"].median():.3f} | s > z in {(t[f"{ax}_s"] > t[f"{ax}_z"]).sum()}/{len(t)}'
          + (f' | Wilcoxon p {wilcoxon(t[f"{ax}_s"], t[f"{ax}_z"]).pvalue:.2g}' if len(t) >= 6 else ''))
    P(T.sort_values('n', ascending=False).head(6)[['cell_type', 'n', 'cores', 'PC1_z', 'PC1_s', 'PC2_z', 'PC2_s', 'DPT_z', 'DPT_s']].round(3).to_string(index=False))
T = pd.DataFrame(rows); T.to_csv(OUT / 'unsup_axes_vs_distance.csv', index=False)
P(f'\n== ALL lesion x cell type ({len(T)}) ==')
for ax in ['PC1', 'PC2', 'DPT']:
    t = T[[f'{ax}_z', f'{ax}_s']].dropna()
    P(f'  {ax}: median |r| z {t[f"{ax}_z"].median():.3f}  s {t[f"{ax}_s"].median():.3f} | s > z in {(t[f"{ax}_s"] > t[f"{ax}_z"]).sum()}/{len(t)} | Wilcoxon p {wilcoxon(t[f"{ax}_s"], t[f"{ax}_z"]).pvalue:.2g}')
best = lambda k: T[[f'PC1_{k}', f'PC2_{k}', f'DPT_{k}']].max(1)
P(f'  best of PC1/PC2/DPT: median z {best("z").median():.3f}  s {best("s").median():.3f} | s > z in {(best("s") > best("z")).sum()}/{len(T)} | Wilcoxon p {wilcoxon(best("s"), best("z")).pvalue:.2g}')
P('done')
