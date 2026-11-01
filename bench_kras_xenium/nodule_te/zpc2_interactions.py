#!/usr/bin/env python3
"""Cell-type interactions around treated nodules at the two ends of the background axis z-PC2 (name_bad_te_z.py).
High = top tertile of z-PC2 within each treated section ('worse TE'), low = bottom tertile ('better TE').
Nodule niche = the nodule's tumour cells + every cell within 50 um of them (same section).
A composition: share of each (non-tumour) cell group in the niche, per nodule; high vs low per section + pooled
  Mann-Whitney (published scipy test), median of per-section differences.
B squidpy.gr.nhood_enrichment (spatial kNN 10, 1000 permutations) per section on high-niche cells and on low-niche cells
  separately; z-scores averaged over sections; reported for the Tumour row and the difference high - low.
C LIANA+ rank_aggregate (resource 'mouseconsensus', expr_prop 0.1, X = normalize+log1p as stored) on high-niche and
  low-niche cells (<= 40k each, stratified by section); interactions between Tumour and Monocyte / macrophages / AT1 /
  Ciliated; magnitude_rank compared high vs low.
D two example nodules (highest and lowest z-PC2 in the largest section) for zoomed maps.
"""
import os, json, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, squidpy as sq, liana as li
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from sklearn.neighbors import radius_neighbors_graph
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_kras_xenium/nodule_te/zpc2_interactions'; OUT.mkdir(exist_ok=True)
a = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r'); o = a.obs
TUMOR = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']
ct = o.celltype.astype(str).to_numpy(); sl = o.slice_id.astype(str).to_numpy(); treated = (o.condition_label.astype(str) == 'KRASi_treated').to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]; tum = np.isin(ct, TUMOR)
def grp(c):
    if c in TUMOR: return 'Tumour'
    if c.startswith(('Macrophage_alveolar',)): return 'Macrophage alveolar'
    if c in ('Macrophage_interstitial', 'moMacrophage'): return 'Macrophage interstitial/mo'
    if c == 'Monocyte': return 'Monocyte'
    if c in ('cDC1', 'moDC', 'mregDC', 'pDC'): return 'DC'
    if c == 'Neutrophil': return 'Neutrophil'
    if c in ('T_cell', 'CD8_T', 'T_proliferating', 'NK'): return 'T/NK'
    if c == 'B_cell': return 'B'
    if c in ('AT1', 'AT2', 'Ciliated'): return c
    if c.endswith('_EC'): return 'Endothelial'
    if c.startswith('Fibroblast'): return 'Fibroblast'
    if c in ('Pericyte', 'SmoothMuscle'): return 'Pericyte/SMC'
    return 'Other'
G = np.array([grp(c) for c in ct])
log = open(OUT / 'zpc2_interactions.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
# nodules, same order as bad_te_background.py / name_bad_te_z.py
TR = []
for s_ in np.unique(sl[treated]):
    q = np.flatnonzero((sl == s_) & tum); _, lab = connected_components(radius_neighbors_graph(xy[q], 25.0, mode='connectivity'), directed=False)
    for l in np.flatnonzero(np.bincount(lab) >= 50): TR.append((s_, q[lab == l]))
F2 = np.load(H / 'bench_kras_xenium/nodule_te/name_zPC2_fig.npz', allow_pickle=True); score = F2['treated_score']; assert len(score) == len(TR)
sec = np.array([s for s, _ in TR]); lev = np.full(len(TR), 'mid', object)
for u in np.unique(sec):
    m = np.flatnonzero(sec == u); qs = np.quantile(score[m], [1 / 3, 2 / 3]); lev[m[score[m] >= qs[1]]] = 'high'; lev[m[score[m] <= qs[0]]] = 'low'
P('nodules per level:', pd.Series(lev).value_counts().to_dict())
niche = []
for k, (s_, idx) in enumerate(TR):
    m = np.flatnonzero(sl == s_); d, _ = cKDTree(xy[idx]).query(xy[m]); niche.append(m[d <= 50])
# A composition
groups = [g for g in sorted(set(G)) if g not in ('Tumour', 'Other')]
C = pd.DataFrame([{**{g: (G[n][G[n] != 'Tumour'] == g).mean() for g in groups}, 'section': s_, 'level': lev[k], 'z_pc2': score[k]} for k, ((s_, _), n) in enumerate(zip(TR, niche))])
C.to_csv(OUT / 'A_niche_composition.csv', index=False)
rows = []
for g in groups:
    d = [C[(C.section == u) & (C.level == 'high')][g].mean() - C[(C.section == u) & (C.level == 'low')][g].mean() for u in np.unique(sec)]
    p = mannwhitneyu(C[C.level == 'high'][g], C[C.level == 'low'][g]).pvalue
    rows.append(dict(group=g, high=C[C.level == 'high'][g].mean(), low=C[C.level == 'low'][g].mean(), diff_median=np.median(d), sections_same_sign=int((np.sign(d) == np.sign(np.median(d))).sum()), p=p))
A = pd.DataFrame(rows).sort_values('diff_median'); A.to_csv(OUT / 'A_summary.csv', index=False)
P('\n== A. non-tumour niche composition, high vs low z-PC2 nodules =='); P(A.round(4).to_string(index=False))
# B squidpy nhood enrichment
cats = sorted(set(G))
Z = {}
for L in ['high', 'low']:
    zs = []
    for u in np.unique(sec):
        cells = np.unique(np.concatenate([niche[k] for k in np.flatnonzero((sec == u) & (lev == L))]))
        A_ = ad.AnnData(np.zeros((len(cells), 1), np.float32)); A_.obsm['spatial'] = xy[cells]
        A_.obs['g'] = pd.Categorical(G[cells], categories=cats)
        sq.gr.spatial_neighbors(A_, coord_type='generic', n_neighs=10)
        sq.gr.nhood_enrichment(A_, cluster_key='g', n_perms=1000, seed=0, show_progress_bar=False)
        zs.append(pd.DataFrame(A_.uns['g_nhood_enrichment']['zscore'], index=cats, columns=cats))
    Z[L] = sum(zs) / len(zs); Z[L].to_csv(OUT / f'B_nhood_z_{L}.csv')
D = Z['high'] - Z['low']; D.to_csv(OUT / 'B_nhood_z_diff.csv')
P('\n== B. squidpy nhood enrichment z (mean over sections), Tumour row =='); P(pd.DataFrame({'high': Z['high'].loc['Tumour'], 'low': Z['low'].loc['Tumour'], 'diff': D.loc['Tumour']}).sort_values('diff').round(2).to_string())
# C LIANA+
rng = np.random.default_rng(0); RES = {}
for L in ['high', 'low']:
    cells = np.unique(np.concatenate([niche[k] for k in np.flatnonzero(lev == L)]))
    if len(cells) > 40000: cells = np.sort(rng.choice(cells, 40000, replace=False))
    A_ = a[cells].to_memory(); A_.obs['g'] = pd.Categorical(G[cells]); A_ = A_[A_.obs.g != 'Other'].copy()
    li.mt.rank_aggregate(A_, groupby='g', resource_name='mouseconsensus', expr_prop=0.1, use_raw=False, verbose=False, n_perms=None)
    r = A_.uns['liana_res']; r['level'] = L; RES[L] = r; r.to_csv(OUT / f'C_liana_{L}.csv.gz', index=False)
    P(f'liana {L}: {len(cells)} cells, {len(r)} interactions')
PART = ['Monocyte', 'Macrophage alveolar', 'Macrophage interstitial/mo', 'AT1', 'Ciliated']
key = ['source', 'target', 'ligand_complex', 'receptor_complex']
M = RES['high'][key + ['magnitude_rank', 'lr_means']].merge(RES['low'][key + ['magnitude_rank', 'lr_means']], on=key, how='outer', suffixes=('_high', '_low'))
M = M[((M.source == 'Tumour') & M.target.isin(PART)) | (M.source.isin(PART) & (M.target == 'Tumour'))]
M[['magnitude_rank_high', 'magnitude_rank_low']] = M[['magnitude_rank_high', 'magnitude_rank_low']].fillna(1.0)
M['log_ratio'] = np.log10(M.magnitude_rank_low / M.magnitude_rank_high)   # > 0 = stronger (lower rank) in high
M.to_csv(OUT / 'C_tumour_partner_lr.csv', index=False)
top = M[(M.magnitude_rank_high < 0.05) | (M.magnitude_rank_low < 0.05)].sort_values('log_ratio')
P('\n== C. tumour <-> partner LR (magnitude rank < 0.05 in either), most HIGH-specific =='); P(top.tail(15)[key + ['magnitude_rank_high', 'magnitude_rank_low', 'log_ratio']].to_string(index=False))
P('most LOW-specific'); P(top.head(10)[key + ['magnitude_rank_high', 'magnitude_rank_low', 'log_ratio']].to_string(index=False))
# D examples
bs = pd.Series(sec).value_counts().idxmax(); m = np.flatnonzero(sec == bs)
hi, lo = m[np.argmax(score[m])], m[np.argmin(score[m])]
np.savez(OUT / 'D_examples.npz', section=bs, hi_cells=niche[hi], lo_cells=niche[lo], hi_tum=TR[hi][1], lo_tum=TR[lo][1], hi_score=score[hi], lo_score=score[lo],
         hi_xy=xy[niche[hi]], lo_xy=xy[niche[lo]], hi_g=G[niche[hi]], lo_g=G[niche[lo]], hi_ct=ct[niche[hi]], lo_ct=ct[niche[lo]])
P(f'\nexamples in {bs}: high nodule z-PC2 {score[hi]:+.2f} ({len(TR[hi][1])} tumour cells), low {score[lo]:+.2f} ({len(TR[lo][1])})')
P('done')
