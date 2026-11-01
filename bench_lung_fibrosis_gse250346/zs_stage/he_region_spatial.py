#!/usr/bin/env python3
"""Is the s advantage on H&E region pairs carried by the spatial salient readout?
Same pairs / cell types / cores as he_region_zs.py (region_pairs.csv).
Decoded k=20 neighbour composition: background D([z,0]) = background_composition.npy, condition D([z,s]) =
condition_composition.npy, salient shift = their difference (salient_composition_shift.npy).
Observed: each cell's k=20 nearest-neighbour cell-type shares within its core.
Per pair x cell type, within each core: region mean difference (R1 - R2) of each 47-vector, then averaged over cores.
  align_bg    = Pearson(observed diff, D([z,0]) diff)      -- what z alone predicts about the regions' niches
  align_cond  = Pearson(observed diff, D([z,s]) diff)      -- with s
  align_shift = Pearson(observed diff, shift diff)         -- the s part alone
  auc_shift   = held-out-donor within-core AUROC of a logistic on the 47-d salient shift (core-centred), as in A
Then: does s's AUROC gain over z track align_shift across pair x cell type?
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import os, warnings, json
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
H = Path(f'{_HR}'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/he_region'
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
pos = pd.Series(np.arange(len(o)), index=o.index)
BG = np.load(MD / 'background_composition.npy', mmap_mode='r'); CO = np.load(MD / 'condition_composition.npy', mmap_mode='r')
SH = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r')
d = pd.read_csv(f'{_HW}/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type', 'stage', 'core'])
d = d[d.stage != 'Unaffected'].drop_duplicates(['obs', 'annotation_type'])
d['i'] = pos.reindex(d.obs).to_numpy(); d['ct'] = o.cell_type.astype(str).to_numpy()[d.i]; d['donor'] = o.donor.astype(str).to_numpy()[d.i]
log = open(OUT / 'he_region_spatial.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
pd.set_option('display.width', 250)

# observed k=20 composition for fibrosis cores
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]; sl = o['sample'].astype(str).to_numpy()
cti = pd.Categorical(o.cell_type.astype(str), categories=types).codes; OBS = np.zeros((len(o), len(types)), np.float32)
for s_ in np.unique(d.core):
    i = np.flatnonzero(sl == s_); _, nb = NearestNeighbors(n_neighbors=21).fit(xy[i]).kneighbors(xy[i])
    lab = cti[i][nb[:, 1:]]
    OBS[i] = np.stack([np.bincount(r, minlength=len(types)) for r in lab]) / 20.0

A = pd.read_csv(OUT / 'region_pairs.csv')
rows = []
for r in A.itertuples():
    r1, r2 = r.pair.split(' vs ')
    q = d[d.annotation_type.isin([r1, r2])]; q = q[~q.obs.duplicated(keep=False)]; g = q[q.ct == r.cell_type]
    n = g.groupby(['core', 'annotation_type']).size().unstack(fill_value=0).reindex(columns=[r1, r2], fill_value=0)
    ok = n.index[(n[r1] >= 10) & (n[r2] >= 10)]; g = g[g.core.isin(ok)].sort_values('i')
    idx = g.i.to_numpy(); y = (g.annotation_type == r1).to_numpy(); core = g.core.to_numpy(); don = g.donor.to_numpy()
    M = {'obs': OBS[idx], 'bg': np.asarray(BG[idx]), 'cond': np.asarray(CO[idx]), 'shift': np.asarray(SH[idx])}
    diff = {k: np.mean([v[(core == c) & y].mean(0) - v[(core == c) & ~y].mean(0) for c in ok], 0) for k, v in M.items()}
    rec = dict(pair=r.pair, cell_type=r.cell_type, n=r.n, auc_z=r.auc_z, auc_s=r.auc_s,
               align_bg=np.corrcoef(diff['obs'], diff['bg'])[0, 1], align_cond=np.corrcoef(diff['obs'], diff['cond'])[0, 1],
               align_shift=np.corrcoef(diff['obs'], diff['shift'])[0, 1],
               obs_diff_L1=np.abs(diff['obs']).sum(), shift_diff_L1=np.abs(diff['shift']).sum(), bg_diff_L1=np.abs(diff['bg']).sum())
    X = M['shift'].astype(float).copy()
    for c in ok: X[core == c] -= X[core == c].mean(0)
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(min(5, len(np.unique(don)))).split(X, y, don):
        if len(np.unique(y[tr])) < 2: continue
        ss = StandardScaler().fit(X[tr]); sc[te] = LogisticRegression(C=0.1, max_iter=500, class_weight='balanced').fit(ss.transform(X[tr]), y[tr]).decision_function(ss.transform(X[te]))
    rec['auc_shift'] = np.mean([roc_auc_score(y[core == c], sc[core == c]) for c in ok])
    top = np.argsort(-np.abs(diff['obs']))[:3]
    rec['top_obs'] = '; '.join(f'{types[j]} {diff["obs"][j]:+.3f} (shift {diff["shift"][j]:+.3f}, bg {diff["bg"][j]:+.3f})' for j in top)
    rows.append(rec)
T = pd.DataFrame(rows); T['gain_s'] = T.auc_s - T.auc_z; T.to_csv(OUT / 'region_pairs_spatial.csv', index=False)

P('== decoded niche vs observed niche difference between H&E regions (same core, same cell type) ==')
for p, t in T.groupby('pair', sort=False):
    P(f'\n{p}: {len(t)} cell types | align D([z,0]) {t.align_bg.median():+.2f}  D([z,s]) {t.align_cond.median():+.2f}  shift {t.align_shift.median():+.2f}'
      f' | AUROC z {t.auc_z.median():.3f} s {t.auc_s.median():.3f} shift-only {t.auc_shift.median():.3f}'
      f' | |diff| obs {t.obs_diff_L1.median():.3f} bg {t.bg_diff_L1.median():.3f} shift {t.shift_diff_L1.median():.3f}')
    for x in t.sort_values('n', ascending=False).head(3).itertuples():
        P(f'   {x.cell_type}: align bg {x.align_bg:+.2f} cond {x.align_cond:+.2f} shift {x.align_shift:+.2f} | {x.top_obs}')
P(f'\nALL ({len(T)}): align D([z,0]) median {T.align_bg.median():+.2f}, D([z,s]) {T.align_cond.median():+.2f} '
  f'(cond > bg in {(T.align_cond > T.align_bg).sum()}/{len(T)}, Wilcoxon p {wilcoxon(T.align_cond, T.align_bg).pvalue:.2g}), shift {T.align_shift.median():+.2f} (>0 in {(T.align_shift > 0).sum()}/{len(T)})')
P(f'  shift-only AUROC median {T.auc_shift.median():.3f} vs z {T.auc_z.median():.3f} vs s {T.auc_s.median():.3f}')
P(f'  does s gain track spatial alignment? Spearman(gain_s, align_shift) {spearmanr(T.gain_s, T.align_shift).statistic:+.2f} (p {spearmanr(T.gain_s, T.align_shift).pvalue:.2g});'
  f' Spearman(gain_s, auc_shift) {spearmanr(T.gain_s, T.auc_shift).statistic:+.2f} (p {spearmanr(T.gain_s, T.auc_shift).pvalue:.2g})')
P('done')
