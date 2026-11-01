#!/usr/bin/env python3
"""Does Haruka's s capture PEtracer tumour lineage? (M2-1, malignant cells of the condition arm)

Representations: z, s (Haruka, trained on tumour-region vs non-tumour cells; lineage never seen)
  and scVI-32 fitted on the same cells (baseline).
All tests use spatially blocked cross-validation (150 um blocks, GroupKFold 5): clones and clades
are spatially clustered, so random splits would reward location.
  T1 clone across tumours (each tumour ~ one clone: lineage = tumour identity)
  T2 within tumour 2: clone 2-1 vs 2-2 (two lineages in the same tumour)
  T3 within clone 5: subclade (tracing cells, 14 clades with >= 20 cells)
  T4 within clone 5: fitness (Spearman)
  T5 within clone 5: lineage character distance vs latent distance (Spearman), and partial on
     spatial distance
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')
D = Path(f'{_HR}/bench_petracer_lineage/m2_lineage')
MD = D / 'fits/petracer_M2-1_lineage/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
log = open(D / 'lineage_test.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
a = ad.read_h5ad(D / 'petracer_M2-1_lineage.h5ad'); o = a.obs
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names)).all()
m = ((o.cell_type.astype(str) == 'Malignant') & (o.condition.astype(str) == 'disease')).to_numpy()
C = a.layers['counts']; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
L = C.multiply(1e4 / np.asarray(C.sum(1)).ravel().clip(1)[:, None]).tocsr(); L.data = np.log1p(L.data)
xy = np.asarray(a.obsm['spatial'], float)
sv = dict(zip(np.load(D / 'scvi32_obs_names.npy', allow_pickle=True).astype(str), np.load(D / 'scvi32.npy')))
R = {'z': np.load(MD / 'background.npy')[m], 's': np.load(MD / 'salient.npy')[m],
     'scVI': np.stack([sv[n] for n in np.asarray(a.obs_names)[m]])}
ob = o[m].reset_index(drop=True); X_ = xy[m]
blocks = (np.floor(X_[:, 0] / 150).astype(int) * 10000 + np.floor(X_[:, 1] / 150).astype(int))
P(f'malignant condition cells {m.sum()}; spatial blocks {len(np.unique(blocks))}')

def cv_cls(Xr, y, grp, kind='bal'):
    pred = np.empty(len(y), object); sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xr, y, grp):
        if len(np.unique(y[tr])) < 2: continue
        ss = StandardScaler().fit(Xr[tr]); clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced').fit(ss.transform(Xr[tr]), y[tr])
        if kind == 'auc': sc[te] = clf.decision_function(ss.transform(Xr[te]))
        else: pred[te] = clf.predict(ss.transform(Xr[te]))
    return roc_auc_score(y, sc) if kind == 'auc' else balanced_accuracy_score(y.astype(str), pred.astype(str))

def cv_reg(Xr, y, grp):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xr, y, grp):
        ss = StandardScaler().fit(Xr[tr]); pred[te] = Ridge(10.0).fit(ss.transform(Xr[tr]), y[tr]).predict(ss.transform(Xr[te]))
    return spearmanr(pred, y).statistic

res = []
def report(test, vals, chance):
    P(f'   {test:46s} ' + '  '.join(f'{k} {v:.3f}' for k, v in vals.items()) + f'   (chance {chance})')
    res.append(dict(test=test, chance=chance, **vals))

P('\n-- T1 clone across tumours (bal. acc) --')
cl = ob.lin_clone.astype(str).to_numpy(); keep = np.isin(cl, [c for c, n in pd.Series(cl).value_counts().items() if c != 'nan' and n >= 100])
report('T1 clone across tumours', {k: cv_cls(v[keep], cl[keep], blocks[keep]) for k, v in R.items()}, f'{1 / len(np.unique(cl[keep])):.2f}')
P('\n-- T2 within tumour 2: clone 2-1 vs 2-2 (AUROC) --')
k2 = np.isin(cl, ['2-1', '2-2']); y2 = cl[k2] == '2-2'
report('T2 clone 2-1 vs 2-2 within tumour 2', {k: cv_cls(v[k2], y2, blocks[k2], 'auc') for k, v in R.items()}, '0.5')
P('\n-- T3 within clone 5: subclade (bal. acc) --')
cd = ob.lin_clade.astype(str).to_numpy(); k3 = (cl == '5') & (cd != 'nan')
big = [c for c, n in pd.Series(cd[k3]).value_counts().items() if n >= 20]; k3 &= np.isin(cd, big)
report(f'T3 subclade within clone 5 ({k3.sum()} cells)', {k: cv_cls(v[k3], cd[k3], blocks[k3]) for k, v in R.items()}, f'{1 / len(big):.3f}')
P('\n-- T4 within clone 5: fitness (Spearman) --')
fit = pd.to_numeric(ob.lin_fitness, errors='coerce').to_numpy(); k4 = (cl == '5') & np.isfinite(fit)
report(f'T4 fitness within clone 5 ({k4.sum()} cells)', {k: cv_reg(v[k4], fit[k4], blocks[k4]) for k, v in R.items()}, '0')

P('\n-- T5 within clone 5: lineage character distance vs latent distance --')
ch = np.asarray(a.obsm['lineage_characters'])[m][k3]
miss = np.isin(ch, ['-1', 'nan', '', 'None', '-1.0'])
rng = np.random.default_rng(0); pick = rng.choice(k3.sum(), min(800, k3.sum()), replace=False)
iu = np.triu_indices(len(pick), 1)
chp, mp = ch[pick], miss[pick]
both = (~mp[:, None, :]) & (~mp[None, :, :]); diff = (chp[:, None, :] != chp[None, :, :]) & both
cdist = diff.sum(-1) / np.maximum(both.sum(-1), 1); valid = both.sum(-1)[iu] >= 5
sdist = np.linalg.norm(X_[k3][pick][:, None] - X_[k3][pick][None], axis=-1)[iu][valid]
lin = cdist[iu][valid]
def partial(x, y, zc):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(zc)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]
P(f'   pairs {valid.sum()} | corr(lineage distance, spatial distance) {spearmanr(lin, sdist).statistic:+.3f}')
vals, pvals = {}, {}
for k, v in R.items():
    V = StandardScaler().fit_transform(v[k3][pick]); ld = np.linalg.norm(V[:, None] - V[None], axis=-1)[iu][valid]
    vals[k] = spearmanr(ld, lin).statistic; pvals[k] = partial(ld, lin, sdist)
report('T5 latent vs lineage distance (Spearman)', vals, '0'); report('T5 same, partial on spatial distance', pvals, '0')
pd.DataFrame(res).to_csv(D / 'lineage_test_summary.csv', index=False)
P('done')
