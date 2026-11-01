#!/usr/bin/env python3
"""Lung fibrosis (GSE250346), per-cell z vs s, per cell type: do they classify disease stage?
Fit: results_real_compgated_nb_final (control = Unaffected, condition = Less + More Affected; batch key = sample,
which is nested in stage -- see caveat).
Per cell type (>= 3000 cells), logistic (C = 0.1, balanced) on standardised z or s, GroupKFold(5) over DONORS
(a donor's cells never in train and test together):
  A  Unaffected vs fibrosis (both arms)                          cell AUROC; core AUROC (mean score per core)
  B  Less vs More Affected (fibrosis cells only)                  cell AUROC; core AUROC
  C  fibrosis cores: Spearman(mean held-out B score, percent pathology)
Cell types subsampled to <= 60,000 cells.
"""
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage'
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
ct = o.cell_type.astype(str).to_numpy(); donor = o.donor.astype(str).to_numpy(); core = o['sample'].astype(str).to_numpy()
stage = o.sample_affect.astype(str).to_numpy(); path = pd.to_numeric(o.percent_pathology, errors='coerce').to_numpy()
L = {'z': np.load(MD / 'background.npy'), 's': np.load(MD / 'salient.npy')}
log = open(OUT / 'zs_stage.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(0)

def cv(X, y, g):
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2: sc[te] = 0.5; continue
        ss = StandardScaler().fit(X[tr]); sc[te] = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced').fit(ss.transform(X[tr]), y[tr]).decision_function(ss.transform(X[te]))
    return sc
def core_auc(sc, y, c):
    d = pd.DataFrame({'s': sc, 'y': y, 'c': c}).groupby('c').agg(s=('s', 'mean'), y=('y', 'first'))
    return roc_auc_score(d.y, d.s) if d.y.nunique() == 2 else np.nan

rows = []
for c in pd.Series(ct).value_counts().index:
    idx = np.flatnonzero(ct == c)
    if len(idx) < 3000: break
    idx = np.sort(rng.choice(idx, min(60000, len(idx)), replace=False))
    rec = dict(cell_type=c, n=len(idx))
    yA = stage[idx] != 'Unaffected'
    fb = idx[stage[idx] != 'Unaffected']; yB = stage[fb] == 'More Affected'
    if yA.mean() in (0, 1) or len(np.unique(donor[idx])) < 10: continue
    for k, M in L.items():
        sA = cv(M[idx], yA, donor[idx]); rec[f'A_cell_{k}'] = roc_auc_score(yA, sA); rec[f'A_core_{k}'] = core_auc(sA, yA, core[idx])
        if yB.mean() not in (0, 1) and len(fb) > 500:
            sB = cv(M[fb], yB, donor[fb]); rec[f'B_cell_{k}'] = roc_auc_score(yB, sB); rec[f'B_core_{k}'] = core_auc(sB, yB, core[fb])
            d = pd.DataFrame({'s': sB, 'p': path[fb], 'c': core[fb]}).groupby('c').mean()
            rec[f'C_path_{k}'] = spearmanr(d.s, d.p).statistic
    rows.append(rec); P(f'{c}: done')
T = pd.DataFrame(rows); T.to_csv(OUT / 'zs_stage.csv', index=False)
pd.set_option('display.width', 250)
P('\n== per cell type, donor-held-out ==')
P(T.round(3).to_string(index=False))
for m in ['A_cell', 'A_core', 'B_cell', 'B_core', 'C_path']:
    t = T[[f'{m}_z', f'{m}_s']].dropna()
    P(f'   {m:7s}: median z {t[f"{m}_z"].median():.3f}  s {t[f"{m}_s"].median():.3f} | s > z in {int((t[f"{m}_s"] > t[f"{m}_z"]).sum())}/{len(t)} | Wilcoxon p {wilcoxon(t[f"{m}_s"], t[f"{m}_z"]).pvalue:.2g}')
P('done')
