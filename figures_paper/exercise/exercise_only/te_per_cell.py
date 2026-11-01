#!/usr/bin/env python3
"""Exercise cells only, per cell: TE = clock-predicted age - own age (19 mo). ONE ridge over all exercise cells
(no stratification), z or s -> TE, one exercise mouse held out at a time. Scored on held-out mice:
  pooled     Spearman over all held-out cells
  type-only  the same score achieved by the cell-type mean TE alone (training mice) = how much is cell identity
  within     Spearman inside each cell type (median over types), using the same pooled predictions
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str)
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
te = C.predicted_age.to_numpy() - o.age.astype(float).to_numpy()
ct = o.celltype.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy()
E = np.flatnonzero((o.condition.astype(str) == 'exercise').to_numpy() & np.isfinite(te))
Z = np.load(MD / 'background.npy')[E].astype(np.float64); S = np.load(MD / 'salient.npy')[E].astype(np.float64)
y = te[E]; ms = mouse[E]; cts = ct[E]
pred = {'z': np.zeros(len(E)), 's': np.zeros(len(E)), 'type': np.zeros(len(E))}
for hm in sorted(set(ms)):
    te_, tr = ms == hm, ms != hm
    for k, M in [('z', Z), ('s', S)]:
        sc = StandardScaler().fit(M[tr]); pred[k][te_] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(M[tr]), y[tr]).predict(sc.transform(M[te_]))
    mu = pd.Series(y[tr]).groupby(cts[tr]).mean(); pred['type'][te_] = pd.Series(cts[te_]).map(mu).fillna(y[tr].mean()).to_numpy()
out = []
for hm in sorted(set(ms)):
    m = ms == hm
    out.append({k: spearmanr(pred[k][m], y[m]).statistic for k in pred} | {'mouse': hm})
R = pd.DataFrame(out).set_index('mouse')
print(f'exercise cells: {len(E)}; pooled per-cell held-out Spearman with TE (per held-out mouse):')
print(R.round(3).to_string()); print('mean', R.mean().round(3).to_dict())
W = []
for c in sorted(set(cts)):
    m = cts == c
    if m.sum() < 500: continue
    W.append(dict(cell_type=c, n=int(m.sum()), z=np.mean([spearmanr(pred['z'][m & (ms == hm)], y[m & (ms == hm)]).statistic for hm in sorted(set(ms))]),
                  s=np.mean([spearmanr(pred['s'][m & (ms == hm)], y[m & (ms == hm)]).statistic for hm in sorted(set(ms))])))
W = pd.DataFrame(W); print('\nwithin each cell type (same pooled predictions):'); print(W.round(3).to_string(index=False))
print(f'median within-type z {W.z.median():.3f}  s {W.s.median():.3f} | s > z in {int((W.s > W.z).sum())}/{len(W)}')
pd.DataFrame({'cell_id': names[E], 'cell_type': cts, 'mouse': ms, 'te': y, 'pred_z': pred['z'], 'pred_s': pred['s']}).to_csv('te_per_cell_predictions.csv.gz', index=False)
