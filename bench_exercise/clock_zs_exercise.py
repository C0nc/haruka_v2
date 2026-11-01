#!/usr/bin/env python3
"""Exercise: does z or s capture the published Spatial Aging Clock shift (rejuvenation), WITHIN each cell type?
Same protocol as the OSKM test, adapted to 4 vs 4 mice.
Ground truth: clock age from the frozen Sun et al. clock run on OBSERVED expression.
Step 0 (observed): per cell type, mouse-mean clock age (subregion-adjusted); exercise - control (4 vs 4 mice), Welch t.
Step 1: per cell type, ridge from z or s -> clock age, trained on 3+3 mice, scored on the held-out control+exercise mouse
        (subregion offsets from training mice only). Spearman per held-out pair; predicted vs observed shift per pair.
usage: clock_zs_exercise.py <model_dir> <tag>
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
MD, TAG = sys.argv[1], sys.argv[2]; OUT = '/home/yancui/haruka_v2/bench_exercise'
log = open(f'{OUT}/clock_zs_{TAG}.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
names = pd.Index(np.load(f'{MD}/obs_names.npy', allow_pickle=True).astype(str))
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
Z = np.load(f'{MD}/background.npy').astype(np.float64); S = np.load(f'{MD}/salient.npy').astype(np.float64)
age = C.predicted_age.to_numpy(); ct = C.celltype.astype(str).to_numpy(); mouse = C.mouse_id.astype(str).to_numpy()
sub = C.subregion.astype(str).to_numpy(); ex = (C.condition.astype(str) == 'exercise').to_numpy()
ok = np.isfinite(age)
cm = sorted(set(mouse[~ex])); em = sorted(set(mouse[ex])); folds = list(zip(cm, em))

def adj(X, g, train):
    X = np.asarray(X, float).copy(); out = X.copy()
    for u in np.unique(g):
        m = g == u
        out[m] = X[m] - (X[m & train].mean(0) if (m & train).any() else X[train].mean(0))
    return out

P(f'{TAG}: {ok.sum()} cells with clock age')
P('\n== Step 0: observed clock shift, exercise - control, mouse level (4 vs 4; months, subregion-adjusted) ==')
obs_rows = []
for c in sorted(set(ct[ok])):
    m = ok & (ct == c)
    if m.sum() < 2000: continue
    y = adj(age[m][:, None], sub[m], np.ones(m.sum(), bool)).ravel()
    mm = pd.Series(y).groupby(mouse[m]).mean()
    a_, b_ = mm[[x for x in em if x in mm]], mm[[x for x in cm if x in mm]]
    t = ttest_ind(a_, b_, equal_var=False)
    obs_rows.append(dict(cell_type=c, shift=a_.mean() - b_.mean(), p=t.pvalue,
                         ex_mice=' '.join(f'{v:+.2f}' for v in a_), ctrl_mice=' '.join(f'{v:+.2f}' for v in b_)))
O = pd.DataFrame(obs_rows); P('   ' + O.round(3).to_string(index=False).replace('\n', '\n   '))

P('\n== Step 1: held-out mouse pair x cell type: z vs s -> clock age ==')
rows = []
for c in O.cell_type:
    m = ok & (ct == c); ms = mouse[m]; sb = sub[m]; ar = ex[m]
    for cmo, emo in folds:
        te = np.isin(ms, [cmo, emo]); tr = ~te
        y = adj(age[m][:, None], sb, tr).ravel()
        rec = dict(cell_type=c, pair=f'{cmo}/{emo}', observed_shift=y[te & ar].mean() - y[te & ~ar].mean())
        for k, M in [('z', Z[m]), ('s', S[m])]:
            X = adj(M, sb, tr); sc = StandardScaler().fit(X[tr])
            mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr]); yh = mdl.predict(sc.transform(X[te]))
            rec[f'rho_{k}'] = spearmanr(yh, y[te]).statistic
            rec[f'shift_{k}'] = yh[ar[te]].mean() - yh[~ar[te]].mean()
        rows.append(rec)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/clock_zs_{TAG}.csv', index=False)
P(f'   {len(T)} units | median Spearman z {T.rho_z.median():.3f}  s {T.rho_s.median():.3f} | s > z in {int((T.rho_s > T.rho_z).sum())}/{len(T)}')
by = T.groupby('cell_type').agg(z=('rho_z', 'median'), s=('rho_s', 'median'), obs_shift=('observed_shift', 'median'),
                                z_shift=('shift_z', 'median'), s_shift=('shift_s', 'median'))
P('   ' + by.round(3).to_string().replace('\n', '\n   '))
for k in ['z', 's']:
    P(f'   {k}: shift sign agreement {np.mean(np.sign(T[f"shift_{k}"]) == np.sign(T.observed_shift)):.0%} | '
      f'corr(predicted, observed shift) {pearsonr(T[f"shift_{k}"], T.observed_shift)[0]:+.2f}')
P('done')
