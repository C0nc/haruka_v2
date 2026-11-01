#!/usr/bin/env python3
"""OSKM: does s or z capture the published Spatial Aging Clock shift, WITHIN each cell type?
(same protocol as the AD plaque-distance test)

Ground truth: clock age predicted by the frozen Sun et al. clock from the OBSERVED expression
(final_results_agent_oskm/.../clock_full/spatial_clock_predictions.csv.gz), both arms.
For each cell type separately: X (z or s) and clock age centred within pair x subregion (pooling arms,
so the OSKM-vs-control difference stays in), ridge (RidgeCV) trained on 3 pairs, scored on the held-out pair.
Units = held-out pair x cell type: Spearman(held-out score, clock age); s > z count over units.
Shift: per unit, held-out predicted OSKM - control difference vs the observed one (sign agreement, corr).
usage: clock_zs_by_celltype.py <model_dir> <tag>
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = '/home/yancui/haruka_v2'; MD, TAG = sys.argv[1], sys.argv[2]; OUT = f'{H}/bench_oskm_reprogramming/zs_rejuvenation'
log = open(f'{OUT}/clock_zs_by_celltype_{TAG}.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
names = pd.Index(np.load(f'{MD}/obs_names.npy', allow_pickle=True).astype(str))
C = pd.read_csv(f'{H}/final_results_agent_oskm/spatial_aging_clock_extension/clock_full/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
Z = np.load(f'{MD}/background.npy').astype(np.float64); S = np.load(f'{MD}/salient.npy').astype(np.float64)
age = C.predicted_age.to_numpy(); ct = C.cell_type.astype(str).to_numpy(); pair = C.paired_batch.astype(str).to_numpy()
sub = C.subregion.astype(str).to_numpy(); oskm = (C.cohort.astype(str) == 'old_treatment').to_numpy()
ok = np.isfinite(age); P(f'{TAG}: {ok.sum()} cells, OSKM {oskm[ok].mean():.1%}')

def centre(X, g):
    X = X.copy(); df = pd.DataFrame(X); return X - df.groupby(g).transform('mean').to_numpy()

rows = []
for c in sorted(set(ct)):
    m = ok & (ct == c)
    if m.sum() < 2000: continue
    g = pd.Series(pair[m]).str.cat(sub[m], sep='|').to_numpy()
    y = centre(age[m][:, None], g).ravel(); pr = pair[m]; arm = oskm[m]
    X = {'z': centre(Z[m], g), 's': centre(S[m], g)}
    for p in np.unique(pr):
        tr, te = pr != p, pr == p
        if te.sum() < 100 or arm[te].sum() < 30 or (~arm[te]).sum() < 30: continue
        rec = dict(cell_type=c, pair=p, n=int(te.sum()), observed_shift=y[te][arm[te]].mean() - y[te][~arm[te]].mean())
        for k, Xk in X.items():
            sc = StandardScaler().fit(Xk[tr]); mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(Xk[tr]), y[tr])
            yh = mdl.predict(sc.transform(Xk[te]))
            rec[f'rho_{k}'] = spearmanr(yh, y[te]).statistic
            rec[f'shift_{k}'] = yh[arm[te]].mean() - yh[~arm[te]].mean()
        rows.append(rec)
T = pd.DataFrame(rows); T['s_minus_z'] = T.rho_s - T.rho_z; T.to_csv(f'{OUT}/clock_zs_by_celltype_{TAG}.csv', index=False)
w = wilcoxon(T.rho_s, T.rho_z)
P(f'\n== within-cell-type clock-age capture, {len(T)} held-out pair x cell type units ==')
P(f'   median Spearman z {T.rho_z.median():.3f}  s {T.rho_s.median():.3f} | s > z in {int((T.s_minus_z > 0).sum())}/{len(T)} units '
  f'| median s - z {T.s_minus_z.median():+.3f} | Wilcoxon p {w.pvalue:.2g}')
by = T.groupby('cell_type').agg(units=('pair', 'size'), z=('rho_z', 'median'), s=('rho_s', 'median'),
                                s_wins=('s_minus_z', lambda v: f'{int((v > 0).sum())}/{len(v)}'),
                                obs_shift=('observed_shift', 'median'), z_shift=('shift_z', 'median'), s_shift=('shift_s', 'median'))
P('   by cell type (median over held-out pairs; shifts = OSKM - control, months):'); P('   ' + by.round(3).to_string().replace('\n', '\n   '))
P('\n== does the held-out latent reproduce the OSKM - control clock shift? (per unit) ==')
for k in ['z', 's']:
    agree = np.mean(np.sign(T[f'shift_{k}']) == np.sign(T.observed_shift))
    P(f'   {k}: sign agreement {agree:.0%} | corr(predicted, observed shift) r {pearsonr(T[f"shift_{k}"], T.observed_shift)[0]:+.2f} '
      f'| slope {np.polyfit(T.observed_shift, T[f"shift_{k}"], 1)[0]:+.2f}')
P('   by pair (mean over cell types): ' + T.groupby('pair')[['observed_shift', 'shift_z', 'shift_s']].mean().round(2).to_string().replace('\n', '\n   '))
P('done')
