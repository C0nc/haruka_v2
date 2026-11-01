#!/usr/bin/env python3
"""Exercise rejuvenation by cell type x brain region: observed Spatial Aging Clock shift vs what z and s carry.

Fit: exercise old control (4 mice) vs old exercise (4 mice), batch = run, weights 10 / 100.
Per cell: clock age from the frozen Sun et al. clock on OBSERVED expression. Out-of-fold prediction of clock age
from z or from s, per cell type: ridge trained on 3 control + 3 exercise mice, predicted on the held-out pair
(subregion offsets from training mice only). Every cell gets exactly one held-out prediction.
Stratum = cell type x subregion. Shift = mean over exercise mice - mean over control mice of the per-mouse stratum
mean (mouse is the unit; >= 20 cells per mouse-stratum, >= 3 mice per arm). Welch t on mouse means for observed.
Panels (one SVG each): a observed, b from z, c from s (own colour scales, blue = younger); d predicted vs observed.
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup()
OUT = Path('/home/yancui/haruka_v2/figures_paper/exercise'); OUT.mkdir(exist_ok=True)
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
names = pd.Index(np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str))
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
Z = np.load(MD / 'background.npy').astype(np.float64); S = np.load(MD / 'salient.npy').astype(np.float64)
age = C.predicted_age.to_numpy(); ct = C.celltype.astype(str).to_numpy(); mouse = C.mouse_id.astype(str).to_numpy()
sub = C.subregion.astype(str).to_numpy(); ex = (C.condition.astype(str) == 'exercise').to_numpy(); ok = np.isfinite(age)
folds = list(zip(sorted(set(mouse[~ex])), sorted(set(mouse[ex]))))

def adj(X, g, train):
    X = np.asarray(X, float); out = X.copy()
    for u in np.unique(g):
        m = g == u; out[m] = X[m] - (X[m & train].mean(0) if (m & train).any() else X[train].mean(0))
    return out

cache = OUT / 'oof_clock_predictions_w10_s100.csv.gz'
if cache.exists():
    D = pd.read_csv(cache)
else:
    rec = []
    for c in sorted(set(ct[ok])):
        m = np.flatnonzero(ok & (ct == c))
        if len(m) < 2000: continue
        ms, sb = mouse[m], sub[m]
        for cm, em in folds:
            te = np.isin(ms, [cm, em]); tr = ~te
            y = adj(age[m][:, None], sb, tr).ravel()
            row = dict(idx=m[te], y=y[te])
            for k, M in [('z', Z[m]), ('s', S[m])]:
                X = adj(M, sb, tr); sc = StandardScaler().fit(X[tr])
                row[k] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
            rec.append(pd.DataFrame({'cell_id': names[row['idx']], 'cell_type': c, 'subregion': sb[te], 'mouse': ms[te],
                                     'exercise': ex[m][te], 'observed': row['y'], 'z': row['z'], 's': row['s']}))
    D = pd.concat(rec, ignore_index=True); D.to_csv(cache, index=False)

# mouse-level stratum shifts
M_ = D.groupby(['cell_type', 'subregion', 'mouse', 'exercise']).agg(n=('observed', 'size'), observed=('observed', 'mean'),
                                                                     z=('z', 'mean'), s=('s', 'mean')).reset_index()
M_ = M_[M_.n >= 20]
rows = []
for (c, r), g in M_.groupby(['cell_type', 'subregion']):
    e, k = g[g.exercise], g[~g.exercise]
    if len(e) < 3 or len(k) < 3: continue
    rec = dict(cell_type=c, subregion=r, n_ex=len(e), n_ctrl=len(k), cells=int(g.n.sum()),
               p_obs=ttest_ind(e.observed, k.observed, equal_var=False).pvalue)
    for v in ['observed', 'z', 's']: rec[v] = e[v].mean() - k[v].mean()
    rows.append(rec)
T = pd.DataFrame(rows); T.to_csv(OUT / 'exercise_clock_shift_by_celltype_region.csv', index=False)

REG = ['CTX_L1/MEN', 'CTX_L2/3', 'CTX_L4/5/6', 'STR_CP/ACB', 'STR_LS/NDB', 'CC/ACO', 'VEN']
order = T.groupby('cell_type')['observed'].mean().sort_values().index.tolist()
cmap = LinearSegmentedColormap.from_list('div', ps.DIVERGING)

def heat(col, title, fname, letter):
    G = T.pivot(index='cell_type', columns='subregion', values=col).reindex(index=order, columns=REG)
    lim = np.nanmax(np.abs(G.to_numpy())); fig, ax = plt.subplots(figsize=(58 * ps.MM, 62 * ps.MM))
    im = ax.imshow(G.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    ax.set_facecolor(ps.NEUTRAL['light'])
    ax.set_xticks(range(len(REG))); ax.set_xticklabels([r.replace('_', ' ') for r in REG], rotation=55, ha='right')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o.replace('Neuron-', 'Neuron ') for o in order])
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
    if col == 'observed':
        P = T.pivot(index='cell_type', columns='subregion', values='p_obs').reindex(index=order, columns=REG).to_numpy()
        for i, j in zip(*np.where(P < 0.05)): ax.text(j, i, '*', ha='center', va='center', fontsize=7, color=ps.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label('exercise − control\nclock age (months)', fontsize=6); cb.outline.set_visible(False)
    ax.set_title(title, fontsize=6.5); ps.panel_label(ax, letter, x=-0.62)
    ps.save(fig, OUT / fname); plt.close(fig)

heat('observed', 'Observed clock shift', 'ex_a_observed', 'a')
heat('z', 'Held-out prediction from z', 'ex_b_from_z', 'b')
heat('s', 'Held-out prediction from s', 'ex_c_from_s', 'c')

fig, axes = plt.subplots(1, 2, figsize=(84 * ps.MM, 40 * ps.MM), sharey=False)
for ax, k in zip(axes, ['z', 's']):
    r = pearsonr(T.observed, T[k])[0]; agree = np.mean(np.sign(T[k]) == np.sign(T.observed))
    ax.axhline(0, color=ps.NEUTRAL['mid'], lw=0.5, zorder=0); ax.axvline(0, color=ps.NEUTRAL['mid'], lw=0.5, zorder=0)
    sig = T.p_obs < 0.05
    ax.scatter(T.observed[~sig], T[k][~sig], s=6, color=ps.LATENT[k], alpha=0.55, lw=0)
    ax.scatter(T.observed[sig], T[k][sig], s=12, facecolor=ps.LATENT[k], edgecolor=ps.INK, lw=0.5)
    ax.set_xlabel('observed shift (months)'); ax.set_ylabel(f'predicted from {k} (months)')
    ax.text(0.03, 0.97, f'r = {r:+.2f}\nsign agree {agree:.0%}', transform=ax.transAxes, va='top', fontsize=6)
ps.panel_label(axes[0], 'd', x=-0.35); fig.tight_layout(); ps.save(fig, OUT / 'ex_d_pred_vs_obs'); plt.close(fig)

vas = T.cell_type.isin(['Endothelial', 'Pericyte', 'VLMC', 'VSMC'])
print(f'strata {len(T)}; observed p<0.05: {(T.p_obs < 0.05).sum()}')
for nm, m in [('all', np.ones(len(T), bool)), ('vascular', vas.to_numpy()), ('non-vascular', ~vas.to_numpy())]:
    t = T[m]
    print(f'  {nm:12s} n={len(t):3d} | corr(obs, z) {pearsonr(t.observed, t.z)[0]:+.2f}  corr(obs, s) {pearsonr(t.observed, t.s)[0]:+.2f} | '
          f'sign agree z {np.mean(np.sign(t.z) == np.sign(t.observed)):.0%}  s {np.mean(np.sign(t.s) == np.sign(t.observed)):.0%} | '
          f'mean shift obs {t.observed.mean():+.2f} z {t.z.mean():+.2f} s {t.s.mean():+.2f}')
for f in sorted(OUT.glob('ex_*.svg')):
    okf, sizes = ps.check_fonts(f); print(f.name, 'fonts ok' if okf else f'FONT FAIL {sizes[:3]}')
