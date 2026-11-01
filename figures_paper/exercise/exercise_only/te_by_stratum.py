#!/usr/bin/env python3
"""Exercise cells only: per-cell treatment effect TE = clock-predicted age - own chronological age (19 months).
Fitted SEPARATELY in every cell type x region stratum (paper's 4 regions: CTX, STR, CC/ACO, VEN):
ridge from z or from s -> TE, one exercise mouse held out at a time (4 folds; strata need >= 30 cells in >= 3 mice).
Score per stratum = mean over held-out mice of Spearman(predicted TE, observed TE) and of out-of-sample R^2.
(The constant 19 months and the stratum mean drop out of a within-stratum fit, so this measures how well each
embedding ranks cells by their clock age inside the stratum.)
Panels: a observed mean TE (as in the paper), b Spearman from z, c from s, d s - z.
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup()
OUT = Path('/home/yancui/haruka_v2/figures_paper/exercise/exercise_only')
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str); assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == names).all()
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
te = C.predicted_age.to_numpy() - o.age.astype(float).to_numpy()
ct = o.celltype.astype(str).to_numpy(); reg = o.region.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy()
ex = (o.condition.astype(str) == 'exercise').to_numpy()
Z = np.load(MD / 'background.npy').astype(np.float64); S = np.load(MD / 'salient.npy').astype(np.float64)
E = ex & np.isfinite(te)
log = open(OUT / 'te_by_stratum.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

rows = []
for (c, r), idx in pd.DataFrame({'c': ct[E], 'r': reg[E]}).groupby(['c', 'r']).indices.items():
    idx = np.flatnonzero(E)[idx]; ms = mouse[idx]
    cnt = pd.Series(ms).value_counts(); good = cnt[cnt >= 30].index
    if len(good) < 3: continue
    rec = dict(cell_type=c, region=r, n=len(idx), mice=len(good), te_mean=te[idx].mean())
    for k, M in [('z', Z), ('s', S)]:
        rho, r2 = [], []
        for hm in good:
            tr, tst = idx[ms != hm], idx[ms == hm]
            sc = StandardScaler().fit(M[tr]); mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(M[tr]), te[tr])
            p = mdl.predict(sc.transform(M[tst])); y = te[tst]
            rho.append(spearmanr(p, y).statistic); r2.append(1 - ((y - p - (y.mean() - p.mean())) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        rec[f'rho_{k}'] = np.mean(rho); rec[f'r2_{k}'] = np.mean(r2)
        rec[f'mice_s_better_{k}'] = rho
    rec['s_minus_z'] = rec['rho_s'] - rec['rho_z']
    rec['mice_s_gt_z'] = int(np.sum(np.array(rec['mice_s_better_s']) > np.array(rec['mice_s_better_z'])))
    rows.append(rec)
T = pd.DataFrame(rows).drop(columns=['mice_s_better_z', 'mice_s_better_s']); T.to_csv(OUT / 'te_by_stratum.csv', index=False)
w = wilcoxon(T.rho_s, T.rho_z)
pd.set_option('display.width', 220)
P(f'== TE = clock age - own age, exercise cells, fitted per cell type x region ({len(T)} strata) ==')
P(f'   median Spearman z {T.rho_z.median():.3f}  s {T.rho_s.median():.3f} | s > z in {int((T.s_minus_z > 0).sum())}/{len(T)} strata | Wilcoxon p {w.pvalue:.2g}')
P(f'   median within-stratum R^2 z {T.r2_z.median():.3f}  s {T.r2_s.median():.3f}')
vas = T.cell_type.isin(['Endothelial', 'Pericyte', 'VLMC', 'VSMC'])
P(f'   vascular strata: s > z in {int((T[vas].s_minus_z > 0).sum())}/{int(vas.sum())} | median s - z {T[vas].s_minus_z.median():+.3f}')
P(T.sort_values(['cell_type', 'region']).round(3).to_string(index=False))

TYPES = ['Neuron-Excitatory', 'Neuron-MSN', 'Neuron-Inhibitory', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Endothelial',
         'Pericyte', 'VSMC', 'VLMC', 'Ependymal', 'Neuroblast', 'NSC', 'Macrophage']
REG = ['CTX', 'STR', 'CC/ACO', 'VEN']
GREENPINK = LinearSegmentedColormap.from_list('gp', ['#3E6B48', '#F4F1ED', '#9C4A74'])
SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL)
def heat(col, title, fname, letter, cmap, center=None, lim=None, label=''):
    M = T.pivot(index='cell_type', columns='region', values=col).reindex(index=TYPES, columns=REG)
    fig, ax = plt.subplots(figsize=(40 * ps.MM, 62 * ps.MM))
    arr = M.to_numpy(); lim = lim or np.nanmax(np.abs(arr))
    norm = TwoSlopeNorm(0, -lim, lim) if center == 0 else plt.Normalize(0, lim)
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect='auto')
    for i, j in zip(*np.where(np.isnan(arr))): ax.text(j, i, '×', ha='center', va='center', fontsize=6, color=ps.NEUTRAL['mid'])
    ax.set_xticks(range(4)); ax.set_xticklabels(REG, rotation=90); ax.set_yticks(range(len(TYPES))); ax.set_yticklabels(TYPES)
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, len(TYPES)), minor=True)
    ax.grid(which='minor', color='white', lw=0.6); ax.tick_params(which='both', length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04); cb.set_label(label, fontsize=6); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
    ax.set_title(title, fontsize=6.5); ps.panel_label(ax, letter, x=-1.15)
    ps.save(fig, OUT / fname); plt.close(fig)
heat('te_mean', 'Exercise: TE (clock − own age)', 'te_a_observed', 'a', GREENPINK, 0, None, 'mean TE (months)')
lim = np.nanmax(T[['rho_z', 'rho_s']].to_numpy())
heat('rho_z', 'z: held-out ρ with TE', 'te_b_z', 'b', SEQ, None, lim, 'Spearman (held-out mouse)')
heat('rho_s', 's: held-out ρ with TE', 'te_c_s', 'c', SEQ, None, lim, 'Spearman (held-out mouse)')
DIVc = LinearSegmentedColormap.from_list('d', [ps.LATENT['z'], '#F4F1ED', ps.LATENT['s']])
heat('s_minus_z', 's − z (orange = s better)', 'te_d_s_minus_z', 'd', DIVc, 0, None, 'Δ Spearman')
P('font failures:', [f.name for f in sorted(OUT.glob('te_*.svg')) if not ps.check_fonts(f)[0]])
