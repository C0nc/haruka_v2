#!/usr/bin/env python3
"""Exercise cells only: do z or s carry the OBSERVED clock-age shift?

Ground truth per exercise cell: shift = observed clock age (frozen Sun et al. clock on observed expression)
  - control baseline for the same cell type x subregion (mean of the 4 control mice's means).
Per cell type, exercise cells only, 4 folds = one exercise mouse held out: ridge from z or s (standardised on
training mice) -> shift. Readouts on held-out mice:
  cell     Spearman(predicted, observed shift) within the held-out mouse
  region   within each cell type, corr across subregions of mean predicted vs mean observed shift (intercept-free)
  mouse    within each cell type, corr across the 4 held-out mice of mean predicted vs observed shift
Figures: UMAPs of z and s (exercise cells) coloured by the observed shift; tissue map of exercise mouse OE3
(observed shift, held-out from z, held-out from s; vascular cells); cell type x region heatmaps.
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import umap
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup()
OUT = Path('/home/yancui/haruka_v2/figures_paper/exercise/exercise_only'); OUT.mkdir(exist_ok=True)
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING); SZ = 46 * ps.MM
VAS = ['Endothelial', 'Pericyte', 'VLMC', 'VSMC']
log = open(OUT / 'exercise_only.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str); assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == names).all()
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
age = C.predicted_age.to_numpy(); ct = o.celltype.astype(str).to_numpy(); sub = o.subregion.astype(str).to_numpy()
mouse = o.mouse_id.astype(str).to_numpy(); ex = (o.condition.astype(str) == 'exercise').to_numpy()
xy = np.asarray(h.obsm['spatial'], float)[:, :2]
Z = np.load(MD / 'background.npy').astype(np.float64); S = np.load(MD / 'salient.npy').astype(np.float64)
ok = np.isfinite(age)

# control baseline per cell type x subregion = mean over control mice of the mouse mean
cb = pd.DataFrame({'ct': ct, 'sub': sub, 'mouse': mouse, 'age': age})[ok & ~ex].groupby(['ct', 'sub', 'mouse']).age.mean() \
       .groupby(['ct', 'sub']).mean()
base = pd.Series(list(zip(ct, sub))).map(cb.to_dict()).to_numpy(dtype=float)
shift = age - base
E = np.flatnonzero(ok & ex & np.isfinite(shift))
exm = sorted(set(mouse[E]))
pred = {k: np.full(len(ct), np.nan) for k in ['z', 's']}
for c in sorted(set(ct[E])):
    m = E[ct[E] == c]
    if len(m) < 1000: continue
    for hm in exm:
        te, tr = m[mouse[m] == hm], m[mouse[m] != hm]
        for k, M in [('z', Z), ('s', S)]:
            sc = StandardScaler().fit(M[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(M[tr]), shift[tr])
            pred[k][te] = r.predict(sc.transform(M[te]))
F = pd.DataFrame({'cell_id': names, 'cell_type': ct, 'subregion': sub, 'mouse': mouse, 'observed_shift': shift,
                  'z': pred['z'], 's': pred['s']}).iloc[E].dropna()
F.to_csv(OUT / 'exercise_cells_shift_predictions.csv.gz', index=False)

rows = []
for c, g in F.groupby('cell_type'):
    cell = {k: np.mean([spearmanr(gm[k], gm.observed_shift).statistic for _, gm in g.groupby('mouse')]) for k in ['z', 's']}
    R = g.groupby('subregion').filter(lambda x: len(x) >= 200).groupby('subregion')[['observed_shift', 'z', 's']].mean()
    Mo = g.groupby('mouse')[['observed_shift', 'z', 's']].mean()
    rows.append(dict(cell_type=c, n=len(g), obs_mean_shift=g.observed_shift.mean(), cell_z=cell['z'], cell_s=cell['s'],
                     region_z=pearsonr(R.observed_shift, R.z)[0] if len(R) >= 4 else np.nan,
                     region_s=pearsonr(R.observed_shift, R.s)[0] if len(R) >= 4 else np.nan,
                     mouse_z=pearsonr(Mo.observed_shift, Mo.z)[0], mouse_s=pearsonr(Mo.observed_shift, Mo.s)[0]))
T = pd.DataFrame(rows).sort_values('obs_mean_shift'); T.to_csv(OUT / 'exercise_only_summary.csv', index=False)
pd.set_option('display.width', 200)
P('== exercise cells only: held-out (one exercise mouse out) prediction of the observed clock shift ==')
P(T.round(3).to_string(index=False))
for lev in ['cell', 'region', 'mouse']:
    t = T[[f'{lev}_z', f'{lev}_s']].dropna()
    P(f'   {lev:6s}: median z {t[f"{lev}_z"].median():+.3f}  s {t[f"{lev}_s"].median():+.3f} | s > z in {int((t[f"{lev}_s"] > t[f"{lev}_z"]).sum())}/{len(t)}'
      + f' | vascular s > z {int((t.loc[T.cell_type.isin(VAS), f"{lev}_s"] > t.loc[T.cell_type.isin(VAS), f"{lev}_z"]).sum())}/{int(T.cell_type.isin(VAS).sum())}')

# ---------- figures ----------
def panel(title, letter, col=None, w=SZ):
    fig, ax = plt.subplots(figsize=(w, SZ)); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(title, pad=2, color=col or ps.INK, fontweight='bold' if col else 'normal'); ps.panel_label(ax, letter, x=-0.06, y=1.06)
    return fig, ax
def cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08); cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
rng = np.random.default_rng(0)
Fi = pd.Index(names).get_indexer(F.cell_id)
for nm, mask, L0 in [('all', np.ones(len(F), bool), 'a'), ('vascular', F.cell_type.isin(VAS).to_numpy(), 'c')]:
    idx = rng.choice(np.flatnonzero(mask), min(40000, mask.sum()), replace=False); gi = Fi[idx]; y = F.observed_shift.to_numpy()[idx]
    lim = np.nanpercentile(np.abs(y), 97)
    for k, M, L in [('z', Z, L0), ('s', S, chr(ord(L0) + 1))]:
        Em = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(M[gi]); pm = rng.permutation(len(idx))
        fig, ax = panel(f'{k}: observed clock shift ({nm}, exercise cells)', L, ps.LATENT[k])
        im = ax.scatter(Em[pm, 0], Em[pm, 1], s=0.4, c=y[pm], cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), lw=0, rasterized=True)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2'); cbar(fig, ax, im, 'exercise − control clock age (months)')
        ps.save(fig, OUT / f'exo_{L}_{k}_umap_{nm}_shift'); plt.close(fig)

# tissue: exercise mouse OE3, vascular cells
mm = mouse == 'OE3'; vs = np.isin(F.cell_id, names[mm]) & F.cell_type.isin(VAS).to_numpy()
Q = F[vs]; qi = pd.Index(names).get_indexer(Q.cell_id)
for k, L, title, col in [('observed_shift', 'e', 'OE3 vascular: observed clock shift', None), ('z', 'f', 'z: held-out prediction', ps.LATENT['z']),
                         ('s', 'g', 's: held-out prediction', ps.LATENT['s'])]:
    v = Q[k].to_numpy(); v = (v - v.mean()) / v.std()
    fig, ax = panel(title, L, col); ax.set_aspect('equal')
    ax.scatter(xy[mm, 0], xy[mm, 1], s=0.1, color=ps.NEUTRAL['light'], lw=0, rasterized=True)
    im = ax.scatter(xy[qi, 0], xy[qi, 1], s=0.8, c=v, cmap=DIV, norm=TwoSlopeNorm(0, -2, 2), lw=0, rasterized=True)
    x0, y0 = xy[mm, 0].min(), xy[mm, 1].min(); ax.plot([x0, x0 + 1000], [y0 - 200] * 2, color=ps.INK, lw=1)
    ax.text(x0 + 500, y0 - 260, '1 mm', ha='center', va='top', fontsize=6)
    if k != 'observed_shift':
        ax.text(0.5, -0.02, f'cell-level ρ with observed {spearmanr(Q[k], Q.observed_shift).statistic:+.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=6)
    cbar(fig, ax, im, 'within-section shift (SD; blue = younger)'); ps.save(fig, OUT / f'exo_{L}_tissue_{k}'); plt.close(fig)

# heatmaps cell type x region (held-out mice), each readout centred within cell type to show the region pattern
REG = ['CTX_L1/MEN', 'CTX_L2/3', 'CTX_L4/5/6', 'STR_CP/ACB', 'STR_LS/NDB', 'CC/ACO', 'VEN']
G = F.groupby(['cell_type', 'subregion']).filter(lambda x: len(x) >= 200).groupby(['cell_type', 'subregion'])[['observed_shift', 'z', 's']].mean().reset_index()
order = T.cell_type.tolist()
for k, L, title in [('observed_shift', 'h', 'Observed clock shift'), ('z', 'i', 'Held-out from z'), ('s', 'j', 'Held-out from s')]:
    Mx = G.pivot(index='cell_type', columns='subregion', values=k).reindex(index=order, columns=REG)
    lim = np.nanmax(np.abs(Mx.to_numpy()))
    fig, ax = plt.subplots(figsize=(56 * ps.MM, 60 * ps.MM))
    im = ax.imshow(Mx.to_numpy(), cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto'); ax.set_facecolor(ps.NEUTRAL['light'])
    ax.set_xticks(range(len(REG))); ax.set_xticklabels([r.replace('_', ' ') for r in REG], rotation=55, ha='right')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([x.replace('Neuron-', 'Neuron ') for x in order]); ax.tick_params(length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label('exercise − control\nclock age (months)', fontsize=6); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
    ax.set_title(title, fontsize=6.5, color=ps.LATENT.get(k, ps.INK)); ps.panel_label(ax, L, x=-0.62)
    ps.save(fig, OUT / f'exo_{L}_heat_{k}'); plt.close(fig)

# summary bars: cell / region / mouse level, z vs s, per cell type
fig, axes = plt.subplots(1, 3, figsize=(150 * ps.MM, 44 * ps.MM))
for ax, lev, title in zip(axes, ['cell', 'region', 'mouse'], ['cell level (within mouse)', 'region pattern (within cell type)', 'mouse level (4 held-out mice)']):
    x = np.arange(len(T)); ax.bar(x - 0.2, T[f'{lev}_z'], 0.4, color=ps.LATENT['z'], label='z'); ax.bar(x + 0.2, T[f'{lev}_s'], 0.4, color=ps.LATENT['s'], label='s')
    ax.axhline(0, color=ps.INK, lw=0.5); ax.set_xticks(x); ax.set_xticklabels([c.replace('Neuron-', 'N-') for c in T.cell_type], rotation=60, ha='right', fontsize=5.8)
    ax.set_title(title); ax.set_ylabel('corr with observed shift')
axes[0].legend(fontsize=6); ps.panel_label(axes[0], 'k', x=-0.3); fig.tight_layout(); ps.save(fig, OUT / 'exo_k_summary'); plt.close(fig)
P('font failures:', [f.name for f in sorted(OUT.glob('exo_*.svg')) if not ps.check_fonts(f)[0]])
P('done')
