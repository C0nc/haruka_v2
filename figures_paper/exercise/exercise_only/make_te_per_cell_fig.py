#!/usr/bin/env python3
"""Per-cell view of the pooled TE fit (exercise cells; held-out mouse predictions from te_per_cell.py).
a/b  hexbin: held-out predicted TE vs observed TE, all cells, from z / from s
c    per cell type: within-type Spearman, z vs s (each point = cell type, size = n)
d-f  tissue, exercise mouse OE3, all cells: observed TE | predicted from z | predicted from s (within-section SD)
g-i  the same, cells centred within cell type (removes the cell-type offset)
"""
import sys, warnings; import numpy as np, pandas as pd, anndata as ad
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup(); MM = ps.MM
D = pd.read_csv('te_per_cell_predictions.csv.gz')
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r')
xy = pd.DataFrame(np.asarray(h.obsm['spatial'], float)[:, :2], index=np.asarray(h.obs_names).astype(str), columns=['x', 'y'])
D = D.join(xy, on='cell_id')
DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING)
fig, axes = plt.subplots(1, 3, figsize=(150 * MM, 46 * MM))
for ax, k in zip(axes[:2], ['z', 's']):
    cm = LinearSegmentedColormap.from_list('c', ['#FFFFFF', ps.LATENT[k], ps.INK])
    ax.hexbin(D.te, D[f'pred_{k}'], gridsize=60, cmap=cm, bins='log', mincnt=1, extent=(-25, 25, -10, 8), lw=0)
    r = np.mean([spearmanr(g[f'pred_{k}'], g.te).statistic for _, g in D.groupby('mouse')])
    ax.set_xlabel('observed TE (clock − age, months)'); ax.set_ylabel(f'held-out prediction from {k}')
    ax.set_title(f'{k}: per cell, ρ = {r:.2f}', color=ps.LATENT[k], fontweight='bold')
W = []
for c, g in D.groupby('cell_type'):
    if len(g) < 500: continue
    W.append((c, len(g), *[np.mean([spearmanr(gm[f'pred_{k}'], gm.te).statistic for _, gm in g.groupby('mouse')]) for k in ['z', 's']]))
W = pd.DataFrame(W, columns=['ct', 'n', 'z', 's']); ax = axes[2]
ax.plot([0, 0.4], [0, 0.4], color=ps.NEUTRAL['mid'], lw=0.6, ls='--')
ax.scatter(W.z, W.s, s=np.sqrt(W.n) / 6, color=ps.LATENT['s'], edgecolor=ps.INK, lw=0.4)
for r in W.itertuples(): ax.text(r.z + 0.005, r.s + 0.004, r.ct.replace('Neuron-', 'N-'), fontsize=5.8)
ax.set_xlabel('within-type ρ, from z'); ax.set_ylabel('within-type ρ, from s'); ax.set_title(f'per cell type: s > z in {int((W.s > W.z).sum())}/{len(W)}')
ps.panel_label(axes[0], 'a', x=-0.3); ps.panel_label(axes[1], 'b', x=-0.3); ps.panel_label(axes[2], 'c', x=-0.3)
fig.tight_layout(); ps.save(fig, 'pc_abc_scatter'); plt.close(fig)

Q = D[D.mouse == 'OE3'].copy()
for k in ['te', 'pred_z', 'pred_s']: Q[f'{k}_c'] = Q[k] - Q.groupby('cell_type')[k].transform('mean')
fig, axes = plt.subplots(2, 3, figsize=(150 * MM, 100 * MM))
for row, suf, lab in [(0, '', 'raw'), (1, '_c', 'centred within cell type')]:
    for j, (k, title, col) in enumerate([('te', 'observed TE', ps.INK), ('pred_z', 'from z', ps.LATENT['z']), ('pred_s', 'from s', ps.LATENT['s'])]):
        ax = axes[row, j]; v = Q[k + suf]; v = (v - v.mean()) / v.std()
        im = ax.scatter(Q.x, Q.y, s=0.08, c=v, cmap=DIV, norm=TwoSlopeNorm(0, -2, 2), lw=0, rasterized=True)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        extra = '' if k == 'te' else f'\nρ with observed {spearmanr(Q[k + suf], Q["te" + suf]).statistic:+.2f}'
        ax.set_title(f'OE3 {title} ({lab}){extra}', color=col, fontsize=6.5)
        ps.panel_label(ax, 'defghi'[row * 3 + j], x=-0.12, y=1.12)
cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04); cb.set_label('per-cell TE (SD within section; blue = younger)', fontsize=6)
cb.ax.tick_params(labelsize=6, length=2)
ps.save(fig, 'pc_defghi_tissue'); plt.close(fig)
print(W.round(3).to_string(index=False))
