#!/usr/bin/env python3
"""PC1 of z and of s vs the per-cell treatment effect TE (= clock-predicted age - own age), exercise cells.
PC1 is computed inside each cell type x region stratum (paper's 4 regions; >= 200 cells) from the standardised latent,
with no use of TE. Its sign is arbitrary, so for display it is oriented to correlate positively with TE (stated on
every panel); the scores use |r|.
  a/b  heatmaps cell type x region: |r(PC1, TE)| from z / from s     c  s - z
  d    example strata: TE against PC1 (binned means +- s.e.m. over 20 PC1 quantile bins), z vs s
  e-g  tissue, exercise mouse OE3: observed TE | z PC1 | s PC1 (each standardised within stratum)
"""
import sys, warnings
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup(); MM = ps.MM
OUT = Path('/home/yancui/haruka_v2/figures_paper/exercise/exercise_only/pc1'); OUT.mkdir(exist_ok=True)
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str)
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
te = C.predicted_age.to_numpy() - o.age.astype(float).to_numpy()
ct = o.celltype.astype(str).to_numpy(); reg = o.region.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy()
xy = np.asarray(h.obsm['spatial'], float)[:, :2]
E = (o.condition.astype(str) == 'exercise').to_numpy() & np.isfinite(te)
L = {'z': np.load(MD / 'background.npy').astype(np.float64), 's': np.load(MD / 'salient.npy').astype(np.float64)}
pc1 = {k: np.full(len(ct), np.nan) for k in L}; teS = np.full(len(ct), np.nan); rows = []
for (c, r), idx in pd.DataFrame({'c': ct[E], 'r': reg[E]}).groupby(['c', 'r']).indices.items():
    idx = np.flatnonzero(E)[idx]
    if len(idx) < 200: continue
    y = te[idx]; rec = dict(cell_type=c, region=r, n=len(idx))
    teS[idx] = (y - y.mean()) / y.std()
    for k, M in L.items():
        p = PCA(1).fit_transform(StandardScaler().fit_transform(M[idx]))[:, 0]
        rr = pearsonr(p, y)[0]; p = p * np.sign(rr)          # orientation for display only
        pc1[k][idx] = (p - p.mean()) / p.std(); rec[f'r_{k}'] = abs(rr)
    rows.append(rec)
T = pd.DataFrame(rows); T['s_minus_z'] = T.r_s - T.r_z; T.to_csv(OUT / 'pc1_by_stratum.csv', index=False)
np.save(OUT / 'pc1_z.npy', pc1['z']); np.save(OUT / 'pc1_s.npy', pc1['s'])

TYPES = ['Neuron-Excitatory', 'Neuron-MSN', 'Neuron-Inhibitory', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Endothelial',
         'Pericyte', 'VSMC', 'VLMC', 'Ependymal', 'Neuroblast', 'Macrophage']
REG = ['CTX', 'STR', 'CC/ACO', 'VEN']
SEQ = LinearSegmentedColormap.from_list('seq', ps.SEQUENTIAL); DIV = LinearSegmentedColormap.from_list('div', ps.DIVERGING)
DZS = LinearSegmentedColormap.from_list('d', [ps.LATENT['z'], '#F4F1ED', ps.LATENT['s']])
fig, axes = plt.subplots(1, 3, figsize=(150 * MM, 62 * MM))
lim = np.nanmax(T[['r_z', 'r_s']].to_numpy())
for ax, col, title, cmap, norm, lab in [(axes[0], 'r_z', 'z PC1: |r| with TE', SEQ, plt.Normalize(0, lim), '|Pearson r|'),
                                         (axes[1], 'r_s', 's PC1: |r| with TE', SEQ, plt.Normalize(0, lim), '|Pearson r|'),
                                         (axes[2], 's_minus_z', 's − z (orange = s better)', DZS, TwoSlopeNorm(0, -0.4, 0.4), 'Δ |r|')]:
    M = T.pivot(index='cell_type', columns='region', values=col).reindex(index=TYPES, columns=REG).to_numpy()
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect='auto')
    for i, j in zip(*np.where(np.isnan(M))): ax.text(j, i, '×', ha='center', va='center', fontsize=6, color=ps.NEUTRAL['mid'])
    ax.set_xticks(range(4)); ax.set_xticklabels(REG, rotation=90); ax.set_yticks(range(len(TYPES)))
    ax.set_yticklabels(TYPES if ax is axes[0] else [''] * len(TYPES)); ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, len(TYPES)), minor=True)
    ax.grid(which='minor', color='white', lw=0.6); ax.tick_params(which='minor', length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.07, pad=0.03); cb.set_label(lab, fontsize=6); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
    ax.set_title(title, fontsize=6.5)
for ax, L_ in zip(axes, 'abc'): ps.panel_label(ax, L_, x=-0.15 if L_ != 'a' else -1.0)
fig.suptitle(f'PC1 inside each cell type × region (no TE used): s > z in {int((T.s_minus_z > 0).sum())}/{len(T)} strata; '
             f'median |r| z {T.r_z.median():.2f}, s {T.r_s.median():.2f}', fontsize=6.5, y=1.02)
fig.tight_layout(); ps.save(fig, OUT / 'pc1_abc_heatmaps'); plt.close(fig)

EX = [('Neuron-Inhibitory', 'CTX'), ('Endothelial', 'CTX'), ('Astrocyte', 'CTX'), ('Microglia', 'STR')]
fig, axes = plt.subplots(1, 4, figsize=(170 * MM, 42 * MM))
for ax, (c, r) in zip(axes, EX):
    m = E & (ct == c) & (reg == r)
    for k in ['z', 's']:
        x = pc1[k][m]; y = te[m]; q = pd.qcut(x, 20, labels=False, duplicates='drop')
        g = pd.DataFrame({'x': x, 'y': y, 'q': q}).groupby('q').agg(x=('x', 'mean'), y=('y', 'mean'), se=('y', 'sem'))
        ax.errorbar(g.x, g.y, yerr=g.se, color=ps.LATENT[k], marker='o', ms=2, lw=0.8, elinewidth=0.5, capsize=0,
                    label=f'{k}: |r| = {T[(T.cell_type == c) & (T.region == r)][f"r_{k}"].iloc[0]:.2f}')
    ax.set_title(f'{c.replace("Neuron-", "Neuron ")}, {r} (n = {m.sum():,})', fontsize=6.5)
    ax.set_xlabel('PC1 (SD; oriented to TE)'); ax.set_ylabel('TE, clock − age (months)'); ax.legend(fontsize=5.8, loc='upper left')
ps.panel_label(axes[0], 'd', x=-0.35); fig.tight_layout(); ps.save(fig, OUT / 'pc1_d_examples'); plt.close(fig)

mm = (mouse == 'OE3') & np.isfinite(pc1['s'])
fig, axes = plt.subplots(1, 3, figsize=(150 * MM, 52 * MM))
for ax, (v, title, col, L_) in zip(axes, [(teS, 'OE3 observed TE', ps.INK, 'e'), (pc1['z'], 'z PC1', ps.LATENT['z'], 'f'),
                                           (pc1['s'], 's PC1', ps.LATENT['s'], 'g')]):
    im = ax.scatter(xy[mm, 0], xy[mm, 1], s=0.08, c=v[mm], cmap=DIV, norm=TwoSlopeNorm(0, -2, 2), lw=0, rasterized=True)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
    t = title if L_ == 'e' else f'{title}\nr with observed {pearsonr(v[mm], teS[mm])[0]:+.2f}'
    ax.set_title(t, color=col, fontsize=6.5, fontweight='bold' if L_ != 'e' else 'normal'); ps.panel_label(ax, L_, x=-0.08, y=1.1)
cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.04, pad=0.04)
cb.set_label('standardised within cell type × region (TE: blue = younger; PC1 oriented to TE)', fontsize=6); cb.ax.tick_params(labelsize=6, length=2)
ps.save(fig, OUT / 'pc1_efg_tissue'); plt.close(fig)
print(T.round(3).to_string(index=False))
print('fonts:', [f.name for f in OUT.glob('*.svg') if not ps.check_fonts(f)[0]])
