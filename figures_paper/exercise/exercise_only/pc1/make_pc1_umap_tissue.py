#!/usr/bin/env python3
"""Exercise cells: z and s UMAPs coloured by TE (clock - own age), brain region and cell type; plus a paper-style
tissue map of the stratum where s PC1 tracks TE best (inhibitory neurons, cortex): only that stratum's cells are
coloured, the rest of the section is grey. PC1 from make_pc1_fig.py (inside the stratum, no TE used; oriented to TE).
UMAP: 15 neighbours, min_dist 0.3, random_state 0, 40,000 exercise cells.
"""
import sys, warnings
import numpy as np, pandas as pd, anndata as ad, umap
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/figures_paper'); import paperstyle as ps
ps.setup(); MM = ps.MM; SZ = 46 * MM
OUT = Path('/home/yancui/haruka_v2/figures_paper/exercise/exercise_only/pc1')
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str)
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
te = C.predicted_age.to_numpy() - o.age.astype(float).to_numpy()
ct = o.celltype.astype(str).to_numpy(); reg = o.region.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy()
xy = np.asarray(h.obsm['spatial'], float)[:, :2]
E = np.flatnonzero((o.condition.astype(str) == 'exercise').to_numpy() & np.isfinite(te))
Z = np.load(MD / 'background.npy'); S = np.load(MD / 'salient.npy')
pc1 = {'z': np.load(OUT / 'pc1_z.npy'), 's': np.load(OUT / 'pc1_s.npy')}
VAS = ['Endothelial', 'Pericyte', 'VLMC', 'VSMC']
GROUP = {'Neuron-Inhibitory': 'Neuron', 'Neuron-Excitatory': 'Neuron', 'Neuron-MSN': 'Neuron', 'Neuroblast': 'Neuron',
         'Astrocyte': 'Astrocyte', 'Oligodendrocyte': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Microglia': 'Microglia',
         'Macrophage': 'Microglia', **{v: 'Vascular' for v in VAS}}
GCOL = {**ps.MAJOR, 'Other': ps.NEUTRAL['mid']}
RCOL = {'CTX': ps.REGION['CTX_L2/3'], 'STR': ps.REGION['STR_CP/ACB'], 'CC/ACO': ps.REGION['CC/ACO'], 'VEN': ps.REGION['VEN']}
GP = LinearSegmentedColormap.from_list('gp', ['#3E6B48', '#F4F1ED', '#9C4A74'])   # green = younger, as in the paper

def ax_off(ax):
    ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
def legend(ax, cols, ncol):
    hs = [plt.Line2D([], [], marker='o', ls='', ms=3, color=v, label=k) for k, v in cols.items()]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=ncol, handletextpad=0.1, columnspacing=0.6, fontsize=5.8)

SKIP_UMAP = len(sys.argv) > 1
rng = np.random.default_rng(0); idx = np.sort(rng.choice(E, 40000, replace=False)); pm = rng.permutation(len(idx))
if not SKIP_UMAP:
  EM = {k: umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(M[idx]) for k, M in [('z', Z), ('s', S)]}
if not SKIP_UMAP:
  lim = np.nanpercentile(np.abs(te[idx]), 97)
  fig, axes = plt.subplots(2, 3, figsize=(3 * SZ, 2 * SZ + 14 * MM))
  for i, k in enumerate(['z', 's']):
      e = EM[k][pm]
      ax = axes[i, 0]; im = ax.scatter(e[:, 0], e[:, 1], s=0.3, c=te[idx][pm], cmap=GP, norm=TwoSlopeNorm(0, -lim, lim), lw=0, rasterized=True)
      ax.set_title(f'{k}: TE (clock − age)', color=ps.LATENT[k], fontweight='bold'); ax_off(ax); ax.set_ylabel('UMAP 2')
      if i == 1:
          cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04); cb.set_label('TE (months; green = younger)', fontsize=6); cb.ax.tick_params(labelsize=6, length=2)
      ax = axes[i, 1]; ax.scatter(e[:, 0], e[:, 1], s=0.3, c=[RCOL[r] for r in reg[idx][pm]], lw=0, rasterized=True)
      ax.set_title(f'{k}: brain region', color=ps.LATENT[k], fontweight='bold'); ax_off(ax)
      if i == 1: legend(ax, RCOL, 4)
      ax = axes[i, 2]; ax.scatter(e[:, 0], e[:, 1], s=0.3, c=[GCOL[GROUP.get(c, 'Other')] for c in ct[idx][pm]], lw=0, rasterized=True)
      ax.set_title(f'{k}: cell type', color=ps.LATENT[k], fontweight='bold'); ax_off(ax)
      if i == 1: legend(ax, GCOL, 3)
  for a_, L_ in zip(axes.ravel(), 'efghij'): ps.panel_label(a_, L_, x=-0.05, y=1.04)
  fig.tight_layout(); ps.save(fig, OUT / 'pc1u_abcdef_umaps'); plt.close(fig)


# paper-style tissue: best s stratum, best mouse for it
T = pd.read_csv(OUT / 'pc1_by_stratum.csv'); best = T[T.n >= 2000].sort_values('s_minus_z', ascending=False).iloc[0]
c, r = best.cell_type, best.region
m_all = (ct == c) & (reg == r) & np.isin(np.arange(len(ct)), E)
per = {mm: pearsonr(pc1['s'][m_all & (mouse == mm)], te[m_all & (mouse == mm)])[0] for mm in sorted(set(mouse[E]))}
mo = max(per, key=per.get); sec = mouse == mo; sel = m_all & sec
print(f'best s stratum: {c} {r} (|r| s {best.r_s:.2f} vs z {best.r_z:.2f}); per-mouse r(s PC1, TE) {per}; shown {mo}')
fig, axes = plt.subplots(1, 3, figsize=(3 * SZ, SZ + 10 * MM))
for ax, (v, title, col, L_) in zip(axes, [(te, f'observed TE', ps.INK, 'k'), (pc1['s'], 's PC1', ps.LATENT['s'], 'l'), (pc1['z'], 'z PC1', ps.LATENT['z'], 'm')]):
    ax.scatter(xy[sec, 0], xy[sec, 1], s=0.06, color='#DCD8D3', lw=0, rasterized=True)
    vv = v[sel]
    if L_ == 'k': norm = TwoSlopeNorm(0, -12, 12)
    else: vv = vv; norm = TwoSlopeNorm(0, -2.5, 2.5)
    im = ax.scatter(xy[sel, 0], xy[sel, 1], s=0.6, c=vv, cmap=GP, norm=norm, lw=0, rasterized=True)
    ax.set_aspect('equal'); ax_off(ax)
    sub = '' if L_ == 'k' else f'\nr with TE {pearsonr(v[sel], te[sel])[0]:+.2f}'
    ax.set_title(f'{c.replace("Neuron-", "").lower() + " neurons" if c.startswith("Neuron") else c}, {r} ({mo})\n{title}{sub}', color=col, fontsize=6.5, fontweight='normal' if L_ == 'k' else 'bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label('TE (months)' if L_ == 'k' else 'PC1 (SD; oriented to TE)', fontsize=6)
    ps.panel_label(ax, L_, x=-0.05, y=1.1)
fig.tight_layout(); ps.save(fig, OUT / 'pc1u_ghi_tissue_best'); plt.close(fig)
print('fonts:', [f.name for f in OUT.glob('pc1u_*.svg') if not ps.check_fonts(f)[0]])
