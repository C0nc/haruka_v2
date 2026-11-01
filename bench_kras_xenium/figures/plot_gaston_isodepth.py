#!/usr/bin/env python3
"""Plot GASTON isodepth on the KRAS salient-CC embedding.

Domains are level sets of a scalar field, i.e. ORDINAL, so both isodepth and
domain use a sequential single-hue ramp (not categorical hues).
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr

D = Path('/home/yancui/haruka_v2/bench_kras_xenium/gaston_scc')
OUT = Path('/home/yancui/haruka_v2/bench_kras_xenium/figures')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8a86', '#e3e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})

cells = pd.read_csv(D / 'gaston_isodepth_cells.csv.gz')
# prefer the properly DP-segmented labels over the quantile fallback
import numpy as _np
for _f in sorted(D.glob('gaston_JB_*_resegmented.npz')):
    _d = _np.load(_f, allow_pickle=True)
    _m = pd.Series(_d['labels'], index=_d['cell_id'].astype(str))
    _sel = cells.cell_id.astype(str).isin(_m.index)
    cells.loc[_sel, 'domain'] = _m.reindex(cells.loc[_sel, 'cell_id'].astype(str)).to_numpy()
print('using DP-resegmented labels; domains per section:',
      cells.groupby('slice_id', observed=True)['domain'].nunique().to_dict())
summ = json.loads((D / 'gaston_summary.json').read_text())
secs = sorted(cells.slice_id.astype(str).unique())
n = len(secs)
fig = plt.figure(figsize=(13.2, 3.05 * n))
gs = gridspec.GridSpec(n, 4, wspace=0.30, hspace=0.42, left=0.05, right=0.97,
                       top=0.955, bottom=0.06)

for r, sec in enumerate(secs):
    d = cells[cells.slice_id.astype(str) == sec]
    x, y = d.x_centroid.to_numpy(), d.y_centroid.to_numpy()
    iso, dom = d.isodepth.to_numpy(), d.domain.to_numpy()

    ax = fig.add_subplot(gs[r, 0])
    _lo, _hi = iso.min(), iso.max()
    sm = ax.scatter(x, y, c=iso, s=1.2, cmap='Blues', linewidths=0, rasterized=True,
                    vmin=_lo - 0.35 * (_hi - _lo), vmax=_hi)
    ax.set_title(f'{sec}  ·  GASTON isodepth', loc='left', fontsize=9.5, color=INK)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.set_label('isodepth', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[r, 1])
    k = int(np.nanmax(dom)) + 1
    from matplotlib.colors import LinearSegmentedColormap
    _cm = LinearSegmentedColormap.from_list('b', ['#c6dbef', '#08306b'], k)
    sm = ax.scatter(x, y, c=dom, s=1.2, cmap=_cm, vmin=-0.5, vmax=k - 0.5,
                    linewidths=0, rasterized=True)
    ax.set_title(f'{sec}  ·  domains (level sets)', loc='left', fontsize=9.5, color=INK)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, ticks=range(k))
    cb.set_label('domain', fontsize=7.5, labelpad=1); cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[r, 2])
    ax.scatter(iso, d.local_tumor_frac, s=1.0, c='#2a78d6', alpha=0.10,
               linewidths=0, rasterized=True)
    b = pd.qcut(iso, 20, duplicates='drop')
    g = pd.DataFrame({'b': b, 'v': d.local_tumor_frac.to_numpy(),
                      'i': iso}).groupby('b', observed=True).mean()
    ax.plot(g['i'], g['v'], '-o', color='#eb6834', lw=2, ms=4)
    rho = summ[sec]['rho_isodepth_local_tumor_frac']
    ax.set_title(f'isodepth ↔ tumour density   |ρ| = {abs(rho):.2f}', loc='left',
                 fontsize=9.5, color=INK)
    ax.set_xlabel('isodepth'); ax.set_ylabel('local tumour fraction')
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[r, 3])
    t = d[np.isfinite(d.MAPK_MPAS)]
    if len(t) > 100:
        bb = pd.qcut(t.isodepth, 12, duplicates='drop')
        gg = pd.DataFrame({'b': bb, 'v': t.MAPK_MPAS.to_numpy(), 'i': t.isodepth.to_numpy()})
        m = gg.groupby('b', observed=True).agg(v=('v', 'mean'), i=('i', 'mean'),
                                               se=('v', lambda z: z.std() / max(np.sqrt(len(z)), 1)))
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.errorbar(m['i'], m['v'], yerr=1.96 * m['se'], fmt='-o', color='#eb6834',
                    lw=2, ms=4, capsize=2, ecolor=MUTED)
        rm = summ[sec]['rho_isodepth_MAPK']
        ax.set_title(f'drug response along isodepth   ρ = {rm:+.3f}', loc='left',
                     fontsize=9.5, color=INK)
        ax.text(0.03, 0.06, f'n tumour = {len(t):,}', transform=ax.transAxes,
                fontsize=7.6, color=INK2)
    ax.set_xlabel('isodepth'); ax.set_ylabel('MAPK MPAS (mean ± 95% CI)')
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)

fig.suptitle('GASTON isodepth fitted on the salient-CC embedding (KRASi-treated sections)',
             x=0.05, ha='left', fontsize=11.5, color=INK, fontweight='bold')
p = OUT / 'gaston_isodepth.png'
fig.savefig(p, dpi=200, bbox_inches='tight'); fig.savefig(str(p).replace('.png', '.pdf'), bbox_inches='tight')
print('wrote', p)
for sec in secs:
    print(f'  {sec}: rho(isodepth, tumour frac) {summ[sec]["rho_isodepth_local_tumor_frac"]:+.3f} | '
          f'rho(isodepth, MAPK) {summ[sec]["rho_isodepth_MAPK"]:+.3f}')
