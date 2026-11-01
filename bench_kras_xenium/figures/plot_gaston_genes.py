#!/usr/bin/env python3
"""GASTON on raw genes vs on the salient-CC embedding, with annotated domains.

Domain colours: Okabe-Ito 6 (validated; worst all-pairs CVD dE 7.6, which is legal
only with secondary encoding -> every domain is also directly labelled on the map).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, linregress

G = Path('/home/yancui/haruka_v2/bench_kras_xenium/gaston_genes')
S = Path('/home/yancui/haruka_v2/bench_kras_xenium/gaston_scc')
OUT = Path('/home/yancui/haruka_v2/bench_kras_xenium/figures')
OKABE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8a86', '#e3e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})

cells = pd.read_csv(G / 'gaston_genes_cells.csv.gz')
summ = json.loads((G / 'gaston_genes_summary.json').read_text())
mk = pd.read_csv(G / 'gaston_genes_domain_markers.csv')
comp = pd.read_csv(G / 'gaston_genes_domain_celltype.csv', index_col=0)

# annotate each domain by its dominant cell types
ann = {}
for g in comp.index:
    t = comp.loc[g].nlargest(2)
    ann[int(g)] = f'{t.index[0]} {t.iloc[0]:.0f}%'
secs = sorted(cells.slice_id.unique())
n = len(secs)

fig = plt.figure(figsize=(14.6, 3.15 * n + 0.5))
gs = gridspec.GridSpec(n, 5, wspace=0.34, hspace=0.46, left=0.045, right=0.985,
                       top=0.94, bottom=0.075)
for r, sec in enumerate(secs):
    d = cells[cells.slice_id == sec]
    x, y, iso, dom = (d.x_centroid.to_numpy(), d.y_centroid.to_numpy(),
                      d.isodepth.to_numpy(), d.domain.to_numpy().astype(int))
    k = int(dom.max()) + 1

    ax = fig.add_subplot(gs[r, 0])
    lo, hi = iso.min(), iso.max()
    sm = ax.scatter(x, y, c=iso, s=1.2, cmap='Blues', linewidths=0, rasterized=True,
                    vmin=lo - 0.35 * (hi - lo), vmax=hi)
    ax.set_title(f'{sec} · isodepth (genes)', loc='left', fontsize=9.5, color=INK)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[r, 1])
    for j in range(k):
        m = dom == j
        ax.scatter(x[m], y[m], s=1.2, color=OKABE[j % 6], linewidths=0, rasterized=True)
        if m.sum() > 200:                       # direct label = the required relief
            ax.text(np.median(x[m]), np.median(y[m]), str(j), fontsize=9, fontweight='bold',
                    color='white', ha='center', va='center',
                    path_effects=[__import__('matplotlib.patheffects', fromlist=['x'])
                                  .withStroke(linewidth=2.2, foreground='black')])
    ax.set_title(f'{sec} · domains (L={k})', loc='left', fontsize=9.5, color=INK)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    ax = fig.add_subplot(gs[r, 2])
    b = pd.qcut(iso, 20, duplicates='drop')
    gg = pd.DataFrame({'b': b, 'v': d.local_tumor_frac.to_numpy(), 'i': iso}) \
           .groupby('b', observed=True).mean()
    ax.plot(gg['i'], gg['v'], '-o', color=OKABE[5], lw=2, ms=4)
    ax.set_title(f"tumour density  |ρ|={abs(summ[sec]['rho_isodepth_local_tumor_frac']):.2f}",
                 loc='left', fontsize=9.5, color=INK)
    ax.set_xlabel('isodepth'); ax.set_ylabel('local tumour fraction')
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[r, 3])
    t = d[np.isfinite(d.MAPK_MPAS)]
    if len(t) > 100:
        bb = pd.qcut(t.isodepth, 12, duplicates='drop')
        m2 = pd.DataFrame({'b': bb, 'v': t.MAPK_MPAS.to_numpy(), 'i': t.isodepth.to_numpy()}) \
               .groupby('b', observed=True).agg(v=('v','mean'), i=('i','mean'),
                                                se=('v', lambda z: z.std()/max(np.sqrt(len(z)),1)))
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.errorbar(m2['i'], m2['v'], yerr=1.96*m2['se'], fmt='-o', color=OKABE[5],
                    lw=2, ms=4, capsize=2, ecolor=MUTED)
    ax.set_title(f"drug response  ρ={summ[sec]['rho_isodepth_MAPK']:+.3f}", loc='left',
                 fontsize=9.5, color=INK)
    ax.set_xlabel('isodepth'); ax.set_ylabel('MAPK MPAS')
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)

    # GASTON's expression model: piecewise-linear in isodepth, fit per domain
    ax = fig.add_subplot(gs[r, 4])
    genes = (mk[(mk.pvals_adj < 0.05)].sort_values('logfoldchanges', ascending=False)
             .drop_duplicates('group').nlargest(3, 'logfoldchanges').names.tolist())
    expr = pd.read_csv(G / 'gaston_genes_expression.csv.gz') if (G/'gaston_genes_expression.csv.gz').exists() else None
    if expr is not None:
        e = expr[expr.slice_id == sec]
        for gi, gn in enumerate(genes):
            if gn not in e.columns: continue
            v = e[gn].to_numpy(); iso_e = e.isodepth.to_numpy(); dm = e.domain.to_numpy().astype(int)
            bq = pd.qcut(iso_e, 20, duplicates='drop')
            mm = pd.DataFrame({'b': bq, 'v': v, 'i': iso_e}).groupby('b', observed=True).mean()
            ax.plot(mm['i'], mm['v'], 'o', ms=3, color=OKABE[gi], alpha=0.55)
            for j in range(k):
                s_ = dm == j
                if s_.sum() < 50: continue
                lr = linregress(iso_e[s_], v[s_])
                xs = np.linspace(iso_e[s_].min(), iso_e[s_].max(), 20)
                ax.plot(xs, lr.intercept + lr.slope*xs, '-', color=OKABE[gi], lw=1.8)
        ax.legend(handles=[Line2D([], [], color=OKABE[i], lw=2, label=g)
                           for i, g in enumerate(genes)], frameon=False, fontsize=7.2)
    ax.set_title('gene ~ isodepth (piecewise per domain)', loc='left', fontsize=9.5, color=INK)
    ax.set_xlabel('isodepth'); ax.set_ylabel('log-normalised expression')
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)

leg = [Line2D([], [], marker='o', ls='', color=OKABE[j], label=f'{j}: {ann.get(j,"")}')
       for j in sorted(ann)]
fig.legend(handles=leg, frameon=False, fontsize=8, ncol=min(6, len(leg)),
           loc='lower center', bbox_to_anchor=(0.5, 0.005))
fig.suptitle('GASTON fitted on RAW GENE EXPRESSION (KRASi-treated sections), domains annotated '
             'by dominant cell type', x=0.045, ha='left', fontsize=11.5, color=INK, fontweight='bold')
p = OUT / 'gaston_genes.png'
fig.savefig(p, dpi=200, bbox_inches='tight'); fig.savefig(str(p).replace('.png','.pdf'), bbox_inches='tight')
print('wrote', p)
for sec in secs:
    print(f"  {sec}: L={summ[sec]['n_domains']} loss={summ[sec]['final_loss']:.3f} "
          f"rho(iso,tumour)={summ[sec]['rho_isodepth_local_tumor_frac']:+.3f} "
          f"rho(iso,MAPK)={summ[sec]['rho_isodepth_MAPK']:+.3f}")
print('\ndomain annotation:', json.dumps(ann, indent=2))
