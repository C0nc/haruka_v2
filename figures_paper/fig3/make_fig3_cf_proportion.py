#!/usr/bin/env python3
"""Figure 3 — Haruka's counterfactual neighbourhood proportions for condition cells.

For each AD cell the composition head gives
  background p([z,0])  the neighbourhood predicted from the shared space alone
  salient    p([z,s])  the fitted condition neighbourhood;  shift = p([z,s]) - p([z,0])
  bins   microglia fraction among the 20 nearest neighbours by distance to Abeta: observed, background,
         salient; dots = slices (STARmap) / donors (CODEX); dashed = observed in the reference arm
  decomposition  AD - reference change in neighbour fraction, matched on anatomy x own cell type,
         split into the part the background already predicts and the part the salient shift adds;
         diamond = observed change
Source: bench_real_discovery/cc_shift_info/check_cf_proportion.py (fit in-sample: the heads were trained on these cells).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import paperstyle as ps
from fig3_layout import LET
ps.setup()
S = Path('/home/yancui/haruka_v2/bench_real_discovery/cc_shift_info')
COL = {'observed': ps.LATENT['space'], 'p_bg': ps.LATENT['z'], 'p_full': ps.LATENT['s']}
LAB = {'observed': 'Observed', 'p_bg': 'Background p([z,0])', 'p_full': 'Salient p([z,s])'}
CFG = {'starmap': dict(title='STARmap, 2 AD slices', ref='Control slice', mic='Micro',
                       xt=['Aβ cell', '≤1', '1–2', '2–4', '4–8', '>8'], xl='Distance to nearest Aβ cell (cell spacings)',
                       show=['Micro', 'Inh', 'CTX-Ex', 'SMC', 'Astro', 'CA2', 'Endo'],
                       rename={'Micro': 'Microglia', 'Inh': 'Inhibitory', 'CTX-Ex': 'Cortical exc.', 'SMC': 'SMC', 'Astro': 'Astrocyte', 'CA2': 'CA2', 'Endo': 'Endothelial'}),
       'codex': dict(title='CODEX, 4 AD donors (grey matter)', ref='Control donors', mic='Microglia',
                     xt=['0–10', '10–20', '20–40', '40–60', '60–100', '>100'], xl='Distance to nearest Aβ plaque (µm)',
                     show=['Astrocyte', 'Microglia', 'Neuron'], rename={})}


def fonts(f):
    ok, sz = ps.check_fonts(HERE / f'{f}.svg')
    if not ok: print(f, 'FONT BELOW MINIMUM', sz)


for ds, c in CFG.items():
    # ---------- bins ----------
    B = pd.read_csv(S / f'check_cf_C4_microglia_bins_{ds}.csv'); Bs = pd.read_csv(S / f'check_cf_C4_microglia_bins_by_sample_{ds}.csv')
    order = list(B.bin); x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(62 * ps.MM, 46 * ps.MM))
    ax.fill_between(x, B.p_bg, B.p_full, color=ps.LATENT['s'], alpha=0.18, lw=0, label='Salient shift')
    for k in ['observed', 'p_bg', 'p_full']:
        ax.plot(x, B[k], color=COL[k], lw=1.1, marker='o', ms=2.6, label=LAB[k], zorder=3)
        for _, g in Bs.groupby('sample'):
            xi = np.array([order.index(b) for b in g.bin])
            ax.scatter(xi, g[k], s=4, color=COL[k], lw=0, alpha=0.55, zorder=2)
    ax.axhline(Bs.reference_observed.iloc[0], color=ps.NEUTRAL['mid'], lw=0.7, ls='--')
    ax.text(len(order) - 1, Bs.reference_observed.iloc[0], c['ref'], ha='right', va='bottom', fontsize=5.6, color=ps.NEUTRAL['dark'])
    ax.set_xticks(x); ax.set_xticklabels(c['xt']); ax.set_xlabel(c['xl'])
    ax.set_ylabel('Microglia among\n20 nearest neighbours'); ax.set_ylim(bottom=0)
    ax.set_title(c['title'], pad=2)
    ax.legend(loc='upper right', handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)
    stem = f'fig3_cf_bins_{ds}'; ps.panel_label(ax, LET(stem), x=-0.32, y=1.04)
    ps.save(fig, HERE / stem); plt.close(fig); fonts(stem)

    # ---------- decomposition ----------
    T = pd.read_csv(S / f'check_cf_C3_{ds}.csv', index_col=0).loc[c['show']]
    Q = pd.read_csv(S / f'check_cf_C3_by_sample_{ds}.csv')
    bg = T.pbg_cond - T.obs_ref; sal = T.pfull_cond - T.pbg_cond; obs = T.obs_cond - T.obs_ref
    y = np.arange(len(T))[::-1]
    fig, ax = plt.subplots(figsize=(56 * ps.MM, 46 * ps.MM))
    ax.barh(y, bg, height=0.62, color=ps.LATENT['z'], lw=0, label='Background p([z,0])')
    ax.barh(y, sal, left=bg, height=0.62, color=ps.LATENT['s'], lw=0, label='Salient shift')
    ax.scatter(obs, y, marker='D', s=9, color=ps.INK, zorder=4, lw=0, label='Observed')
    for t_, yi in zip(T.index, y):
        q = Q[Q.type == t_]
        ax.scatter(q.obs_cond - q.obs_ref, [yi] * len(q), marker='|', s=14, color=ps.INK, lw=0.6, zorder=4)
    ax.axvline(0, color=ps.NEUTRAL['mid'], lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([c['rename'].get(t_, t_) for t_ in T.index]); ax.tick_params(axis='y', length=0)
    ax.set_xlabel('AD − control change in\nneighbour fraction (matched)')
    ax.set_title('Region × cell-type matched', pad=2)
    ax.legend(loc='lower right', handlelength=1.0, borderaxespad=0.2, labelspacing=0.25, handletextpad=0.4)
    stem = f'fig3_cf_decomposition_{ds}'; ps.panel_label(ax, LET(stem), x=-0.38, y=1.04)
    ps.save(fig, HERE / stem); plt.close(fig); fonts(stem)
    pd.DataFrame({'observed_change': obs, 'background_part': bg, 'salient_part': sal}).to_csv(HERE / f'{stem}_source_data.csv')
    B.to_csv(HERE / f'fig3_cf_bins_{ds}_source_data.csv', index=False)
print('done')
