#!/usr/bin/env python3
"""Figure 3a (draft) — myeloid state annotation of the CODEX AD clusters.

Heatmap of per-cluster marker z-scores, rows grouped by map category (colour strip),
with each cluster's white-matter share — the reactive/activated states sit in white
matter, which matters for every plaque analysis downstream.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import paperstyle as ps
ps.setup()
ANN = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/celltype_annotation')

Z = pd.read_csv(ANN / 'cluster_marker_zscores.csv', index_col=0); Z.index = Z.index.astype(str)
T = pd.read_csv(ANN / 'myeloid_annotation_table.csv', dtype={'cluster': str}).set_index('cluster')
ROWS = ['10', '3', '4', '5', '2', '7', '6', '1', '8', '0', '9']
SHORT = {'10': 'Homeostatic', '3': 'Homeostatic, CD45-low', '4': 'Homeostatic', '5': 'Homeostatic, CD68-low',
         '2': 'MHC-II+ homeostatic', '7': 'Reactive MHC-II+', '6': 'Activated (TMEM119\u2212)',
         '1': 'CD163+ MHC-II+ macrophage', '8': 'Border-assoc. macrophage',
         '0': 'Unresolved (low signal)', '9': 'QC flag (IBA1-low)'}
COLS = ['TMEM119', 'CD74', 'HLA-DR', 'CD68', 'CD11c', 'CD45', 'CD11b', 'CD14', 'CD163', 'IBA1']
M = Z.loc[ROWS, COLS].to_numpy()
cmap = LinearSegmentedColormap.from_list('div', ps.DIVERGING)


def draw(fig, gs_cell):
    g = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_cell, width_ratios=[.035, 1, .34, .05], wspace=.07)
    ax_s, ax_h, ax_w, ax_c = (fig.add_subplot(g[0, i]) for i in range(4))
    n = len(ROWS)
    for i, c in enumerate(ROWS):                                   # category strip
        ax_s.add_patch(plt.Rectangle((0, i - .5), 1, 1, color=ps.MYELOID[T.loc[c, 'map_category']], lw=0))
    ax_s.set_xlim(0, 1); ax_s.set_ylim(n - .5, -.5); ax_s.set_xticks([])
    ax_s.set_yticks(range(n)); ax_s.set_yticklabels([f'{SHORT[c]} (c{c})' for c in ROWS])
    for sp in ax_s.spines.values(): sp.set_visible(False)
    ax_s.tick_params(axis='y', length=0)
    im = ax_h.imshow(M, cmap=cmap, vmin=-1.5, vmax=1.5, aspect='auto')   # most cluster means lie within +/-1.5
    ax_h.set_xticks(range(len(COLS))); ax_h.set_xticklabels(COLS, rotation=90)
    ax_h.set_yticks([]); ax_h.tick_params(axis='x', length=0)
    for sp in ax_h.spines.values(): sp.set_visible(False)
    for b in [3.5, 5.5, 6.5, 8.5]:                                  # group dividers
        ax_h.axhline(b, color='white', lw=1.2)
    wm = T.loc[ROWS, 'white_matter_fraction'].to_numpy()
    ax_w.barh(range(n), wm, color=[ps.MYELOID[T.loc[c, 'map_category']] for c in ROWS], height=.7)
    ax_w.set_ylim(n - .5, -.5); ax_w.set_xlim(0, 1); ax_w.set_xticks([0, 1])
    ax_w.set_yticks([]); ax_w.set_xlabel('White-matter\nfraction', labelpad=1)
    ax_w.axvline(np.average(wm, weights=T.loc[ROWS, 'n_cells']), color=ps.NEUTRAL['mid'], lw=.6, ls='--')
    cb = fig.colorbar(im, cax=ax_c); cb.set_label('Marker z-score', labelpad=1)
    cb.set_ticks([-1.5, 0, 1.5]); cb.outline.set_linewidth(.4)
    pos = ax_c.get_position(); ax_c.set_position([pos.x0, pos.y0 + pos.height * .3, pos.width, pos.height * .4])
    return ax_s


if __name__ == '__main__':
    fig = plt.figure(figsize=(100 * ps.MM, 64 * ps.MM))
    gs = gridspec.GridSpec(1, 1, left=.37, right=.93, top=.94, bottom=.2)
    ax0 = draw(fig, gs[0, 0])
    ps.panel_label(ax0, 'a', x=-14, y=1.0)
    files = ps.save(fig, HERE / 'fig3a_myeloid_annotation')
    T.loc[ROWS].assign(marker_profile=[dict(zip(COLS, np.round(M[i], 2))) for i in range(len(ROWS))]) \
        .to_csv(HERE / 'fig3a_source_data.csv')
    for f in files:
        if f.endswith('.svg'):
            ok, sz = ps.check_fonts(f); print(Path(f).name, 'fonts', sz, 'OK' if ok else 'BELOW MINIMUM')
