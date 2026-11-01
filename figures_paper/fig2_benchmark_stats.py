#!/usr/bin/env python3
"""Figure 2 (stats): benchmark on the paired mouse-brain simulation.

Per-section ARIs come from the scorer's saved outputs (raw spatial embedding ->
KMeans, k = number of ground-truth classes, seed 0, n_init 20), already verified
to reproduce the published table exactly for all 35 method x section cells.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

B = Path('/home/yancui/haruka_v2/bench_mouse_brain_scviva')
OUT = Path('/home/yancui/haruka_v2/figures_paper')
DEFAULT, ABL, BASE = '#0f6e7a', '#8cc3c8', '#a3acb5'
INK, INK2, MUTED, GRID = '#16202b', '#4d5966', '#8a96a3', '#e4e9ee'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.6,'axes.edgecolor':MUTED,
  'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,
  'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})

C = pd.read_csv(B/'results_shared_full_niche_compgated_lambda25_comparison/benchmark_with_baselines_per_section.csv')
C = C.rename(columns={'background_subregion_ari':'bg','salient_domain_ari':'sal'})
L = pd.read_csv(B/'results_shared_full_niche_compgated_lambda_sweep/per_section_metrics.csv')
L10 = L[np.isclose(L['lambda'], 10.0)][['section','background_ari','salient_ari']] \
        .rename(columns={'background_ari':'bg','salient_ari':'sal'}).assign(method='Haruka λ=10 composition-gated')
D = pd.concat([L10, C[['section','method','bg','sal']]], ignore_index=True)
D['balanced'] = (D.bg + D.sal) / 2

ORDER = [('Haruka λ=10 composition-gated','Haruka (default)',DEFAULT),
         ('Haruka λ=25 composition-gated','Haruka gated λ=25',ABL),
         ('Haruka λ=25 independent','Haruka indep. λ=25',ABL),
         ('Mini-batch CINEMA-OT + CellCharter','CINEMA-OT + CC',BASE),
         ('Expression + neighborhood aggregation','Expr. + aggregation',BASE),
         ('scVI + CellCharter','scVI + CC',BASE),
         ('ContrastiveVI z/s + CellCharter','ContrastiveVI + CC',BASE)]
assert set(D.method) == {o[0] for o in ORDER}, set(D.method) ^ {o[0] for o in ORDER}
assert (D.groupby('method').size() == 5).all()
D.to_csv(OUT/'fig2_benchmark_per_section.csv', index=False)

def boxpanel(ax, col, title, ylabel):
    rng = np.random.default_rng(0)
    for i, (m, lab, c) in enumerate(ORDER):
        v = D.loc[D.method == m, col].to_numpy()
        bp = ax.boxplot(v, positions=[i], widths=.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.4), whiskerprops=dict(color=MUTED, lw=1),
                        capprops=dict(color=MUTED, lw=1))
        bp['boxes'][0].set(facecolor=c, edgecolor=MUTED, alpha=.55 if c != DEFAULT else .7, lw=.9)
        ax.scatter(i + rng.uniform(-.14, .14, len(v)), v, s=22, color=c if c != BASE else '#6f7b88',
                   edgecolor='white', linewidth=.7, zorder=3)
        ax.scatter([i], [v.mean()], marker='D', s=26, color='white', edgecolor=INK, lw=1.1, zorder=4)
        ax.text(i, 1.035, f'{v.mean():.3f}', ha='center', fontsize=7.2, color=INK,
                fontweight='bold' if c == DEFAULT else 'normal')
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([o[1] for o in ORDER], fontsize=7.6, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_ylim(-.03, 1.08); ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=10, color=INK, pad=14)
    ax.grid(axis='y', color=GRID, lw=.7); ax.set_axisbelow(True)
    ax.axvspan(2.5, 6.5, color='#f4f6f8', zorder=0)

fig = plt.figure(figsize=(15.6, 9.4))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, .9], hspace=.55, wspace=.24,
                       left=.055, right=.985, top=.9, bottom=.07)
gsb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], width_ratios=[1.55, 1], wspace=.42)
boxpanel(fig.add_subplot(gs[0, 0]), 'bg',  'a  Background subregion recovery', 'ARI vs planted subregions (k = 7)')
boxpanel(fig.add_subplot(gs[0, 1]), 'sal', 'b  Salient domain recovery', 'ARI vs planted domains (k = 3)')
boxpanel(fig.add_subplot(gs[0, 2]), 'balanced', 'c  Balanced recovery', 'mean of the two ARIs')

# d: paired difference, Haruka default minus each comparator, per section
ax = fig.add_subplot(gsb[0, 0])
ref = D[D.method == ORDER[0][0]].set_index('section')
comps = ORDER[1:]
secs = sorted(ref.index)
w = .38
for k, metric, lab in [(0, 'bg', 'background'), (1, 'sal', 'salient')]:
    for i, (m, labm, c) in enumerate(comps):
        other = D[D.method == m].set_index('section')
        diff = (ref[metric] - other.loc[ref.index, metric]).to_numpy()
        x = i + (k - .5) * w
        ax.scatter(np.full(len(diff), x) + np.linspace(-.07, .07, len(diff)), diff, s=18,
                   color=DEFAULT if k == 0 else '#e08a3c', edgecolor='white', lw=.6, zorder=3)
        ax.plot([x - .12, x + .12], [diff.mean()] * 2, color=INK, lw=1.6, zorder=4)
        n_pos = int((diff > 0).sum())
        ax.text(x, max(diff.max(), 0) + .035, f'{n_pos}/5', ha='center', fontsize=7.2,
                color=INK, fontweight='bold' if n_pos == 5 else 'normal')
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(len(comps))); ax.set_xticklabels([c[1] for c in comps], fontsize=7.6)
ax.set_ylim(-.2, 1.02)
ax.set_ylabel('Haruka (default) − comparator ARI')
ax.set_title('d  Paired per-section difference (above 0 = Haruka better; label = sections won)',
             loc='left', fontsize=10, color=INK, pad=10)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=DEFAULT, label='background ARI'),
                   Line2D([], [], marker='o', ls='', color='#e08a3c', label='salient ARI'),
                   Line2D([], [], color=INK, lw=1.6, label='mean over 5 sections')],
          frameon=False, fontsize=7.6, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3)
ax.grid(axis='y', color=GRID, lw=.7); ax.set_axisbelow(True)

# e: per-section heat table, balanced recovery
ax = fig.add_subplot(gsb[0, 1])
T = D.pivot_table(index='method', columns='section', values='balanced').loc[[o[0] for o in ORDER]]
im = ax.imshow(T.to_numpy(), cmap='Blues', vmin=.2, vmax=.8, aspect='auto')
for r in range(T.shape[0]):
    for c in range(T.shape[1]):
        v = T.iat[r, c]
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7.3,
                color='white' if v > .62 else INK)
ax.set_xticks(range(T.shape[1])); ax.set_xticklabels(T.columns, fontsize=7.8)
ax.set_yticks(range(T.shape[0])); ax.set_yticklabels([o[1] for o in ORDER], fontsize=7.6)
ax.set_title('e  Balanced recovery by section', loc='left', fontsize=10, color=INK, pad=10)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03); cb.ax.tick_params(labelsize=7)

fig.suptitle('Figure 2 — Benchmark on the paired mouse-brain simulation (5 section pairs; points = sections, ◇ = mean)',
             x=.055, ha='left', fontsize=12.5, color=INK, fontweight='bold')
fig.text(.055, .935, 'Haruka variants (teal) vs external baselines (grey, shaded). Every method scored identically: '
         'raw spatial embedding → KMeans with k = number of planted classes.', fontsize=8.6, color=INK2)
p = OUT/'fig2_benchmark_stats.png'
fig.savefig(p, dpi=220, bbox_inches='tight'); fig.savefig(str(p).replace('.png', '.pdf'), bbox_inches='tight')
print('wrote', p)
S = D.groupby('method')[['bg','sal','balanced']].agg(['mean','std']).round(3)
print(S.loc[[o[0] for o in ORDER]].to_string())
