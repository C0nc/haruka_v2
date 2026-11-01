#!/usr/bin/env python3
"""Module pathway enrichment from the saved module table.

Fixes the bug in hotspot_heatmap_enrichment.py: when `background=` is passed,
gseapy's Enrichr result has NO 'Overlap' column, so indexing row['Overlap']
raised KeyError for every module. Reads modules from the main run, so no
Hotspot recompute is needed.
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd, gseapy, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path('/home/yancui/haruka_v2/bench_kras_xenium/hotspot_arms')
OUT = Path('/home/yancui/haruka_v2/bench_kras_xenium/figures')
SECTION, TOPN = 'JB_M1', 5
ARMS = ['space', 'z-CC', 'z-CC'.replace('z','s')]
COL = {'space':'#0f6e7a','z-CC':'#2a78d6','s-CC':'#eb6834'}
INK, INK2, MUTED, GRID = '#0b0b0b','#52514e','#8a8a86','#e3e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':MUTED,
  'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':'white',
  'axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})

M = pd.read_csv(D/'modules.csv')
M = M[(M.section == SECTION) & (M.module > 0)]
A = pd.read_csv(D/'autocorrelation.csv.gz')
panel = sorted(A[A.section == SECTION].gene.astype(str).unique())
print(f'{SECTION}: {M.arm.nunique()} arms, background panel {len(panel)} genes', flush=True)

recs = []
for arm in ARMS:
    sub = M[M.arm == arm]
    for m in sub.module.value_counts().head(TOPN).index:
        genes = sub[sub.module == m].gene.astype(str).tolist()
        try:
            e = gseapy.enrichr(gene_list=genes, background=panel,
                               gene_sets=['GO_Biological_Process_2026','MSigDB_Hallmark_2020'],
                               organism='Mouse', outdir=None, no_plot=True)
            r = e.results.sort_values('Adjusted P-value').head(6)
            for _, row in r.iterrows():
                recs.append(dict(arm=arm, module=int(m), n_genes=len(genes),
                                 term=row['Term'], padj=float(row['Adjusted P-value']),
                                 odds=float(row.get('Odds Ratio', np.nan)),
                                 overlap=row.get('Overlap', ''),     # absent when background= is used
                                 genes=row['Genes']))
            t = r.iloc[0]
            print(f'  {arm:6s} mod {int(m):2d} (n={len(genes):3d}): {t["Term"][:58]}  padj={t["Adjusted P-value"]:.2e}',
                  flush=True)
        except Exception as ex:
            print(f'  {arm} mod {int(m)}: FAILED {type(ex).__name__}: {ex}', flush=True)
        time.sleep(1.0)

E = pd.DataFrame(recs)
E.to_csv(D/f'enrichment_{SECTION}.csv', index=False)
sig = E[E.padj < 0.05]
print(f'\n{len(E)} terms returned, {len(sig)} at padj<0.05', flush=True)

fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6))
for ax, arm in zip(axes, ARMS):
    s = sig[sig.arm == arm].drop_duplicates('term').nsmallest(10, 'padj')
    if len(s) == 0:
        ax.text(.5,.5,'no term at padj < 0.05', ha='center', va='center',
                transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        y = np.arange(len(s))[::-1]
        ax.barh(y, -np.log10(s.padj), color=COL[arm], height=.62)
        ax.set_yticks(y); ax.set_yticklabels([t[:60] for t in s.term], fontsize=7.2)
        ax.set_xlabel('−log10 adjusted P')
        ax.axvline(-np.log10(0.05), color=MUTED, ls='--', lw=1)
        ax.grid(axis='x', color=GRID, lw=.7); ax.set_axisbelow(True)
    ax.set_title(f'{arm}', loc='left', fontsize=10, color=INK)
fig.suptitle(f'Hotspot module enrichment by graph — section {SECTION} '
             f'(GO BP 2026 + Hallmark, panel background)',
             x=0.015, ha='left', fontsize=12, color=INK, fontweight='bold')
fig.tight_layout(rect=[0,0,1,.93])
p = OUT/'hotspot_enrichment.png'
fig.savefig(p, dpi=200, bbox_inches='tight'); fig.savefig(str(p).replace('.png','.pdf'), bbox_inches='tight')
print('wrote', p, flush=True)
