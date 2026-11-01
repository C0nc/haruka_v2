#!/usr/bin/env python3
"""STARmap (mouse, RNA) <-> CODEX (human, protein) major cell-type crosswalk + composition.

CODEX lineages come from segmentation masks (NeuN / GFAP / IBA1 / Olig2); there is no
vascular mask, and Olig2 was segmented only in AD-1, AD-4, Ctl-1.  Compositions are
therefore compared two ways: over the 3 CODEX-shared types (Neuron/Astrocyte/Microglia,
renormalised) for all slices, and over 4 types for the full3 donors.
"""
from pathlib import Path
import pandas as pd, anndata as ad
OUT = Path(__file__).resolve().parent
CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron',
         'Inh': 'Neuron', 'LHb': 'Neuron', 'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte',
         'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
CODEX_DEF = {'Neuron': 'NeuN mask', 'Astrocyte': 'GFAP mask', 'Oligodendrocyte': 'Olig2 mask (AD-1, AD-4, Ctl-1 only)',
             'Microglia': 'IBA1 mask (incl. macrophage states)', 'Vascular': 'not segmented'}
rows = [dict(major_type=m, starmap_ct_top=', '.join(k for k, v in CROSS.items() if v == m), codex_source=CODEX_DEF[m])
        for m in ['Neuron', 'Astrocyte', 'Oligodendrocyte', 'Microglia', 'Vascular']]
pd.DataFrame(rows).to_csv(OUT / 'crosswalk_table.csv', index=False)

s = ad.read_h5ad('/home/yancui/haruka_v2/bench_real_discovery/data/ad.h5ad', backed='r').obs
s = s.assign(major=s.ct_top.astype(str).map(CROSS), sample=s.slice_id.astype(str), condition=s.condition.astype(str))
assert s.major.notna().all()
c = pd.read_csv(OUT.parent / 'wholeslice_all_cells.csv.gz', usecols=['donor_id', 'lineage_mask'])
c = c.assign(major=c.lineage_mask.replace({'Myeloid': 'Microglia'}), sample=c.donor_id,
             condition=c.donor_id.str.startswith('AD').map({True: 'AD', False: 'control'}))
long = pd.concat([s[['sample', 'condition', 'major']].assign(dataset='STARmap (mouse, RNA)'),
                  c[['sample', 'condition', 'major']].assign(dataset='CODEX (human, protein)')])
cnt = long.groupby(['dataset', 'sample', 'condition', 'major']).size().unstack(fill_value=0)
cnt.to_csv(OUT / 'composition_counts.csv')
TR = ['Neuron', 'Astrocyte', 'Microglia']
f3 = cnt[TR].div(cnt[TR].sum(1), axis=0).round(3); f3.to_csv(OUT / 'composition_3shared_fraction.csv')
full = cnt[cnt.get('Oligodendrocyte', 0) > 0]
F4 = ['Neuron', 'Astrocyte', 'Oligodendrocyte', 'Microglia']
f4 = full[F4].div(full[F4].sum(1), axis=0).round(3); f4.to_csv(OUT / 'composition_4type_fraction.csv')
print(pd.DataFrame(rows).to_string(index=False)); print('\ncounts\n', cnt.to_string())
print('\n3 shared types (fractions)\n', f3.to_string()); print('\n4 types where Olig2 exists\n', f4.to_string())
