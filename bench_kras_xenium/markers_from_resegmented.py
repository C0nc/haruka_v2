#!/usr/bin/env python3
"""Per-domain markers using the DP-resegmented GASTON labels (not the quantile fallback)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, warnings; warnings.filterwarnings('ignore')

D = Path('/home/yancui/haruka_v2/bench_kras_xenium/gaston_scc')
lab = {}
for f in sorted(D.glob('gaston_JB_*_resegmented.npz')):
    d = np.load(f, allow_pickle=True)
    for c, l in zip(d['cell_id'].astype(str), d['labels']):
        lab[c] = int(l)
print(f'{len(lab)} cells with DP domain labels', flush=True)

h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
# cell_id is not globally unique (52 collide across arms), so restrict to the
# treated rows the CC was built on before matching on cell_id.
CCROWS = np.load('/home/yancui/haruka_v2/results_real_compgated_nb_final/'
                 'mouselung_kras_xenium_clean_noC3/haruka_z_s_cellcharter_cc3_condition_only/'
                 'source_rows.npy')
treated = np.zeros(h.n_obs, dtype=bool); treated[CCROWS] = True
cid = h.obs['cell_id'].astype(str).to_numpy()
sel = treated & np.isin(cid, list(lab.keys()))
a = h[sel].to_memory(); h.file.close()
a.obs['cid'] = a.obs['cell_id'].astype(str)
a.obs['domain'] = pd.Categorical(a.obs.cid.map(lab).astype(int).astype(str))
a.X = a.layers['counts'].copy(); sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.tl.rank_genes_groups(a, 'domain', method='wilcoxon', n_genes=30)
mk = sc.get.rank_genes_groups_df(a, None)
mk.to_csv(D / 'domain_markers_resegmented.csv', index=False)

print('\n=== markers per DP domain (top 8 by logFC, padj<0.05) ===')
for g in sorted(mk.group.unique(), key=int):
    t = mk[(mk.group == g) & (mk.pvals_adj < 0.05)].nlargest(8, 'logfoldchanges')
    n = (a.obs.domain == g).sum()
    print(f'  domain {g} (n={n:6d}): ' + ', '.join(t.names.tolist()), flush=True)

print('\n=== cell-type composition per DP domain (top 4) ===')
comp = pd.crosstab(a.obs.domain, a.obs.celltype.astype(str), normalize='index') * 100
for g in comp.index:
    top = comp.loc[g].nlargest(4)
    print(f'  domain {g}: ' + ', '.join(f'{k} {v:.0f}%' for k, v in top.items()), flush=True)
print('\nwrote', D / 'domain_markers_resegmented.csv')
