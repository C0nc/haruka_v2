#!/usr/bin/env python3
"""Independent myeloid cell typing with Astir (Geuenich et al., Cell Systems 2021).

Astir assigns each cell from a literature marker table; it never sees the Leiden
clusters, so agreement with the marker-profile labels is an independent check.
IBA1 and CD11b are pan-myeloid (every cell was selected IBA1+) and are excluded.
Donor is a design covariate so donor intensity offsets don't drive the calls.
"""
import numpy as np, pandas as pd, anndata as ad, torch
from pathlib import Path
from astir import Astir
from sklearn.metrics import adjusted_rand_score

OUT = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/celltype_annotation')
a = ad.read_h5ad('/home/yancui/haruka_v2/bench_codex_cns_2026/data/processed/codex_cns_myeloid_primary_no_aml_no_combat.h5ad')
expr = pd.DataFrame(np.asarray(a.layers['log1p_intensity'], float), index=a.obs_names.astype(str), columns=list(a.var_names))
donor = a.obs.donor_id.astype(str)
design = pd.get_dummies(donor).astype(float); design.insert(0, 'intercept', 1.0)
design = design.iloc[:, :-1]                         # drop one donor level (reference)
design.index = expr.index
markers = {'cell_type': {
    'Homeostatic microglia': ['TMEM119'],
    'Activated microglia': ['CD68', 'HLA-DR', 'CD74', 'CD11c'],
    'Border-associated macrophage': ['CD163', 'CD14'],
    'Monocyte-derived macrophage': ['CD14', 'CD45', 'HLA-DR'],
}, 'cell_state': {}}
torch.manual_seed(0)
ast = Astir(input_expr=expr, marker_dict=markers, design=design, random_seed=0, dtype=torch.float32)
ast.fit_type(max_epochs=100, batch_size=1024, learning_rate=1e-3, n_init=5, n_init_epochs=5)
prob = ast.get_celltype_probabilities()
call = ast.get_celltypes(threshold=0.7).iloc[:, 0]
call.index = expr.index; prob.index = expr.index
cl = a.obs.paper_initial_leiden.astype(str).to_numpy()
pd.concat([call.rename('astir_call'), prob.add_prefix('p_')], axis=1).assign(leiden=cl).to_csv(OUT / 'astir_calls.csv.gz', compression='gzip')

order = sorted(np.unique(cl), key=int)
T = pd.crosstab(pd.Series(cl, name='leiden'), call.rename('astir').reset_index(drop=True), normalize='index').loc[order] * 100
print('=== Astir call (% of each Leiden cluster), threshold 0.7 ===')
pd.set_option('display.width', 220); print(T.round(1).to_string())
print('\nmajority Astir call per cluster:')
for c in order:
    r = T.loc[c].sort_values(ascending=False)
    print(f'  {c:>2s}: {r.index[0]:30s} {r.iloc[0]:5.1f}%   (2nd: {r.index[1]} {r.iloc[1]:.1f}%)')
print(f'\nARI(Astir call, Leiden) = {adjusted_rand_score(cl, call.to_numpy()):.3f}   overall Unknown/Other = '
      f'{100*call.isin(["Unknown","Other"]).mean():.1f}%')
