#!/usr/bin/env python3
"""HD fit inputs from hd_haruka.h5ad.
Unannotated cells (cluster_general nan, ~10%, low counts) -> cell_type 'Unassigned', condition 'context':
they stay as spatial neighbours for the niche targets but are excluded from both training arms (--exclude_label context).
  hd_fit_all.h5ad       14 donors (6 control / 8 HD), batch Aug is HD-only (2 donors)
  hd_fit_balanced.h5ad  12 donors without the Aug batch: June / July / Sep each 2 control + 2 HD
"""
import numpy as np, pandas as pd, anndata as ad
D = '/project/zhihuanglab/yancui/haruka_hd/'
a = ad.read_h5ad(D + 'hd_haruka.h5ad')
un = a.obs.cell_type.astype(str).isin(['nan', '']).to_numpy()
a.obs['cell_type'] = np.where(un, 'Unassigned', a.obs.cell_type.astype(str))
a.obs['condition'] = np.where(un, 'context', a.obs.condition.astype(str))
for c in ['cell_type', 'condition', 'batch', 'slice_id']: a.obs[c] = pd.Categorical(a.obs[c].astype(str))
a.write_h5ad(D + 'hd_fit_all.h5ad'); print('all', a.shape); print(pd.crosstab(a.obs.batch, a.obs.condition))
b = a[a.obs.batch.astype(str) != 'Aug'].copy()
for c in ['batch', 'slice_id']: b.obs[c] = b.obs[c].cat.remove_unused_categories()
b.write_h5ad(D + 'hd_fit_balanced.h5ad'); print('balanced', b.shape); print(pd.crosstab(b.obs.batch, b.obs.condition))
