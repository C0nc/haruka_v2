#!/usr/bin/env python3
"""HD MERFISH (MM-188, 6 control / 8 HD donors, human frontal cortex) -> Haruka input.
counts = obsm['X_raw'] (raw integers); spatial = obsm['X_spatial']; condition control/disease;
cell_type = cluster_general; slice_id = patient (one section per donor); batch = experimental batch."""
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
SRC = '/project/zhihuanglab/yancui/haruka_hd/raw/adata_short_all.h5ad'
OUT = '/project/zhihuanglab/yancui/haruka_hd/hd_haruka.h5ad'
h = ad.read_h5ad(SRC)
C = sparse.csr_matrix(np.asarray(h.obsm['X_raw'], np.float32))
assert np.allclose(C[:5000].toarray(), np.round(C[:5000].toarray()))
o = h.obs.copy()
o['condition'] = np.where(o.HD.astype(str) == 'True', 'disease', 'control')
o['cell_type'] = o.cluster_general.astype(str); o['subtype'] = o.cluster_names.astype(str)
o['slice_id'] = o.patient.astype(str); o['batch'] = o.batch.astype(str)
a = ad.AnnData(X=C.copy(), obs=o, var=pd.DataFrame(index=h.var_names.astype(str))); a.layers['counts'] = C
a.obsm['spatial'] = np.asarray(h.obsm['X_spatial'], np.float32)[:, :2]
# per-donor coordinate sanity: one contiguous section per donor
for p in sorted(o.slice_id.unique()):
    xy = a.obsm['spatial'][o.slice_id.to_numpy() == p]
    print(p, o.loc[o.slice_id == p, 'condition'].iloc[0], len(xy), 'x range %.0f-%.0f y range %.0f-%.0f' % (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()))
a.write_h5ad(OUT); print(a); print(pd.crosstab(o.batch, o.condition))
