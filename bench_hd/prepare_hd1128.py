#!/usr/bin/env python3
"""HD MERFISH MM-1128 (adata_AGG_CAG.h5ad; 4 control / 5 HD donors, one cortical section each, 1,128 genes) -> Haruka input.
This is the panel that carries per-cell HD ground truth (evaluation only, never model input):
CAG (smFISH CAG-repeat signal, somatic expansion proxy), nHTT, Agg (1 = intranuclear inclusion, 0 = cell nearest an
extranuclear inclusion), Agg_bright/vol, lipo (lipofuscin granules), m_gamma (gammaH2AX).
counts = obsm['X_raw']; spatial = obsm['X_spatial']; slice_id = patient; batch Aug (2C/3H) / Sep (2C/2H)."""
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_hd/raw/adata_AGG_CAG.h5ad')
C = sparse.csr_matrix(np.asarray(h.obsm['X_raw'], np.float32)); assert np.allclose(C[:5000].toarray(), np.round(C[:5000].toarray()))
o = h.obs.copy()
o['condition'] = np.where(o.HD.astype(str) == 'True', 'disease', 'control')
o['cell_type'] = o.cluster_general.astype(str); o['subtype'] = o.cluster_names.astype(str)
o['slice_id'] = o.patient.astype(str); o['batch'] = o.batch.astype(str)
for c in ['condition', 'cell_type', 'subtype', 'slice_id', 'batch']: o[c] = pd.Categorical(o[c])
a = ad.AnnData(X=C.copy(), obs=o, var=pd.DataFrame(index=h.var_names.astype(str))); a.layers['counts'] = C
a.obsm['spatial'] = np.asarray(h.obsm['X_spatial'], np.float32)[:, :2]
a.write_h5ad('/project/zhihuanglab/yancui/haruka_hd/hd1128_haruka.h5ad'); print(a); print(pd.crosstab(o.batch, o.condition))
