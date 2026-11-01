#!/usr/bin/env python3
"""Exercise intervention (Sun et al. aging atlas, MERFISH 300 genes): old control (4 mice) vs old exercise (4 mice).
Young mice dropped. condition: control / exercise; cell_type = celltype; slice_id = mouse_id (one section per mouse,
neighbourhoods built within it); run_batch = Xenium/MERFISH run A/B (2 control + 2 exercise each, the batch key)."""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import numpy as np, pandas as pd, anndata as ad, json
from scipy import sparse
SRC = '/home/yancui/virtualorganoid/exercise.h5ad'
OUT = f'{_HW}/haruka_exercise/data/exercise_old_full.h5ad'
h = ad.read_h5ad(SRC, backed='r')
coh = h.obs.cohort.astype(str).to_numpy(); sel = np.flatnonzero(np.isin(coh, ['old_control', 'old_exercise']))
X = np.empty((len(sel), h.shape[1]), np.float32)
for l in range(0, len(sel), 20000):
    X[l:l + 20000] = np.asarray(h.X[sel[l:l + 20000]], np.float32)
assert np.allclose(X[:1000], np.round(X[:1000])), 'X is not integer counts'
C = sparse.csr_matrix(X); o = h.obs.iloc[sel].copy()
o['condition'] = np.where(o.cohort.astype(str) == 'old_exercise', 'exercise', 'control')
o['cell_type'] = o.celltype.astype(str); o['slice_id'] = o.mouse_id.astype(str); o['run_batch'] = o.batch.astype(str)
a = ad.AnnData(X=C.copy(), obs=o, var=h.var.copy()); a.layers['counts'] = C
a.obsm['spatial'] = np.asarray(h.obsm['spatial'])[sel].astype(np.float32)
a.write_h5ad(OUT)
audit = dict(source=SRC, output=OUT, shape=list(a.shape),
             counts=pd.crosstab([o.run_batch, o.slide_id], [o.condition, o.mouse_id]).to_dict())
json.dump(audit, open(OUT.replace('.h5ad', '.audit.json'), 'w'), indent=1, default=str)
print(a); print(pd.crosstab(o.run_batch, o.condition))
