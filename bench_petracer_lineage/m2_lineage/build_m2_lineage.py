#!/usr/bin/env python3
"""M2-1 tumour vs non-tumour WITH malignant cells in the condition arm (for the lineage question).

Starts from the re-audit design (reaudit_2026-10-16/petracer_M2-1_tumour_vs_nontumour.h5ad):
control = non-malignant cells outside the tumours beyond the 50 um buffer; disease = non-malignant
cells inside beyond the buffer; halo = buffer band + types found in one region only.
Change: malignant cells inside the tumours beyond the buffer move from halo to disease, so s is
trained on them.  Lineage fields (clone, clade, fitness, type, detection_rate, tree) are copied from
the source h5td for evaluation only; they are not model inputs.
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import h5py, numpy as np, pandas as pd, anndata as ad
SRC = '/home/yancui/Jill/fourteen/petracer_data/M2_tumor_tracing.h5td'
A = ad.read_h5ad(f'{_HR}/bench_petracer_lineage/reaudit_2026-10-16/petracer_M2-1_tumour_vs_nontumour.h5ad')
BUF = float(A.uns['design']['buffer_um'])
with h5py.File(SRC, 'r') as h:
    S = ad.io.read_elem(h['obs']); chars = ad.io.read_elem(h['obsm/characters'])
key = pd.Index([n.split('_', 1)[1] for n in A.obs_names]); idx = S.index.get_indexer(key); assert (idx >= 0).all()
mal = (A.obs.cell_type.astype(str) == 'Malignant').to_numpy(); inside = A.obs.signed_boundary_dist.to_numpy() >= BUF
cond = A.obs.condition.astype(str).to_numpy().copy(); cond[mal & inside] = 'disease'
A.obs['condition'] = pd.Categorical(cond)
for c in ['clone', 'clade', 'fitness', 'type', 'detection_rate', 'tree', 'tumor']:
    A.obs[f'lin_{c}'] = S[c].astype(str).to_numpy()[idx]
ch = chars.iloc[idx] if hasattr(chars, 'iloc') else chars[idx]
A.obsm['lineage_characters'] = np.asarray(pd.DataFrame(ch).astype(str).to_numpy())
A.uns['design'] = dict(A.uns['design'], malignant='inside tumour beyond buffer -> disease (trained); lineage fields evaluation only')
A.write_h5ad('petracer_M2-1_lineage.h5ad', compression='gzip')
print(pd.crosstab(A.obs.condition, A.obs.cell_type.astype(str) == 'Malignant'))
print(pd.crosstab(A.obs.lin_clone[mal], A.obs.condition[mal]))
