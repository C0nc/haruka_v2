#!/usr/bin/env python3
"""Arm-swap control (Fig. 3 / ED Fig. 3 'arm swap'): control-vs-control CODEX design.

codex_wholeslice_all7_no_oligo.h5ad restricted to Ctl-1, Ctl-3, Ctl-4; Ctl-3 (amyloid-positive control) is relabelled
condition = 'disease'. Expression, coordinates and annotations are copied unchanged from all7_no_oligo.
    python build_ctl3_vs_ctl14.py [--check]   (--check compares against an existing file instead of writing)
"""
import sys, os
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, anndata as ad
from pathlib import Path

D = Path(__file__).resolve().parent
a = ad.read_h5ad(D / 'codex_wholeslice_all7_no_oligo.h5ad')
b = a[a.obs.donor_id.astype(str).isin(['Ctl-1', 'Ctl-3', 'Ctl-4'])].copy()
b.obs['condition'] = np.where(b.obs.donor_id.astype(str) == 'Ctl-3', 'disease', 'control')
b.obs['condition'] = b.obs['condition'].astype('category')
w = dict(b.uns['wholeslice']); w['variant'] = 'ctl3VsCtl14'
w['note'] = 'F2: control-vs-control; Ctl-3 (amyloid-positive control) as the "disease" arm'; b.uns['wholeslice'] = w
out = D / 'codex_wholeslice_ctl3VsCtl14.h5ad'
if '--check' in sys.argv:
    r = ad.read_h5ad(out)
    X1 = b.X.toarray() if hasattr(b.X, 'toarray') else b.X; X2 = r.X.toarray() if hasattr(r.X, 'toarray') else r.X
    print('obs_names equal', (b.obs_names == r.obs_names).all(), '| X max abs diff', float(np.abs(X1 - X2).max()),
          '| condition equal', (b.obs.condition.astype(str).values == r.obs.condition.astype(str).values).all())
else:
    b.write_h5ad(out); print('wrote', out, b.shape)
