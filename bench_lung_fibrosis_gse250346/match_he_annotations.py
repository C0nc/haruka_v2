#!/usr/bin/env python3
"""Match the GSE250346 H&E region annotations to the Haruka fibrosis h5ad cells.

Input:  GSE250346_HE_annotations.tar.gz (GEO supplementary), extracted to <HE>/x/
        bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad
Key:    barcode | x_centroid (0.1 um) | y_centroid (0.1 um); barcode = obs_name after the first '_'.
Output: <HE>/he_annot_matched.csv.gz  (annotation rows + obs, stage = sample_affect, core = sample)

Download:
  mkdir -p <HE>/raw <HE>/x && cd <HE>/raw
  curl -sSLO https://ftp.ncbi.nlm.nih.gov/geo/series/GSE250nnn/GSE250346/suppl/GSE250346_HE_annotations.tar.gz
  tar -xzf GSE250346_HE_annotations.tar.gz -C ../x
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import os
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import anndata as ad, pandas as pd, numpy as np

H = f'{_HR}'
HE = os.environ.get('HE_DIR', f'{_HW}/haruka_fibrosis_he')
o = ad.read_h5ad(f'{H}/bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r').obs
d = pd.read_csv(f'{HE}/x/GSE250346_HE_annotations/cells_partitioned_by_annotation.csv')
bc = pd.Series(o.index.str.split('_', n=1).str[1], index=o.index)
k = lambda b, x, y: b + '|' + np.round(x, 1).astype(str) + '|' + np.round(y, 1).astype(str)
ko = pd.Series(o.index, index=k(bc.values, o.x_centroid.astype(float).values, o.y_centroid.astype(float).values))
print('duplicate keys in h5ad', ko.index.duplicated().sum())
d['obs'] = ko.reindex(k(d.cell_id.values, d.x_centroid.values, d.y_centroid.values)).values
print('matched fraction', d.obs.notna().mean())
d = d[d.obs.notna()].copy()
d['stage'] = o.sample_affect.astype(str).reindex(d.obs).values
d['core'] = o['sample'].astype(str).reindex(d.obs).values
f = d[d.stage != 'Unaffected']
print('fibrosis cells annotated', f.obs.nunique(), 'of', (o.sample_affect != 'Unaffected').sum(), 'cores', f.core.nunique())
d.to_csv(f'{HE}/he_annot_matched.csv.gz', index=False)
