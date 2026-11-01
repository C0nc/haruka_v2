"""Read Stereo-seq cell-bin GEF (HDF5) with h5py and write one h5ad per chip.
Usage: python -I 01_build_h5ad.py <raw_dir> <out_dir>
"""
import sys, os, json
import numpy as np, h5py, scipy.sparse as sp, anndata as ad, pandas as pd

raw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
CHIPS = {'C02943C3': 'GSM8449354', 'B03018A2': 'GSM9659883', 'A03599E2': 'GSM9659884'}
summary = {}
for chip, gsm in CHIPS.items():
    f = h5py.File(os.path.join(raw, f'{gsm}_{chip}.adjusted.cellbin.gef'), 'r')
    cb = f['cellBin']
    cell = cb['cell'][:]
    gene = cb['gene'][:]
    exp = cb['cellExp'][:]
    n_c, n_g = len(cell), len(gene)
    # cellExp is ordered by cell; each cell owns geneCount entries starting at offset
    off = cell['offset'].astype(np.int64)
    gc = cell['geneCount'].astype(np.int64)
    indptr = np.concatenate([off, [off[-1] + gc[-1]]])
    assert np.all(np.diff(indptr) == gc), 'offsets not contiguous'
    assert indptr[-1] == len(exp)
    X = sp.csr_matrix((exp['count'].astype(np.float32), exp['geneID'].astype(np.int64), indptr), shape=(n_c, n_g))
    X.sum_duplicates()
    names = np.array([g.decode() for g in gene['geneName']])
    # cross-check with gene-major totals
    gtot = np.asarray(X.sum(0)).ravel()
    mism = int((np.abs(gtot - gene['expCount']) > 0).sum())
    var = pd.DataFrame({'gef_cellCount': gene['cellCount'], 'gef_expCount': gene['expCount']}, index=names)
    var.index = pd.Index(names).astype(str)
    if var.index.duplicated().any():
        var.index = ad.utils.make_index_unique(var.index)
    obs = pd.DataFrame({k: cell[k] for k in ['id', 'geneCount', 'expCount', 'dnbCount', 'area']},
                       index=[f'{chip}_{i}' for i in cell['id']])
    obs['chip'] = chip
    obs['total_counts'] = np.asarray(X.sum(1)).ravel()
    a = ad.AnnData(X=X.copy(), obs=obs, var=var)
    a.layers['counts'] = X
    a.obsm['spatial'] = np.c_[cell['x'], cell['y']].astype(np.float32)
    a.uns['gef_attrs'] = {k: str(v) for k, v in f.attrs.items()}
    a.write_h5ad(os.path.join(out, f'{chip}.h5ad'), compression='gzip')
    summary[chip] = dict(n_cells=n_c, n_genes=n_g, nnz=int(X.nnz), gene_total_mismatch=mism,
                         expCount_uint16_saturated=int((cell['expCount'] == 65535).sum()))
    print(chip, summary[chip], flush=True)
json.dump(summary, open(os.path.join(out, 'build_summary.json'), 'w'), indent=1)
