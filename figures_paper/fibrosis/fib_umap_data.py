#!/usr/bin/env python3
"""Global UMAPs of z and s for the fibrosis figure (panels m-p).  Fibrosis-core (condition) cells with one H&E region
label, sampled per region (<= 5000 cells each, all cells for smaller regions) so rare lesions such as fibroblastic foci are
visible; the same cells for z and s.  UMAP on the standardised latent: 15 neighbours, min_dist 0.3, random_state 0.
-> data/umap.npz (umap_z, umap_s, region, lineage, cell_type)"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import os, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, umap
from pathlib import Path
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = Path(f'{_HR}'); OUT = H / 'figures_paper/fibrosis/data'
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
o = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r').obs
pos = pd.Series(np.arange(len(o)), index=o.index)
d = pd.read_csv(f'{_HW}/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type', 'stage'])
d = d[d.stage != 'Unaffected'].drop_duplicates(['obs', 'annotation_type']); d = d[~d.obs.duplicated(keep=False)]
d['i'] = pos.reindex(d.obs).to_numpy()
rng = np.random.default_rng(0)
pick = np.sort(np.concatenate([g.i.to_numpy() if len(g) <= 5000 else rng.choice(g.i.to_numpy(), 5000, replace=False) for _, g in d.groupby('annotation_type')]))
reg = d.set_index('i').annotation_type.reindex(pick).to_numpy()
res = {}
for k, f in [('z', 'background.npy'), ('s', 'salient.npy')]:
    X = StandardScaler().fit_transform(np.asarray(np.load(MD / f, mmap_mode='r')[pick]))
    res[k] = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(X); print(k, 'done', flush=True)
np.savez(OUT / 'umap.npz', umap_z=res['z'], umap_s=res['s'], region=reg, lineage=o.lineage.astype(str).to_numpy()[pick], cell_type=o.cell_type.astype(str).to_numpy()[pick])
print(len(pick), 'cells', pd.Series(reg).value_counts().to_dict())
