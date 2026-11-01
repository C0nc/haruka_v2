#!/usr/bin/env python3
"""Per cell type: is the s advantage on H&E regions (he_region/region_pairs.csv) larger in the cell types that carry
the most fibrosis salient signal?  Salient signal per cell type, on fibrosis cells (200k subsample):
  shift_L1   mean L1 of the decoded salient composition shift D([z,s]) - D([z,0])
  s_disease  held-out-donor cell AUROC of s for Unaffected vs fibrosis (zs_stage.csv, A_cell_s), and its gain over z
  path_rho   |Spearman| over fibrosis cores of the cell type's share with % pathology (composition change with progression)
"""
import os, json
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import spearmanr
H = Path('/home/yancui/haruka_v2'); W = H / 'bench_lung_fibrosis_gse250346/zs_stage'
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
o = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r').obs
A = pd.read_csv(W / 'he_region/region_pairs.csv'); A['gain'] = A.auc_s - A.auc_z
G = A.groupby('cell_type').agg(gain=('gain', 'mean'), n_pairs=('gain', 'size'), auc_z=('auc_z', 'mean'), auc_s=('auc_s', 'mean'))
F = np.flatnonzero(o.sample_affect.astype(str).to_numpy() != 'Unaffected'); sub = np.sort(np.random.default_rng(0).choice(F, 200000, replace=False))
ct = o.cell_type.astype(str).to_numpy()
SH = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r')[sub]; S = np.load(MD / 'salient.npy', mmap_mode='r')[sub]
G['shift_L1'] = pd.Series(np.abs(SH).sum(1)).groupby(ct[sub]).mean()
Z = pd.read_csv(W / 'zs_stage.csv').set_index('cell_type'); G['s_disease'] = Z.A_cell_s; G['s_minus_z_disease'] = Z.A_cell_s - Z.A_cell_z
cf = pd.crosstab(o['sample'].astype(str).to_numpy()[F], ct[F], normalize='index')
pth = pd.to_numeric(o.percent_pathology, errors='coerce').groupby(o['sample'].astype(str)).first().reindex(cf.index)
G['path_rho'] = pd.Series({c: spearmanr(cf[c], pth).statistic for c in cf.columns})
G = G.sort_values('gain', ascending=False); G.round(3).to_csv(W / 'he_region/celltype_gain_vs_salient.csv')
pd.set_option('display.width', 250); print(G.round(3).to_string())
t = G[G.n_pairs >= 2].dropna()
print(f'\ncell types with >= 2 pairs: {len(t)}')
for m in ['shift_L1', 's_disease', 's_minus_z_disease', 'path_rho']:
    r = spearmanr(t.gain, t[m] if m != 'path_rho' else t[m].abs()); print(f'  Spearman(s gain on H&E regions, {m}{" (abs)" if m=="path_rho" else ""}) = {r.statistic:+.2f}  p {r.pvalue:.2g}')
