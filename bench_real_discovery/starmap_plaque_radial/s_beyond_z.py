#!/usr/bin/env python3
"""What does s add beyond z at plaques?  Within each cell type (AD cells, features centred
within sample x fine type [x tissue domain for STARmap]):
  a_z, a_s = in-sample ridge projections of z and s onto log distance to Abeta
  r        = a_s residualised on a_z (the part of s's plaque axis that z does not carry)
  report   features most correlated with r (negative = higher near plaque), and how much of
           the s axis is unique: 1 - corr(a_s, a_z)^2
usage: s_beyond_z.py starmap | codex"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import sys, warnings
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
D = sys.argv[1]
if D == 'starmap':
    MD = (f'{_HR}/results_real_compgated_lambda10_samplebatch_sepheads/ad/'
          'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
    a = ad.read_h5ad(f'{_HR}/bench_real_discovery/data/ad.h5ad'); o = a.obs
    C = a.layers['counts']; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
    L = C.multiply(1e4 / np.asarray(C.sum(1)).ravel()[:, None]).tocsr(); L.data = np.log1p(L.data); F = L.toarray()
    CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
             'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
    fine = o.ct_top.astype(str).to_numpy(); major = pd.Series(fine).map(CROSS).to_numpy()
    grp = o.slice_id.astype(str) + '|' + o.ct_top.astype(str) + '|' + o.tissue_domain.astype(str)
    Dt = pd.read_csv(f'{_HR}/final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
    dist = np.log1p(np.minimum(pd.Series(a.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float), 16 * 31.4))
    use = (o.condition.astype(str) == 'disease').to_numpy(); TYPES = ['Neuron', 'Astrocyte', 'Oligodendrocyte', 'Microglia', 'Vascular']
    detect = lambda m: (F[m] > 0).mean(0) > 0.05
else:
    MD = (f'{_HR}/results_codex_wholeslice/codex_wholeslice_all7_no_oligo/'
          'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
    a = ad.read_h5ad(f'{_HR}/bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
    F = np.asarray(a.X, dtype=np.float64); major = o.cell_type.astype(str).to_numpy()
    grp = o.donor_id.astype(str) + '|' + o.cell_type.astype(str)
    dist = np.log1p(np.minimum(np.load(f'{_HR}/bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy'), 150))
    use = ((o.condition.astype(str) == 'disease') & (o.region.astype(str) == 'GM')).to_numpy(); TYPES = ['Neuron', 'Astrocyte', 'Microglia']
    detect = lambda m: np.ones(F.shape[1], bool)
names = np.array(a.var_names)
Z = np.load(f'{MD}/background.npy').astype(np.float64); S = np.load(f'{MD}/salient.npy').astype(np.float64)
g = grp.to_numpy()
for x in (F, Z, S):
    for k in np.unique(g):
        m = g == k; x[m] -= x[m].mean(0)
rows = []
for t in TYPES:
    m = use & (major == t) & np.isfinite(dist)
    proj = {}
    for k, x in [('z', Z), ('s', S)]:
        sc = StandardScaler().fit(x[m]); proj[k] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[m]), dist[m]).predict(sc.transform(x[m]))
    az, as_ = proj['z'], proj['s']
    r = as_ - np.polyval(np.polyfit(az, as_, 1), az)
    ok = detect(m)
    rho_r = pd.Series([spearmanr(r, F[m][:, j]).statistic for j in np.where(ok)[0]], index=names[ok]).sort_values()
    rho_z = pd.Series([spearmanr(az, F[m][:, j]).statistic for j in np.where(ok)[0]], index=names[ok])
    uniq = 1 - np.corrcoef(az, as_)[0, 1] ** 2
    print(f'{D} {t:15s} n={m.sum():6d}  corr(s-axis, z-axis)={np.corrcoef(az, as_)[0,1]:.2f}  unique share of s axis {uniq:.2f}  '
          f'rho(residual, distance)={spearmanr(r, dist[m]).statistic:+.3f}')
    print('   s-beyond-z, higher NEAR plaque: ' + ', '.join(f'{n} {v:+.2f} (z-axis {rho_z[n]:+.2f})' for n, v in rho_r[:8].items()))
    print('   s-beyond-z, higher FAR:         ' + ', '.join(f'{n} {v:+.2f}' for n, v in rho_r[::-1][:5].items()))
    for n, v in rho_r.items(): rows.append(dict(dataset=D, cell_type=t, feature=n, rho_s_beyond_z=v, rho_z_axis=rho_z[n]))
pd.DataFrame(rows).to_csv(f's_beyond_z_{D}.csv', index=False)
