#!/usr/bin/env python3
"""Which latent explains each feature?  Cross-validated variance partition of every input
feature into z-only, s-only and joint, within cell type, disease arm, features centred
within sample x fine type.  Then: are plaque-gradient features more s-dependent?

usage: variance_partition.py starmap
       variance_partition.py codex <variant>
unique_s = R2(z+s) - R2(z): what s adds on top of z.  Plaque-gradient features = top 10%
by |Spearman(feature, log distance)| in that cell type (computed on the same cells).
"""
import sys, warnings
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
SEED = 0
if sys.argv[1] == 'starmap':
    MD = ('/home/yancui/haruka_v2/results_real_compgated_lambda10_samplebatch_sepheads/ad/'
          'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
    a = ad.read_h5ad('/home/yancui/haruka_v2/bench_real_discovery/data/ad.h5ad'); o = a.obs
    C = a.layers['counts']; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
    L = C.multiply(1e4 / np.asarray(C.sum(1)).ravel()[:, None]).tocsr(); L.data = np.log1p(L.data); F = L.toarray()
    samp = o.slice_id.astype(str).to_numpy(); fine = o.ct_top.astype(str).to_numpy()
    CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
             'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
    major = pd.Series(fine).map(CROSS).to_numpy()
    Dt = pd.read_csv('/home/yancui/haruka_v2/final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
    dist = pd.Series(a.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
    cap = 16 * 31.4; use = (o.condition.astype(str) == 'disease').to_numpy(); tag = 'starmap'
    keep = (F[use] > 0).mean(0) > 0.05                              # genes detected in >5% of AD cells
    F = F[:, keep]; names = np.array(a.var_names)[keep]
else:
    V = sys.argv[2]
    MD = (f'/home/yancui/haruka_v2/results_codex_wholeslice/codex_wholeslice_{V}/'
          'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
    a = ad.read_h5ad(f'/home/yancui/haruka_v2/bench_codex_cns_2026/wholeslice/codex_wholeslice_{V}.h5ad'); o = a.obs
    F = np.asarray(a.X, dtype=np.float64); names = np.array(a.var_names)
    samp = o.donor_id.astype(str).to_numpy(); fine = major = o.cell_type.astype(str).to_numpy()
    dall = np.load('/home/yancui/haruka_v2/bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy')
    ref = ad.read_h5ad('/home/yancui/haruka_v2/bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad', backed='r')
    dist = pd.Series(dall, index=ref.obs_names).reindex(a.obs_names).to_numpy(float)
    cap = 150.0; use = ((o.condition.astype(str) == 'disease') & (o.region.astype(str) == 'GM')).to_numpy(); tag = f'codex_{V}'
Z = np.load(f'{MD}/background.npy').astype(np.float64); S = np.load(f'{MD}/salient.npy').astype(np.float64)
for x in (F, Z, S):
    for s_ in np.unique(samp):
        for t in np.unique(fine):
            m = (samp == s_) & (fine == t)
            if m.any(): x[m] -= x[m].mean(0)

def cv_r2(Xin, Y):
    pred = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xin):
        sc = StandardScaler().fit(Xin[tr]); pred[te] = Ridge(alpha=10).fit(sc.transform(Xin[tr]), Y[tr]).predict(sc.transform(Xin[te]))
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)

rows = []
for t in sorted(set(major[use])):
    m = use & (major == t) & np.isfinite(dist)
    if m.sum() < 500: continue
    Y = F[m]; ok = Y.std(0) > 0; Y = Y[:, ok]; nm = names[ok]
    rz, rs, rzs = cv_r2(Z[m], Y), cv_r2(S[m], Y), cv_r2(np.hstack([Z[m], S[m]]), Y)
    ld = np.log1p(np.minimum(dist[m], cap))
    grad = np.array([abs(spearmanr(Y[:, j], ld).statistic) for j in range(Y.shape[1])])
    top = grad >= np.nanquantile(grad, 0.9)
    us = rzs - rz; frac = us / np.clip(rzs, 1e-6, None)
    p = mannwhitneyu(frac[top], frac[~top]).pvalue
    print(f'{tag} {t:15s} n={m.sum():6d} features={len(nm):4d} | median R2 z {np.median(rz):.3f} s {np.median(rs):.3f} z+s {np.median(rzs):.3f} '
          f'| s-unique share of explained: plaque-gradient {np.median(frac[top]):.3f} vs others {np.median(frac[~top]):.3f} (MWU p={p:.1e})')
    for j in range(len(nm)):
        rows.append(dict(dataset=tag, cell_type=t, feature=nm[j], r2_z=rz[j], r2_s=rs[j], r2_zs=rzs[j], unique_s=us[j],
                         s_share=frac[j], abs_rho_dist=grad[j], plaque_gradient_top10=bool(top[j])))
R = pd.DataFrame(rows); R.to_csv(f'variance_partition_{tag}.csv', index=False)
for t, g in R.groupby('cell_type'):
    g = g[g.plaque_gradient_top10].sort_values('abs_rho_dist', ascending=False).head(8)
    print(f'  {t}: top plaque-gradient features (R2 z / s / z+s, s-share): ' +
          '; '.join(f'{r.feature} {r.r2_z:.2f}/{r.r2_s:.2f}/{r.r2_zs:.2f} ({r.s_share:.2f})' for r in g.itertuples()))
