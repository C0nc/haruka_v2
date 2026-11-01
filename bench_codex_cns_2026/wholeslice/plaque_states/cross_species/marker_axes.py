#!/usr/bin/env python3
"""Biology behind the s plaque gradient, and human (CODEX protein) vs mouse (STARmap RNA).

Per dataset, AD cells, within major cell type (Neuron / Astrocyte / Microglia), features and
latents centred within donor|slice x type [x tissue domain for STARmap]:
  grad      within-group Pearson(feature, plaque proximity = -log distance)   raw biology
  load_s    Spearman(feature, held-out s plaque score)                   what s's plaque axis carries
  load_z    Spearman(feature, held-out z plaque score)
  s_only    Spearman(feature, s score residualised on z score)           what s carries beyond z
Held-out scores = per-type ridge on log distance, donor/slice held out (as in the radial tests).
Writes marker_axes_<dataset>.csv (all features) for the cross-species comparison.
"""
import sys, warnings
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
H = '/home/yancui/haruka_v2/'
D = sys.argv[1]


def centre(X, g):
    X = X.astype(np.float64).copy()
    for k in np.unique(g):
        m = g == k; X[m] -= X[m].mean(0)
    return X


def heldout(X, dl, types, gt, mask):
    out = np.full(len(dl), np.nan)
    for t in np.unique(types[mask]):
        for g in np.unique(gt[mask]):
            tr = mask & (types == t) & (gt != g); te = mask & (types == t) & (gt == g)
            sc = StandardScaler().fit(X[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), dl[tr])
            pr = -r.predict(sc.transform(X[te]))
            out[te] = (pr - pr.mean()) / pr.std()        # per held-out donor/slice: removes prediction offsets before pooling
    return out


def within_corr(F, y, g):
    """Pearson of every column of F with y after centring BOTH within group g.
    (Ranking group-centred sparse features recovers group membership, which then correlates
    with an uncentred distance -- an artefact; within-group Pearson avoids it.)"""
    y = y.astype(np.float64).copy(); F = F.copy()
    for k in np.unique(g):
        m = g == k; y[m] -= y[m].mean(); F[m] -= F[m].mean(0)
    y = y / (y.std() + 1e-12); F = F / (F.std(0) + 1e-12)
    return (F * y[:, None]).mean(0)

if D == 'starmap':
    a = ad.read_h5ad(H + 'bench_real_discovery/data/ad.h5ad'); o = a.obs
    MD = H + 'results_real_compgated_lambda10_samplebatch_sepheads/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
    C = a.layers['counts']; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
    L = C.multiply(1e4 / np.asarray(C.sum(1)).ravel()[:, None]).tocsr(); L.data = np.log1p(L.data); F = L.toarray()
    CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
             'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
    typ = pd.Series(o.ct_top.astype(str).to_numpy()).map(CROSS).to_numpy()
    grp = (o.slice_id.astype(str) + '|' + o.ct_top.astype(str) + '|' + o.tissue_domain.astype(str)).to_numpy()
    Dt = pd.read_csv(H + 'final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
    dist = pd.Series(a.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
    dl = np.log1p(np.minimum(dist, 16 * 31.4)); gt = o.slice_id.astype(str).to_numpy()
    use = (o.condition.astype(str) == 'disease').to_numpy() & np.isfinite(dl)
else:
    a = ad.read_h5ad(H + 'bench_codex_cns_2026/wholeslice/codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
    MD = H + 'results_codex_wholeslice/codex_wholeslice_all7_no_oligo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
    F = np.asarray(a.X, dtype=np.float64); typ = o.cell_type.astype(str).to_numpy()
    grp = (o.donor_id.astype(str) + '|' + o.cell_type.astype(str)).to_numpy()
    dist = np.load(H + 'bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy')
    dl = np.log1p(np.minimum(dist, 150)); gt = o.donor_id.astype(str).to_numpy()
    use = ((o.condition.astype(str) == 'disease') & (o.region.astype(str) == 'GM')).to_numpy()
names = np.array(a.var_names)
F = centre(F, grp); Z = centre(np.load(MD + '/background.npy'), grp); S = centre(np.load(MD + '/salient.npy'), grp)
sz, ss = heldout(Z, dl, typ, gt, use), heldout(S, dl, typ, gt, use)
rows = []
for t in ['Neuron', 'Astrocyte', 'Microglia']:
    m = use & (typ == t)
    det = (F[m] != 0).mean(0) if D == 'codex' else ((F[m] + 0) > -1e9).mean(0)
    X = F[m]; prox = -dl[m]
    res = ss[m] - np.polyval(np.polyfit(sz[m], ss[m], 1), sz[m])
    gm_ = grp[m]
    g_, ls, lz, so = (within_corr(X, v, gm_) for v in (prox, ss[m], sz[m], res))
    rho_sz = spearmanr(ss[m], sz[m]).statistic
    print(f'{D} {t:10s} n={m.sum():6d}  held-out rho: s {spearmanr(ss[m], prox).statistic:+.3f}  z {spearmanr(sz[m], prox).statistic:+.3f}  '
          f'| corr(s score, z score) {rho_sz:+.2f}', flush=True)
    for j, n in enumerate(names):
        rows.append(dict(dataset=D, cell_type=t, feature=n, grad=g_[j], load_s=ls[j], load_z=lz[j], s_only=so[j]))
R = pd.DataFrame(rows); R.to_csv(f'marker_axes_{D}.csv', index=False)
