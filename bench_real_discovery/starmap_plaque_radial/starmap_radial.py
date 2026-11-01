#!/usr/bin/env python3
"""STARmap (TauPS2APP 13 mo) version of the plaque-centred z-vs-s test.

Default Haruka fit (composition-gated, lambda 10, decoder-only slice batch).  The control
slice has no Abeta or ptau, so here z's background contains NO plaque by construction.
Distance = official table, to the nearest Abeta- / ptau-associated cell, in spatial units;
units are undocumented, so bins are multiples of the median nearest-neighbour cell
spacing (S).  Features are centred within slice x fine cell type; tests within major type.
  A  eta^2 of Abeta-distance bin, z / s / expression
  B  held-out-slice decoding of log distance (train r1 -> test r2 and back), Abeta and ptau
  C  'which process': cells within 2S of Abeta only vs ptau only, held-out-slice AUROC
  D  genes most correlated with the decoded Abeta axis of z and of s
Expression baselines = log1p(CP10k): PCA 32 of all 2,766 genes (same width as z, s) and the 500 most
variable genes in AD cells (the stronger baseline; all 2,766 was too slow for the logistic fits).
"""
import warnings
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import sys, os
SEED = 0
BY_DOMAIN = os.environ.get('BY_DOMAIN', '0') == '1'   # also centre within tissue domain (anatomy control)
MD = sys.argv[1] if len(sys.argv) > 1 else ('/home/yancui/haruka_v2/results_real_compgated_lambda10_samplebatch_sepheads/ad/'
                                            'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
a = ad.read_h5ad('/home/yancui/haruka_v2/bench_real_discovery/data/ad.h5ad'); o = a.obs
on = np.load(f'{MD}/obs_names.npy', allow_pickle=True); assert (on == a.obs_names.to_numpy()).all()
CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
         'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
fine = o.ct_top.astype(str).to_numpy(); major = pd.Series(fine).map(CROSS).to_numpy()
sl = o.slice_id.astype(str).to_numpy(); xy = a.obsm['spatial'].astype(float)
Dt = pd.read_csv('/home/yancui/haruka_v2/final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
dab = pd.Series(a.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
dtau = pd.Series(a.obs_names).map(Dt.distance_to_ptau_associated_cell_spatial_units).to_numpy(float)
S = np.median([np.median(NearestNeighbors(n_neighbors=2).fit(xy[sl == s]).kneighbors(xy[sl == s])[0][:, 1]) for s in np.unique(sl)])
print(f'BY_DOMAIN={BY_DOMAIN}; median NN cell spacing S = {S:.1f} units; Abeta-assoc cells {int((dab == 0).sum())}, ptau-assoc {int((dtau == 0).sum())}')
C = a.layers['counts'] if 'counts' in a.layers else a.X
C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
L = C.multiply(1e4 / np.asarray(C.sum(1)).ravel()[:, None]).tocsr(); L.data = np.log1p(L.data)
Ld = L.toarray(); E = PCA(32, random_state=SEED).fit_transform(Ld)
HV = Ld[:, np.argsort(Ld[np.isin(o.slice_id.astype(str).to_numpy(), ['disease_r1', 'disease_r2'])].var(0))[::-1][:500]]   # 500 most variable genes in AD cells
X = {'z': np.load(f'{MD}/background.npy'), 's': np.load(f'{MD}/salient.npy'), 'expr PCA32': E, 'expr HVG500': HV}
dom = o.tissue_domain.astype(str).to_numpy() if BY_DOMAIN else np.full(len(o), 'all')
for k in X:
    x = X[k].astype(np.float64).copy()
    for s in np.unique(sl):
        for t in np.unique(fine):
            for g in np.unique(dom):
                m = (sl == s) & (fine == t) & (dom == g)
                if m.any(): x[m] -= x[m].mean(0)
    X[k] = x
AD = ['disease_r1', 'disease_r2']; dis = np.isin(sl, AD)
edges = np.array([-0.5, 1e-9, 1, 2, 4, 8, np.inf]) * S; BL = ['0 (assoc)', '<=1S', '1-2S', '2-4S', '4-8S', '>8S']
bins = pd.cut(dab, edges, labels=BL).astype(str)

def eta2(x, lab):
    xs = StandardScaler().fit_transform(x); gm = xs.mean(0)
    return sum((lab == g).sum() * ((xs[lab == g].mean(0) - gm) ** 2).sum() for g in np.unique(lab)) / ((xs - gm) ** 2).sum()

def cross_slice(x, y, m, kind):
    out = []
    for tr_s, te_s in [(AD[0], AD[1]), (AD[1], AD[0])]:
        tr = m & (sl == tr_s); te = m & (sl == te_s)
        if kind == 'clf' and (len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2): continue
        sc = StandardScaler().fit(x[tr])
        if kind == 'reg':
            r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[tr]), y[tr])
            out.append(spearmanr(r.predict(sc.transform(x[te])), y[te]).statistic)
        else:
            c = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED).fit(sc.transform(x[tr]), y[tr])
            out.append(roc_auc_score(y[te], c.predict_proba(sc.transform(x[te]))[:, 1]))
    return out

rows, axes = [], {}
genes = np.array(a.var_names)
for t in ['Neuron', 'Astrocyte', 'Oligodendrocyte', 'Microglia', 'Vascular']:
    m = dis & (major == t)
    near_ab = (dab <= 2 * S) & (dtau > 2 * S); near_tau = (dtau <= 2 * S) & (dab > 2 * S)
    mp = m & (near_ab | near_tau)
    print(f'\n===== {t}: AD cells {m.sum()} | Abeta bins {pd.Series(bins[m]).value_counts().reindex(BL).fillna(0).astype(int).to_dict()} '
          f'| near-Abeta-only {int((m & near_ab).sum())}, near-ptau-only {int((m & near_tau).sum())} =====')
    for k, x in X.items():
        e = eta2(x[m], bins[m])
        rab = cross_slice(x, np.log1p(np.minimum(dab, 16 * S)), m, 'reg')
        rtau = cross_slice(x, np.log1p(np.minimum(dtau, 16 * S)), m, 'reg')
        auc = cross_slice(x, near_ab.astype(int), mp, 'clf') if (m & near_ab).sum() >= 20 and (m & near_tau).sum() >= 20 else []
        rows.append(dict(cell_type=t, rep=k, eta2_abeta=e, rho_abeta=np.mean(rab), rho_ptau=np.mean(rtau),
                         abeta_vs_ptau_auroc=np.mean(auc) if auc else np.nan))
        print(f'  {k:10s} eta2 {e:.4f}  Abeta-distance rho {np.mean(rab):+.3f} ({", ".join(f"{v:+.2f}" for v in rab)})  '
              f'ptau-distance rho {np.mean(rtau):+.3f}  Abeta-vs-ptau AUROC ' + (f'{np.mean(auc):.3f}' if auc else 'n/a'))
        if k in ('z', 's') and t in ('Neuron', 'Astrocyte', 'Microglia', 'Oligodendrocyte'):
            sc = StandardScaler().fit(x[m]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[m]), np.log1p(np.minimum(dab[m], 16 * S)))
            proj = r.predict(sc.transform(x[m])); Lm = L[np.where(m)[0]].toarray()
            ok = (Lm > 0).mean(0) > 0.05
            rho = np.full(len(genes), np.nan)
            rho[ok] = [spearmanr(proj, Lm[:, j]).statistic for j in np.where(ok)[0]]
            axes[(t, k)] = pd.Series(rho, index=genes)
R = pd.DataFrame(rows); R.to_csv(f'starmap_radial_summary{"_bydomain" if BY_DOMAIN else ""}.csv', index=False)
print('\n===== D. genes on the decoded Abeta axis (negative = higher near Abeta) =====')
for (t, k), sr in axes.items():
    sr = sr.dropna().sort_values()
    print(f'  {t:15s} {k}: near ' + ', '.join(f'{g} {v:+.2f}' for g, v in sr[:8].items()) + ' | far ' + ', '.join(f'{g} {v:+.2f}' for g, v in sr[::-1][:4].items()))
pd.DataFrame({f'{t}|{k}': v for (t, k), v in axes.items()}).to_csv(f'starmap_radial_axis_gene_rho{"_bydomain" if BY_DOMAIN else ""}.csv')
