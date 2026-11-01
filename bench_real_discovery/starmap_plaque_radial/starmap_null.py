#!/usr/bin/env python3
"""STARmap: Abeta-distance decoding, real vs shifted plaques, with a denoised (scVI) baseline.
Features centred within slice x fine type x tissue domain (anatomy controlled).  Cross-slice
ridge decoding (train r1 -> test r2 and back), Spearman.  Shifted null: the Abeta-associated
cell positions translated by a random offset (toroidal wrap in the slice bounding box)."""
import warnings
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
import os
MD = os.environ.get('MODEL_DIR', '/home/yancui/haruka_v2/results_real_compgated_lambda10_samplebatch_sepheads/ad/'
      'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
NSHIFT = int(os.environ.get('NSHIFT', '5')); TAG = os.environ.get('TAG', '')
a = ad.read_h5ad('/home/yancui/haruka_v2/bench_real_discovery/data/ad.h5ad'); o = a.obs
CROSS = {'CTX-Ex': 'Neuron', 'CA1': 'Neuron', 'CA2': 'Neuron', 'CA3': 'Neuron', 'DG': 'Neuron', 'Inh': 'Neuron', 'LHb': 'Neuron',
         'Astro': 'Astrocyte', 'Oligo': 'Oligodendrocyte', 'OPC': 'Oligodendrocyte', 'Micro': 'Microglia', 'Endo': 'Vascular', 'SMC': 'Vascular'}
fine = o.ct_top.astype(str).to_numpy(); major = pd.Series(fine).map(CROSS).to_numpy()
sl = o.slice_id.astype(str).to_numpy(); dom = o.tissue_domain.astype(str).to_numpy(); xy = a.obsm['spatial'].astype(float)
assert (np.load('scvi32_obs_names.npy', allow_pickle=True) == a.obs_names.to_numpy()).all()
C = a.layers['counts']; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
L = C.multiply(1e4 / np.asarray(C.sum(1)).ravel()[:, None]).tocsr(); L.data = np.log1p(L.data); Ld = L.toarray()
AD = ['disease_r1', 'disease_r2']; dis = np.isin(sl, AD)
X = {'z': np.load(f'{MD}/background.npy'), 's': np.load(f'{MD}/salient.npy'), 'scVI32': np.load('scvi32.npy'),
     'expr HVG500': Ld[:, np.argsort(Ld[dis].var(0))[::-1][:500]]}
for k in X:
    x = X[k].astype(np.float64).copy()
    for s_ in np.unique(sl):
        for t in np.unique(fine):
            for g in np.unique(dom):
                m = (sl == s_) & (fine == t) & (dom == g)
                if m.any(): x[m] -= x[m].mean(0)
    X[k] = x
S = 31.4; ab = o.plaque_abeta.to_numpy() == 1
rng = np.random.default_rng(0)

def dist(shift):
    d = np.full(len(o), np.nan)
    for s_ in AD:
        m = sl == s_; p = xy[m & ab]; lo, hi = xy[m].min(0), xy[m].max(0)
        if shift is not None: p = lo + np.mod(p - lo + shift[s_] * (hi - lo), hi - lo)
        d[m] = NearestNeighbors(n_neighbors=1).fit(p).kneighbors(xy[m])[0][:, 0]
    return np.log1p(np.minimum(d, 16 * S))

rows = []
for it in range(NSHIFT + 1):
    ld = dist(None if it == 0 else {s_: rng.uniform(0.2, 0.8, 2) for s_ in AD})
    for t in ['Neuron', 'Astrocyte', 'Oligodendrocyte', 'Microglia', 'Vascular']:
        m = dis & (major == t)
        for k, x in X.items():
            r = []
            for a_, b_ in [(AD[0], AD[1]), (AD[1], AD[0])]:
                tr, te = m & (sl == a_), m & (sl == b_)
                sc = StandardScaler().fit(x[tr]); mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[tr]), ld[tr])
                r.append(spearmanr(mdl.predict(sc.transform(x[te])), ld[te]).statistic)
            rows.append(dict(kind='real' if it == 0 else 'shifted', it=it, cell_type=t, rep=k, rho=np.mean(r)))
    print('done', it, flush=True)
R = pd.DataFrame(rows); R.to_csv(f'starmap_null_scvi{TAG}.csv', index=False)
T = R.groupby(['kind', 'cell_type', 'rep']).rho.mean().unstack()
print(T.round(3).to_string())
if NSHIFT: print('\nreal minus shifted:'); print((T.loc['real'] - T.loc['shifted']).round(3).to_string())
if NSHIFT: print('\nshifted SD across shifts:'); print(R[R.kind == 'shifted'].groupby(['cell_type', 'rep']).rho.std().unstack().round(3).to_string())
