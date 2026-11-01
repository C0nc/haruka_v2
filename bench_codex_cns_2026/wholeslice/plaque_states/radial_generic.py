#!/usr/bin/env python3
"""Plaque-centred z-vs-s gradient for ANY whole-slice fit, scored identically.
usage: radial_generic.py <variant>   (codex_wholeslice_<variant>.h5ad + its Haruka fit)

Only the 'disease' arm of that fit is scored (where s is used by the decoder).
Grey matter; features centred within donor x cell type; tests within cell type.
  eta2      distance-bin eta^2 per donor (bins 0-10/10-20/20-40/40-60/60-100/>100 um)
  rho_half  within-donor half-slice CV (train left half -> test right half and back),
            ridge on log distance (cap 150 um), Spearman; comparable across ALL fits,
            including the one-donor control-vs-control fit
  protein   the same tests on the fit's input proteins (the baseline)
Distance = nearest official Abeta object of either class.
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import sys, warnings
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
V = sys.argv[1]; PX = 0.3774
W = Path(f'{_HR}/bench_codex_cns_2026/wholeslice')
import os
M = Path(os.environ.get('MODEL_DIR', f'{_HR}/results_codex_wholeslice/codex_wholeslice_{V}/'
         'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'))
TAG = os.environ.get('TAG', '')
SRC = Path(f'{_HR}/bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks/marker_intensities')
FILES = {'AD-1': '11162021_Bahareh-3155_EDTA_abeta.csv', 'AD-2': '10202021_BaharehNP_2997-ADCortex_EDTA_abeta.csv',
         'AD-3': '10122021_BaharehNP_ADCortex_EDTA_abeta.csv', 'AD-4': '12032021-Bahareh_3280_Alz_abeta.csv',
         'Ctl-3': '10122021_BaharehNP_1-CtlCortex1796_EDTA_MKT6_abeta.csv'}
BINS = [0, 10, 20, 40, 60, 100, np.inf]
a = ad.read_h5ad(W / f'codex_wholeslice_{V}.h5ad'); o = a.obs
assert (np.load(M / 'obs_names.npy', allow_pickle=True) == a.obs_names.to_numpy()).all()
don = o.donor_id.astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy(); xy = a.obsm['spatial'].astype(float)
dist = np.full(len(o), np.nan)
for d, f in FILES.items():
    m = don == d
    if m.any():
        p = pd.read_csv(SRC / f, usecols=['x', 'y']).to_numpy(float)
        dist[m] = NearestNeighbors(n_neighbors=1).fit(p).kneighbors(xy[m])[0][:, 0] * PX
X = {'z': np.load(M / 'background.npy'), 's': np.load(M / 'salient.npy'), 'protein': np.asarray(a.X)}
for k in X:
    x = X[k].astype(np.float64).copy()
    for d in np.unique(don):
        for t in np.unique(ct):
            m = (don == d) & (ct == t); x[m] -= x[m].mean(0)
    X[k] = x
ARM_ALL = len(sys.argv) > 2 and sys.argv[2] == 'all'          # score every donor with plaques, either arm
use = (o.region.to_numpy() == 'GM') & np.isfinite(dist) & (ARM_ALL | (o.condition.astype(str).to_numpy() == 'disease'))
lab = np.asarray(pd.cut(dist, BINS)).astype(str); ld = np.log1p(np.minimum(dist, 150))

def eta2(x, l):
    xs = StandardScaler().fit_transform(x); gm = xs.mean(0)
    return sum((l == g).sum() * ((xs[l == g].mean(0) - gm) ** 2).sum() for g in np.unique(l)) / ((xs - gm) ** 2).sum()

def half_rho(x, m):
    left = xy[:, 0] < np.median(xy[m, 0]); out = []
    for tr, te in [(m & left, m & ~left), (m & ~left, m & left)]:
        sc = StandardScaler().fit(x[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[tr]), ld[tr])
        out.append(spearmanr(r.predict(sc.transform(x[te])), ld[te]).statistic)
    return np.mean(out)

rows = []
for d in sorted(set(don[use])):
    for t in ['Neuron', 'Astrocyte', 'Microglia']:
        m = use & (don == d) & (ct == t)
        if m.sum() < 300: continue
        for k, x in X.items():
            rows.append(dict(fit=V, donor=d, cell_type=t, rep=k, n=int(m.sum()), eta2=eta2(x[m], lab[m]), rho_half=half_rho(x, m)))
R = pd.DataFrame(rows); R.to_csv(f'radial_generic_{V}{"_allarms" if ARM_ALL else ""}{TAG}.csv', index=False)
P = R.pivot_table(index=['cell_type', 'donor'], columns='rep', values=['eta2', 'rho_half'])
print(f'== {V} (disease arm, GM) =='); print(P.round(4).to_string())
print('\nmean over donors:'); print(R.groupby(['cell_type', 'rep'])[['eta2', 'rho_half']].mean().unstack().round(4).to_string())
