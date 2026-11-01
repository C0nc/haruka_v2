#!/usr/bin/env python3
"""Calibrated null for the radial z-vs-s test: the same plaque set, translated by a random
offset (toroidal wrap inside the donor's cell bounding box), keeps plaque number, size of
the set and its clustering but breaks its link to the tissue.  If s still beats z on fake
plaques, 's > z' reflects spatially smooth variance in s, not plaque biology.
usage: radial_null.py <variant> [n_shifts]"""
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
V = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 5; PX = 0.3774
W = Path(f'{_HR}/bench_codex_cns_2026/wholeslice')
M = Path(f'{_HR}/results_codex_wholeslice/codex_wholeslice_{V}/'
         'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
SRC = Path(f'{_HR}/bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks/marker_intensities')
FILES = {'AD-1': '11162021_Bahareh-3155_EDTA_abeta.csv', 'AD-2': '10202021_BaharehNP_2997-ADCortex_EDTA_abeta.csv',
         'AD-3': '10122021_BaharehNP_ADCortex_EDTA_abeta.csv', 'AD-4': '12032021-Bahareh_3280_Alz_abeta.csv',
         'Ctl-3': '10122021_BaharehNP_1-CtlCortex1796_EDTA_MKT6_abeta.csv'}
a = ad.read_h5ad(W / f'codex_wholeslice_{V}.h5ad'); o = a.obs
don = o.donor_id.astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy(); xy = a.obsm['spatial'].astype(float)
X = {'z': np.load(M / 'background.npy'), 's': np.load(M / 'salient.npy'), 'protein': np.asarray(a.X)}
for k in X:
    x = X[k].astype(np.float64).copy()
    for d in np.unique(don):
        for t in np.unique(ct):
            m = (don == d) & (ct == t); x[m] -= x[m].mean(0)
    X[k] = x
use = (o.region.to_numpy() == 'GM') & (o.condition.astype(str).to_numpy() == 'disease')
rng = np.random.default_rng(0)

def dist_for(shift):
    dd = np.full(len(o), np.nan)
    for d, f in FILES.items():
        m = don == d
        if not m.any(): continue
        p = pd.read_csv(SRC / f, usecols=['x', 'y']).to_numpy(float)
        lo, hi = xy[m].min(0), xy[m].max(0)
        if shift is not None: p = lo + np.mod(p - lo + shift[d] * (hi - lo), hi - lo)
        dd[m] = NearestNeighbors(n_neighbors=1).fit(p).kneighbors(xy[m])[0][:, 0] * PX
    return dd

def half_rho(x, m, ld):
    left = xy[:, 0] < np.median(xy[m, 0]); out = []
    for tr, te in [(m & left, m & ~left), (m & ~left, m & left)]:
        sc = StandardScaler().fit(x[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[tr]), ld[tr])
        out.append(spearmanr(r.predict(sc.transform(x[te])), ld[te]).statistic)
    return np.mean(out)

rows = []
for it in range(NS + 1):
    shift = None if it == 0 else {d: rng.uniform(0.2, 0.8, 2) for d in FILES}
    ld = np.log1p(np.minimum(dist_for(shift), 150))
    for d in sorted(set(don[use])):
        for t in ['Neuron', 'Astrocyte', 'Microglia']:
            m = use & (don == d) & (ct == t)
            for k, x in X.items():
                rows.append(dict(kind='real' if it == 0 else 'shifted', it=it, donor=d, cell_type=t, rep=k, rho=half_rho(x, m, ld)))
    print('done', it, flush=True)
R = pd.DataFrame(rows); R.to_csv(f'radial_null_{V}.csv', index=False)
T = R.groupby(['kind', 'cell_type', 'rep']).rho.mean().unstack(); print(T.round(3).to_string())
print('\nexcess over shifted null (real - mean shifted):')
E = T.loc['real'] - T.loc['shifted']; print(E.round(3).to_string())
print('\ns - z: real', (T.loc['real', 's'] - T.loc['real', 'z']).round(3).to_dict(), '| shifted', (T.loc['shifted', 's'] - T.loc['shifted', 'z']).round(3).to_dict())
