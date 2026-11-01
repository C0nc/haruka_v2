#!/usr/bin/env python3
"""Exercise cells, per cell TE = clock age - own age. In-sample: ONE ridge over all exercise cells, z or s -> TE.
Reports R^2 and Pearson r (pooled; within cell type), and the same after removing per-mouse means from TE and latents
(so mouse-to-mouse differences cannot contribute)."""
import os, warnings; os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str)
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
te = C.predicted_age.to_numpy() - o.age.astype(float).to_numpy()
E = np.flatnonzero((o.condition.astype(str) == 'exercise').to_numpy() & np.isfinite(te))
y0 = te[E]; ms = o.mouse_id.astype(str).to_numpy()[E]; cts = o.celltype.astype(str).to_numpy()[E]
L = {'z': np.load(MD / 'background.npy')[E].astype(np.float64), 's': np.load(MD / 'salient.npy')[E].astype(np.float64)}
def cen(X, g):
    X = np.asarray(X, float).copy()
    for u in np.unique(g): X[g == u] -= X[g == u].mean(0)
    return X
rows = []
for mode in ['as is', 'mouse means removed']:
    y = y0 if mode == 'as is' else cen(y0[:, None], ms).ravel()
    for k, X in L.items():
        X = X if mode == 'as is' else cen(X, ms)
        Xs = StandardScaler().fit_transform(X); p = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xs, y).predict(Xs)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(); r = pearsonr(p, y)[0]
        wr = {c: pearsonr(p[cts == c], y[cts == c])[0] for c in np.unique(cts) if (cts == c).sum() >= 500}
        wr2 = {c: pearsonr(p[cts == c], y[cts == c])[0] ** 2 for c in wr}
        rows.append(dict(mode=mode, latent=k, R2=r2, pearson=r, within_type_median_r=np.median(list(wr.values())),
                         within_type_median_r2=np.median(list(wr2.values())), **{f'r_{c}': v for c, v in wr.items()}))
T = pd.DataFrame(rows); T.to_csv('te_insample.csv', index=False)
pd.set_option('display.width', 250)
print(T[['mode', 'latent', 'R2', 'pearson', 'within_type_median_r', 'within_type_median_r2']].round(3).to_string(index=False))
W = T.set_index(['mode', 'latent'])[[c for c in T.columns if c.startswith('r_')]].T
W.index = [i[2:] for i in W.index]; print('\nwithin-cell-type Pearson r:'); print(W.round(3).to_string())
for mode in ['as is', 'mouse means removed']:
    print(f'{mode}: s > z within type in {int((W[(mode, "s")] > W[(mode, "z")]).sum())}/{len(W)}')
