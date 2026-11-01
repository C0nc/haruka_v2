#!/usr/bin/env python3
"""Exercise cells, per-cell TE = clock-predicted age - own age (19 mo). In-sample, inside every cell type x region
stratum (paper's 4 regions; strata >= 200 cells), z vs s:
  A  ridge (32 dims -> TE): R^2 and Pearson r
  B  unsupervised directions, no fitting to TE: PC1 and PC2 of the latent (PCA inside the stratum):
     |Pearson r| of each PC with TE (sign of a PC is arbitrary), and R^2 of TE on [PC1, PC2]
  C  diffusion pseudotime (scanpy dpt; 15-NN graph on the latent, 10 diffusion components, root = the cell at the
     minimum of diffusion component 1, chosen without TE): |Pearson r| and R^2 with TE
     (strata subsampled to <= 8,000 cells for C)
"""
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from pathlib import Path
from scipy.stats import pearsonr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
OUT = Path('/home/yancui/haruka_v2/figures_paper/exercise/exercise_only')
MD = sorted(Path('/project/zhihuanglab/yancui/haruka_exercise/results_w10_s100/exercise_old_full').glob('*/'))[0]
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r'); o = h.obs
names = np.asarray(h.obs_names).astype(str)
C = pd.read_csv('/project/zhihuanglab/yancui/haruka_exercise/clock/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
te = C.predicted_age.to_numpy() - o.age.astype(float).to_numpy()
ct = o.celltype.astype(str).to_numpy(); reg = o.region.astype(str).to_numpy()
E = (o.condition.astype(str) == 'exercise').to_numpy() & np.isfinite(te)
L = {'z': np.load(MD / 'background.npy').astype(np.float64), 's': np.load(MD / 'salient.npy').astype(np.float64)}
rng = np.random.default_rng(0)

def dpt(X):
    a = ad.AnnData(X=X.astype(np.float32)); sc.pp.neighbors(a, n_neighbors=15, use_rep='X', random_state=0)
    sc.tl.diffmap(a, n_comps=10); a.uns['iroot'] = int(np.argmin(a.obsm['X_diffmap'][:, 1]))
    sc.tl.dpt(a, n_dcs=10); return a.obs['dpt_pseudotime'].to_numpy()

rows = []
for (c, r), idx in pd.DataFrame({'c': ct[E], 'r': reg[E]}).groupby(['c', 'r']).indices.items():
    idx = np.flatnonzero(E)[idx]
    if len(idx) < 200: continue
    y = te[idx]; rec = dict(cell_type=c, region=r, n=len(idx), te_mean=y.mean())
    sub = np.sort(rng.choice(len(idx), min(8000, len(idx)), replace=False))
    for k, M in L.items():
        X = StandardScaler().fit_transform(M[idx])
        p = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X, y).predict(X)
        rec[f'ridge_r_{k}'] = pearsonr(p, y)[0]; rec[f'ridge_R2_{k}'] = rec[f'ridge_r_{k}'] ** 2
        pcs = PCA(2).fit_transform(X)
        rec[f'pc1_r_{k}'] = abs(pearsonr(pcs[:, 0], y)[0]); rec[f'pc2_r_{k}'] = abs(pearsonr(pcs[:, 1], y)[0])
        rec[f'pc12_R2_{k}'] = LinearRegression().fit(pcs, y).score(pcs, y)
        t = dpt(X[sub]); ok = np.isfinite(t)
        rec[f'dpt_r_{k}'] = abs(pearsonr(t[ok], y[sub][ok])[0]); rec[f'dpt_R2_{k}'] = rec[f'dpt_r_{k}'] ** 2
    rows.append(rec); print(f'{c} | {r}: done', flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT / 'te_stratum_pc_dpt.csv', index=False)
log = open(OUT / 'te_stratum_pc_dpt.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n')
P(f'\n== exercise cells, in-sample inside {len(T)} cell type x region strata: z vs s vs TE (clock - own age) ==')
for m in ['ridge_r', 'ridge_R2', 'pc1_r', 'pc2_r', 'pc12_R2', 'dpt_r', 'dpt_R2']:
    a_, b_ = T[f'{m}_z'], T[f'{m}_s']; w = wilcoxon(b_, a_).pvalue
    P(f'   {m:9s} median z {a_.median():.3f}  s {b_.median():.3f} | s > z in {int((b_ > a_).sum())}/{len(T)} | Wilcoxon p {w:.2g}')
pd.set_option('display.width', 260)
cols = ['cell_type', 'region', 'n', 'te_mean'] + [f'{m}_{k}' for m in ['ridge_R2', 'pc1_r', 'pc2_r', 'pc12_R2', 'dpt_r'] for k in ['z', 's']]
P(T[cols].sort_values(['cell_type', 'region']).round(3).to_string(index=False))
