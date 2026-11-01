#!/usr/bin/env python3
"""Lung fibrosis (GSE250346), fit Unaffected vs all fibrosis: what does s encode on fibrosis (condition) cells?
1  variance partition: share of the variance of standardised z / s explained by each annotation alone
   (one-hot OLS on a 200k fibrosis-cell subsample; percent_pathology as a single numeric column).
2  s axes: PCA of s on fibrosis cells; per PC: variance share, Spearman with % pathology at core level,
   cell types with the most extreme mean, niche (CNiche / TNiche) means, and the top genes correlated with the
   PC *within cell type* (gene log1p-CP10k and PC both centred per cell type, so cell identity is removed).
3  decoded meaning: salient composition shift D([z,s]) - D([z,0]) (k=20 neighbour cell-type shares):
   mean shift per neighbour type over fibrosis cells; per core mean shift vs % pathology; and compared with the
   observed per-type change in k=20 neighbour share fibrosis vs Unaffected.
Niches are named by their top-3 cell types.
"""
import os, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import json, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/what_is_s'; OUT.mkdir(exist_ok=True)
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
log = open(OUT / 'what_is_s.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
stage = o.sample_affect.astype(str).to_numpy(); F = np.flatnonzero(stage != 'Unaffected'); U = np.flatnonzero(stage == 'Unaffected')
rng = np.random.default_rng(0); sub = np.sort(rng.choice(F, 200000, replace=False))
ob = o.iloc[sub]; ct = ob.cell_type.astype(str).to_numpy()
Z = np.load(MD / 'background.npy', mmap_mode='r'); S = np.load(MD / 'salient.npy', mmap_mode='r')
L = {'z': StandardScaler().fit_transform(np.asarray(Z[sub])), 's': StandardScaler().fit_transform(np.asarray(S[sub]))}

# niche names
def niche_names(col):
    t = pd.crosstab(o[col].astype(str), o.cell_type.astype(str), normalize='index')
    return {n: ' / '.join(f'{c} {v:.0%}' for c, v in r.sort_values(ascending=False).head(3).items()) for n, r in t.iterrows()}
NN = {c: niche_names(c) for c in ['CNiche', 'TNiche']}
for c in NN:
    P(f'\n{c} composition (all cells):')
    for k in sorted(NN[c], key=lambda x: int(x[1:])): P(f'  {k}: {NN[c][k]}')

# 1 variance partition
def r2(X, f):
    if f == 'percent_pathology':
        D = pd.to_numeric(ob[f], errors='coerce').to_numpy()[:, None]; D = np.c_[np.ones(len(D)), D - np.nanmean(D)]
    else:
        D = pd.get_dummies(ob[f].astype(str)).to_numpy(float)
    B, *_ = np.linalg.lstsq(D, X, rcond=None); res = X - D @ B
    return 1 - (res ** 2).sum() / (X ** 2).sum()
facs = ['cell_type', 'lineage', 'CNiche', 'TNiche', 'sample', 'donor', 'run', 'sample_affect', 'percent_pathology']
VP = pd.DataFrame({k: {f: r2(X, f) for f in facs} for k, X in L.items()}).round(3)
P('\n== 1. variance of standardised latent explained by each annotation alone (fibrosis cells, 200k) =='); P(VP.to_string())
VP.to_csv(OUT / 'variance_partition.csv')

# 2 s axes
pca = PCA(10, random_state=0).fit(L['s']); PC = pca.transform(L['s'])
P('\n== 2. s PCA on fibrosis cells: variance share', np.round(pca.explained_variance_ratio_, 3).tolist())
X = a.layers['counts'][sub] if False else None
Xr = a[sub].layers['counts']; Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
Xn = np.log1p(Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e4); genes = np.asarray(a.var_names).astype(str)
def centre(M, g):
    M = M.astype(np.float64).copy()
    for c in np.unique(g): i = g == c; M[i] -= M[i].mean(0)
    return M
Xc = centre(Xn, ct); PCc = centre(PC, ct)
core = ob['sample'].astype(str).to_numpy(); path = pd.to_numeric(ob.percent_pathology, errors='coerce').to_numpy()
rows = []
for j in range(6):
    p = PC[:, j]; pc = PCc[:, j]
    cm = pd.DataFrame({'p': p, 'c': core, 'y': path}).groupby('c').mean()
    rho = spearmanr(cm.p, cm.y).statistic
    tm = pd.Series(p).groupby(ct).mean().sort_values()
    nm = pd.Series(p).groupby(ob.CNiche.astype(str).to_numpy()).mean().sort_values()
    r = (Xc * pc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (pc ** 2).sum() + 1e-12)
    up, dn = genes[np.argsort(-r)[:10]], genes[np.argsort(r)[:10]]
    w = 1 - ((PC[:, j] - PCc[:, j]) ** 2).sum() / ((PC[:, j] - PC[:, j].mean()) ** 2).sum()
    P(f'\n s-PC{j+1} ({pca.explained_variance_ratio_[j]:.1%}): within-cell-type share of PC variance {w:.2f} | core-level rho with % pathology {rho:+.2f}')
    P('   high cell types:', ', '.join(f'{k} {v:+.2f}' for k, v in tm.tail(4)[::-1].items()))
    P('   low  cell types:', ', '.join(f'{k} {v:+.2f}' for k, v in tm.head(4).items()))
    P(f'   high CNiche: {nm.index[-1]} {nm.iloc[-1]:+.2f} ({NN["CNiche"][nm.index[-1]]}) | low: {nm.index[0]} {nm.iloc[0]:+.2f} ({NN["CNiche"][nm.index[0]]})')
    P('   within-type genes up  :', ', '.join(f'{g} {r[genes.tolist().index(g)]:+.2f}' for g in up))
    P('   within-type genes down:', ', '.join(f'{g} {r[genes.tolist().index(g)]:+.2f}' for g in dn))
    rows.append(dict(pc=j + 1, var=pca.explained_variance_ratio_[j], within_type_share=w, rho_path=rho,
                     high_types=';'.join(tm.tail(4).index[::-1]), low_types=';'.join(tm.head(4).index),
                     genes_up=';'.join(up), genes_down=';'.join(dn)))
pd.DataFrame(rows).to_csv(OUT / 's_pcs.csv', index=False)

# same within-type gene question for the whole of s vs z: how much within-type expression variance is linearly predictable
def within_r2(M):
    Mc = centre(M, ct); B, *_ = np.linalg.lstsq(np.c_[Mc], Xc, rcond=None); res = Xc - Mc @ B
    return 1 - (res ** 2).sum(0) / (Xc ** 2).sum(0)
Rz, Rs = within_r2(L['z']), within_r2(L['s'])
g = pd.DataFrame({'gene': genes, 'R2_z': Rz, 'R2_s': Rs}); g['s_minus_z'] = g.R2_s - g.R2_z
g.to_csv(OUT / 'within_type_gene_r2.csv', index=False)
P(f'\n within-cell-type expression variance explained (median over genes): z {np.median(Rz):.3f}  s {np.median(Rs):.3f}')
P('   genes where s explains most more than z:', ', '.join(f'{r.gene} {r.R2_s:.2f}/{r.R2_z:.2f}' for r in g.sort_values('s_minus_z').tail(12)[::-1].itertuples()))

# 3 decoded shift
SH = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r')
idxF = np.sort(rng.choice(F, 400000, replace=False)); sh = np.asarray(SH[idxF])
mean_shift = pd.Series(sh.mean(0), index=types)
# observed: k=20 neighbour share, fibrosis vs unaffected (from the target the model trains on is not saved; use background/condition composition? -> recompute observed)
from sklearn.neighbors import NearestNeighbors
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]; sl = o['sample'].astype(str).to_numpy()
cti = pd.Categorical(o.cell_type.astype(str), categories=types).codes
obs = np.zeros((len(o), len(types)), np.float32)
for s_ in np.unique(sl):
    i = np.flatnonzero(sl == s_); nn = NearestNeighbors(n_neighbors=min(21, len(i))).fit(xy[i]); _, nb = nn.kneighbors(xy[i])
    obs[i] = np.stack([np.bincount(cti[i][r[1:]], minlength=len(types)) for r in nb]) / (nb.shape[1] - 1)
obs_diff = pd.Series(obs[F].mean(0) - obs[U].mean(0), index=types)
# matched on own cell type: mean over own types of (fibrosis - unaffected) neighbour share
ctall = o.cell_type.astype(str).to_numpy(); md = []
for c in types:
    f_ = (ctall == c) & (stage != 'Unaffected'); u_ = (ctall == c) & (stage == 'Unaffected')
    if f_.sum() > 200 and u_.sum() > 200: md.append((f_.sum(), obs[f_].mean(0) - obs[u_].mean(0)))
w = np.array([m[0] for m in md]); obs_matched = pd.Series((np.stack([m[1] for m in md]) * w[:, None]).sum(0) / w.sum(), index=types)
T3 = pd.DataFrame({'s_shift': mean_shift, 'observed_fib_minus_unaff': obs_diff, 'observed_same_type': obs_matched})
T3.to_csv(OUT / 'composition_shift.csv')
P('\n== 3. decoded composition shift D([z,s]) - D([z,0]) on fibrosis cells (k=20 neighbour share) ==')
P(f'   r(s shift, observed fibrosis - unaffected) = {np.corrcoef(T3.s_shift, T3.observed_fib_minus_unaff)[0,1]:+.2f};'
  f' same-own-type version {np.corrcoef(T3.s_shift, T3.observed_same_type)[0,1]:+.2f}')
P(T3.sort_values('s_shift').round(4).iloc[list(range(8)) + list(range(-8, 0))].to_string())
# per core: magnitude of shift and its main axis vs pathology
cF = sl[idxF]; cm = pd.DataFrame(sh, columns=types).groupby(cF).mean()
pth = pd.to_numeric(o.percent_pathology, errors='coerce').groupby(sl).first().reindex(cm.index)
rho_t = pd.Series({t: spearmanr(cm[t], pth).statistic for t in types}).sort_values()
P('\n   per-core mean shift vs % pathology (35 cores), most negative / positive neighbour types:')
P('   ', ', '.join(f'{k} {v:+.2f}' for k, v in rho_t.head(5).items())); P('   ', ', '.join(f'{k} {v:+.2f}' for k, v in rho_t.tail(5)[::-1].items()))
P(f'   |shift| L1 per core vs % pathology: rho {spearmanr(cm.abs().sum(1), pth).statistic:+.2f}')
rho_t.to_csv(OUT / 'core_shift_vs_pathology.csv')
P('done')
