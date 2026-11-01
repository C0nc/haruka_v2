#!/usr/bin/env python3
"""What does the KRAS s-CC actually reflect?

s-CC = 3-hop neighbourhood mean of the 32-d salient latent over treated cells only.
Candidate contents: local tissue composition (where you sit relative to the tumour
mass), anatomic compartment, section identity, cell identity, or drug response.
"""
from __future__ import annotations
import numpy as np, pandas as pd, anndata as ad, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, GroupKFold
from scipy.stats import spearmanr

SEED = 0
CC = ('/home/yancui/haruka_v2/results_real_compgated_nb_final/mouselung_kras_xenium_clean_noC3/'
      'haruka_z_s_cellcharter_cc3_condition_only/')
TUMOR = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']

h = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r')
rows = np.load(CC + 'source_rows.npy')
o = h.obs.iloc[rows][['slice_id', 'celltype', 'compartment', 'x_centroid', 'y_centroid']].copy()
h.file.close()
sl = o.slice_id.astype(str).to_numpy(); ct = o.celltype.astype(str).to_numpy()
comp = o.compartment.astype(str).to_numpy()
xy = o[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float64)
is_tum = np.isin(ct, TUMOR)
print(f'treated cells {len(sl)}, tumour {is_tum.sum()}, compartments {sorted(set(comp))}', flush=True)

# local tumour fraction over 20 spatial neighbours, within section
locfrac = np.zeros(len(sl)); nbrs = {}
for s in np.unique(sl):
    i = np.where(sl == s)[0]
    nb = NearestNeighbors(n_neighbors=21).fit(xy[i]).kneighbors(xy[i], return_distance=False)[:, 1:]
    nbrs[s] = (i, nb); locfrac[i] = is_tum[i][nb].mean(1)
print('local tumour fraction computed', flush=True)

def moran(x, idx):
    out = []
    for s, (i, nb) in nbrs.items():
        ii = np.intersect1d(i, idx, assume_unique=False)
        v = x[np.searchsorted(idx, ii)] if False else None
    return None

def eta2(x, lab):
    xs = StandardScaler().fit_transform(x); gm = xs.mean(0)
    b = sum((lab == g).sum() * ((xs[lab == g].mean(0) - gm) ** 2).sum() for g in np.unique(lab))
    return b / ((xs - gm) ** 2).sum()

rng = np.random.default_rng(SEED)
sub = rng.choice(len(sl), size=min(150000, len(sl)), replace=False); sub.sort()

def summarize(tag, x):
    xs = StandardScaler().fit_transform(x[sub])
    pc1 = PCA(n_components=1, random_state=SEED).fit_transform(xs)[:, 0]
    r_loc = abs(spearmanr(pc1, locfrac[sub]).statistic)
    print(f'  {tag:22s} eta2_slice {eta2(x[sub], sl[sub]):6.3f} | '
          f'eta2_celltype {eta2(x[sub], ct[sub]):6.3f} | eta2_compartment {eta2(x[sub], comp[sub]):6.3f} | '
          f'|rho(PC1, local tumour frac)| {r_loc:6.3f}')

z = np.load(CC + 'background.npy'); s_ = np.load(CC + 'salient.npy')
zcc = np.load(CC + 'spatial.npy'); scc = np.load(CC + 'salient_spatial.npy')
print('\n=== what each representation is organized by ===')
summarize('s intrinsic', s_)
for hh in range(3): summarize(f's-CC hop {hh+1}', scc[:, hh*32:(hh+1)*32])
summarize('s-CC all hops', scc)
summarize('z intrinsic', z)
summarize('z-CC all hops', zcc)

print('\n=== can s-CC just reproduce the local neighbourhood composition? ===')
types = pd.get_dummies(pd.Series(ct)).to_numpy(dtype=np.float32)
loccomp = np.zeros((len(sl), types.shape[1]), dtype=np.float32)
for s, (i, nb) in nbrs.items():
    loccomp[i] = types[i][nb].mean(1)
gkf = GroupKFold(n_splits=4)
for tag, x in [('s-CC', scc), ('z-CC', zcc), ('s intrinsic', s_)]:
    X = StandardScaler().fit_transform(x[sub])
    pred = cross_val_predict(Ridge(alpha=1.0), X, loccomp[sub], cv=gkf, groups=sl[sub])
    ss_res = ((loccomp[sub] - pred) ** 2).sum(0)
    ss_tot = ((loccomp[sub] - loccomp[sub].mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res.sum() / ss_tot.sum()
    print(f'  {tag:14s} held-section R^2 predicting local cell-type composition = {r2:.3f}')

print('\n=== does s-CC reflect drug response? (tumour cells, MAPK MPAS) ===')
bio = pd.read_csv('/home/yancui/haruka_v2/bench_kras_xenium/biological_responder/'
                  'treated_tumor_biological_response.csv.gz')
names = pd.Index(np.load(CC + 'obs_names.npy', allow_pickle=True).astype(str))
loc = names.get_indexer(bio.cell_id.astype(str)); ok = loc >= 0
for tag, x in [('s-CC', scc), ('z-CC', zcc), ('s intrinsic', s_), ('z intrinsic', z)]:
    xs = StandardScaler().fit_transform(x[loc[ok]])
    pc1 = PCA(n_components=1, random_state=SEED).fit_transform(xs)[:, 0]
    print(f'  {tag:14s} |rho(PC1, MAPK_MPAS)| {abs(spearmanr(pc1, bio.MAPK_MPAS[ok]).statistic):.3f}'
          f' | |rho(PC1, local tumour frac)| {abs(spearmanr(pc1, locfrac[loc[ok]]).statistic):.3f}')
