#!/usr/bin/env python3
"""Lung fibrosis, condition (fibrosis) cells only: latent domains from a kNN graph on z or on s, and whether each
core's domain proportions separate Less vs More Affected.
Domains: k-means (k = 8 / 15 / 25, same k for z and s; MiniBatchKMeans, random_state 0, fit on a 200k-cell
subsample of the standardised latent) assigned to every fibrosis cell.
Per core: domain proportions -> centred log-ratio (pseudocount 1e-3).
Stage test (31 cores, 23 donors): logistic L2 (C chosen from {0.01,0.1,1} by inner CV is too unstable at n=31,
so fixed C = 0.1) on standardised CLR, leave-one-donor-out -> AUROC; Spearman(held-out score, % pathology);
best single domain: max |Spearman(proportion, % pathology)|.
"""
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage'
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
stage = o.sample_affect.astype(str).to_numpy(); F = np.flatnonzero(stage != 'Unaffected')
core = o['sample'].astype(str).to_numpy()[F]; donor = o.donor.astype(str).to_numpy()[F]; st = stage[F]
path = pd.to_numeric(o.percent_pathology, errors='coerce').to_numpy()[F]; ct = o.cell_type.astype(str).to_numpy()[F]
L = {'z': np.load(MD / 'background.npy')[F], 's': np.load(MD / 'salient.npy')[F]}
log = open(OUT / 'domain_stage.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
rng = np.random.default_rng(0); sub = np.sort(rng.choice(len(F), 200000, replace=False))
C = pd.DataFrame({'core': core, 'donor': donor, 'stage': st, 'path': path}).groupby('core').first()
y = (C.stage == 'More Affected').to_numpy(); dn = C.donor.to_numpy()
P(f'fibrosis cells {len(F)}, cores {len(C)} (More {y.sum()}, Less {(~y).sum()}), donors {len(set(dn))}')

rows, keep = [], {}
for target in [8, 15, 25]:
    for k, M in L.items():
        X = StandardScaler().fit_transform(M)
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=target, random_state=0, n_init=5, batch_size=8192).fit(X[sub]); lab = km.predict(X); r = np.nan
        Pm = pd.crosstab(core, lab, normalize='index').reindex(C.index).fillna(0)
        clr = np.log(Pm.to_numpy() + 1e-3); clr -= clr.mean(1, keepdims=True)
        sc_ = np.zeros(len(C))
        for d in np.unique(dn):
            te = dn == d; ss = StandardScaler().fit(clr[~te])
            sc_[te] = LogisticRegression(C=0.1, max_iter=2000).fit(ss.transform(clr[~te]), y[~te]).decision_function(ss.transform(clr[te]))
        rho = [spearmanr(Pm.iloc[:, j], C.path).statistic for j in range(Pm.shape[1])]
        jb = int(np.nanargmax(np.abs(rho)))
        top_ct = pd.Series(ct[lab == Pm.columns[jb]]).value_counts(normalize=True).head(3)
        rows.append(dict(target=target, latent=k, n_domains=Pm.shape[1], resolution=r, auroc=roc_auc_score(y, sc_),
                         rho_score_pathology=spearmanr(sc_, C.path).statistic, best_domain_rho=rho[jb],
                         best_domain_cells=', '.join(f'{i} {v:.0%}' for i, v in top_ct.items())))
        keep[(target, k)] = (lab, Pm)
        P(f'target {target:2d} {k}: {Pm.shape[1]} domains | LODO AUROC {rows[-1]["auroc"]:.3f} | rho(score, pathology) {rows[-1]["rho_score_pathology"]:+.3f} '
          f'| best domain rho {rho[jb]:+.3f} ({rows[-1]["best_domain_cells"]})')
T = pd.DataFrame(rows); T.to_csv(OUT / 'domain_stage.csv', index=False)
for (t, k), (lab, Pm) in keep.items():
    np.save(OUT / f'domains_{k}_{t}.npy', lab); Pm.to_csv(OUT / f'domain_props_{k}_{t}.csv')
P('\n' + T.round(3).to_string(index=False)); P('done')
