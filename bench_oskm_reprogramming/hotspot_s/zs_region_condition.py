#!/usr/bin/env python3
"""OSKM: is z = brain-region anatomy and s = condition? Per cell type, both arms, pair-batch fit.

A  decoding, leave-one-pair-out: subregion from z vs s (balanced accuracy, chance = 1/#subregions);
   condition (OSKM vs control) from z vs s, within subregion-centred latents (AUROC).
B  latent variance partition: per latent dimension, eta^2 of subregion and of condition (within pair);
   mean over dimensions.
C  Hotspot modules (replicated, from hotspot_s.py): module score = mean log expression of its genes;
   eta^2 of subregion vs eta^2 of condition (within pair), for z modules and s modules.
"""
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_oskm_reprogramming/hotspot_s'
MD = H / 'results_oskm_cf_expression/oskm_old_full_684253/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
CTS = ['Microglia', 'Neuron-Inhibitory', 'Neuron-Excitatory', 'Oligodendrocyte', 'Astrocyte', 'Neuron-MSN', 'Endothelial', 'OPC']
log = open(OUT / 'zs_region_condition.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

h = ad.read_h5ad(H / 'bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad', backed='r')
o = h.obs[['cell_type', 'paired_batch', 'condition', 'subregion']].copy()
ct = o.cell_type.astype(str).to_numpy(); pair = o.paired_batch.astype(str).to_numpy()
oskm = (o.condition.astype(str) == 'oskm').to_numpy(); sub = o.subregion.astype(str).to_numpy()
Z = np.load(MD / 'background.npy'); S = np.load(MD / 'salient.npy')
genes = np.asarray(h.var_names).astype(str)

def centre(X, g):
    X = np.asarray(X, float).copy()
    for u in np.unique(g): X[g == u] -= X[g == u].mean(0)
    return X
def eta2(v, g):
    v = v - v.mean(); tot = (v ** 2).sum()
    return sum(((g == u).sum() * v[g == u].mean() ** 2) for u in np.unique(g)) / tot if tot > 0 else np.nan

rng = np.random.default_rng(0); rows = []
for c in CTS:
    idx = np.flatnonzero(ct == c)
    idx = np.sort(rng.choice(idx, min(40000, len(idx)), replace=False))
    sb = sub[idx]; keep = pd.Series(sb).map(pd.Series(sb).value_counts()).to_numpy() >= 200
    idx = idx[keep]; sb, pr, ar = sub[idx], pair[idx], oskm[idx]
    rec = dict(cell_type=c, n=len(idx), n_sub=len(np.unique(sb)))
    for k, M in [('z', Z), ('s', S)]:
        X = centre(M[idx], pr)
        yp = np.empty(len(idx), object); sc = np.zeros(len(idx))
        Xs = centre(X, np.char.add(pr.astype(str), sb.astype(str)))
        for p in np.unique(pr):
            te, tr = pr == p, pr != p
            ss = StandardScaler().fit(X[tr]); yp[te] = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced').fit(ss.transform(X[tr]), sb[tr]).predict(ss.transform(X[te]))
            ss = StandardScaler().fit(Xs[tr]); sc[te] = LogisticRegression(max_iter=500, C=0.1).fit(ss.transform(Xs[tr]), ar[tr]).decision_function(ss.transform(Xs[te]))
        rec[f'region_balacc_{k}'] = balanced_accuracy_score(sb.astype(str), yp.astype(str))
        rec[f'condition_auroc_{k}'] = np.mean([roc_auc_score(ar[pr == p], sc[pr == p]) for p in np.unique(pr)])
        rec[f'eta2_region_{k}'] = np.mean([eta2(X[:, j], sb) for j in range(X.shape[1])])
        rec[f'eta2_condition_{k}'] = np.mean([eta2(centre(X[:, [j]], np.char.add(pr.astype(str), sb.astype(str))).ravel(), ar) for j in range(X.shape[1])])
    # C: Hotspot modules
    f = OUT / f'{c.replace("-", "_")}_replicated_modules.csv'
    if f.exists():
        R = pd.read_csv(f); A = h[idx].to_memory(); Cn = A.layers['counts'].tocsr()
        tot = np.asarray(Cn.sum(1)).ravel(); L = Cn.multiply(1e3 / np.maximum(tot, 1)[:, None]).tocsr(); L.data = np.log1p(L.data)
        for k in ['s', 'z']:
            er, ec = [], []
            for r in R[R.arm == k].itertuples():
                gi = [list(genes).index(g) for g in r.genes.split(';')]
                v = centre(np.asarray(L[:, gi].mean(1)), pr).ravel()
                er.append((eta2(v, sb), r.n)); ec.append((eta2(centre(v[:, None], np.char.add(pr.astype(str), sb.astype(str))).ravel(), ar), r.n))
            if er:
                rec[f'module_eta2_region_{k}'] = np.average([e for e, _ in er], weights=[n for _, n in er])
                rec[f'module_eta2_condition_{k}'] = np.average([e for e, _ in ec], weights=[n for _, n in ec])
    rows.append(rec); P(f'{c}: done')

T = pd.DataFrame(rows); T.to_csv(OUT / 'zs_region_condition.csv', index=False)
P('\n== A. decoding (leave-one-pair-out) ==')
P(T[['cell_type', 'n_sub', 'region_balacc_z', 'region_balacc_s', 'condition_auroc_z', 'condition_auroc_s']].round(3).to_string(index=False))
P('\n== B. latent variance explained (mean eta^2 over 32 dims; condition within pair x subregion) ==')
P(T[['cell_type', 'eta2_region_z', 'eta2_region_s', 'eta2_condition_z', 'eta2_condition_s']].round(4).to_string(index=False))
P('\n== C. Hotspot module scores (gene-weighted mean eta^2) ==')
P(T[['cell_type', 'module_eta2_region_z', 'module_eta2_region_s', 'module_eta2_condition_z', 'module_eta2_condition_s']].round(4).to_string(index=False))
P('done')
