#!/usr/bin/env python3
"""Exercise: is z = brain-region anatomy and s = condition? Per cell type, both arms.
Folds hold out one control + one exercise mouse (4 folds), so decoding generalises to unseen mice.
A  subregion from z vs s (balanced accuracy); condition from z vs s on subregion-centred latents (AUROC on held-out mice).
B  mean eta^2 over latent dims: subregion, and condition and mouse identity (within subregion; mouse includes condition).
C  mouse-level replication: per held-out mouse pair, sign of the mean predicted condition score (exercise - control).
usage: zs_region_condition_ex.py <model_dir> <tag>
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')
MD = Path(sys.argv[1]); TAG = sys.argv[2]; OUT = Path('/home/yancui/haruka_v2/bench_exercise')
CTS = ['Microglia', 'Neuron-Inhibitory', 'Neuron-Excitatory', 'Oligodendrocyte', 'Astrocyte', 'Neuron-MSN', 'Endothelial', 'OPC']
log = open(OUT / f'zs_region_condition_{TAG}.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
h = ad.read_h5ad('/project/zhihuanglab/yancui/haruka_exercise/data/exercise_old_full.h5ad', backed='r')
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(h.obs_names).astype(str)).all()
o = h.obs; ct = o.cell_type.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy()
ex = (o.condition.astype(str) == 'exercise').to_numpy(); sub = o.subregion.astype(str).to_numpy(); run = o.run_batch.astype(str).to_numpy()
Z = np.load(MD / 'background.npy'); S = np.load(MD / 'salient.npy')
ctrl_m = sorted(set(mouse[~ex])); ex_m = sorted(set(mouse[ex]))
folds = [(ctrl_m[i], ex_m[i]) for i in range(4)]
def centre(X, g):
    X = np.asarray(X, float).copy()
    for u in np.unique(g): X[g == u] -= X[g == u].mean(0)
    return X
def eta2(v, g):
    v = v - v.mean(); tot = (v ** 2).sum()
    return sum(((g == u).sum() * v[g == u].mean() ** 2) for u in np.unique(g)) / tot if tot > 0 else np.nan
rng = np.random.default_rng(0); rows = []
for c in CTS:
    idx = np.flatnonzero(ct == c); idx = np.sort(rng.choice(idx, min(48000, len(idx)), replace=False))
    sb = sub[idx]; keep = pd.Series(sb).map(pd.Series(sb).value_counts()).to_numpy() >= 200; idx = idx[keep]
    sb, ms, ar, rb = sub[idx], mouse[idx], ex[idx], run[idx]
    rec = dict(cell_type=c, n=len(idx))
    for k, M in [('z', Z), ('s', S)]:
        X = centre(M[idx], ms)            # remove per-mouse offsets for the region readout
        Xc = centre(M[idx], np.char.add(rb.astype(str), sb.astype(str)))  # condition readout: within run x subregion
        yp = np.empty(len(idx), object); sc = np.zeros(len(idx)); pair_sign = []
        for cm, em in folds:
            te = np.isin(ms, [cm, em]); tr = ~te
            ss = StandardScaler().fit(X[tr]); yp[te] = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced').fit(ss.transform(X[tr]), sb[tr]).predict(ss.transform(X[te]))
            # subregion offsets from TRAINING mice only (no held-out data in any centring)
            Mi = np.asarray(M[idx], float); Xt = Mi.copy()
            for u in np.unique(sb):
                Xt[sb == u] -= Mi[tr & (sb == u)].mean(0)
            ss = StandardScaler().fit(Xt[tr]); sc[te] = LogisticRegression(max_iter=500, C=0.1).fit(ss.transform(Xt[tr]), ar[tr]).decision_function(ss.transform(Xt[te]))
            pair_sign.append(sc[te & ar].mean() > sc[te & ~ar].mean())
        rec[f'region_balacc_{k}'] = balanced_accuracy_score(sb.astype(str), yp.astype(str))
        rec[f'condition_auroc_{k}'] = np.mean([roc_auc_score(ar[np.isin(ms, f)], sc[np.isin(ms, f)]) for f in folds])
        rec[f'mouse_pairs_correct_{k}'] = int(np.sum(pair_sign))
        rec[f'eta2_region_{k}'] = np.mean([eta2(X[:, j], sb) for j in range(X.shape[1])])
        rec[f'eta2_condition_{k}'] = np.mean([eta2(centre(M[idx][:, [j]], sb).ravel(), ar) for j in range(M.shape[1])])
        rec[f'eta2_mouse_{k}'] = np.mean([eta2(centre(M[idx][:, [j]], sb).ravel(), ms) for j in range(M.shape[1])])
    rows.append(rec); P(f'{c}: done')
T = pd.DataFrame(rows); T.to_csv(OUT / f'zs_region_condition_{TAG}.csv', index=False)
P(f'\n== {TAG}: A. decoding on held-out mice (4 folds = 1 control + 1 exercise mouse each) ==')
P(T[['cell_type', 'region_balacc_z', 'region_balacc_s', 'condition_auroc_z', 'condition_auroc_s', 'mouse_pairs_correct_z', 'mouse_pairs_correct_s']].round(3).to_string(index=False))
P('\n== B. mean eta^2 over latent dims ==')
P(T[['cell_type', 'eta2_region_z', 'eta2_region_s', 'eta2_condition_z', 'eta2_condition_s', 'eta2_mouse_z', 'eta2_mouse_s']].round(4).to_string(index=False))
P('done')
