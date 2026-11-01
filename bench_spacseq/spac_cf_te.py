#!/usr/bin/env python3
"""SPAC-seq lung: treatment effect (TE) of each knockout on Haruka's counterfactual neighbourhood
cell-type proportions.

Fit: results_real_compgated_nb_final/spac_lung_functional_guides_k8_hvg3000 (condition = KO-labelled
bins, reference = NT bins).  For every KO bin the composition head gives
  background p([z,0]), salient p([z,s]); shift dp = p([z,s]) - p([z,0])   (dp = 0 on NT bins)
TE of target t on neighbour type j, per slice, matched on the bin's own cell type against NT bins
of the same slice and cell type:
  observed   TE_obs = obs(t) - obs(NT)           (observed k=20 neighbourhood composition)
  background TE_bg  = p_bg(t) - obs(NT)
  salient    TE_sal = p_full(t) - p_bg(t) = mean dp(t)
Questions: (1) how much of TE_obs is background vs salient; (2) is the salient TE TARGET-SPECIFIC
(deviation from the mean over targets) and does it replicate across the 3 slices (permutation of
target labels); (3) does it agree with the observed target-specific TE.
"""
import os, json, itertools, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
H = '/home/yancui/haruka_v2'; OUT = f'{H}/bench_spacseq/results_multislice/cf_te'; os.makedirs(OUT, exist_ok=True)
MD = f'{H}/results_real_compgated_nb_final/spac_lung_functional_guides_k8_hvg3000/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(f'{H}/bench_spacseq/data/spac_lung_functional_guides_k8_hvg3000.h5ad', backed='r'); o = a.obs
ct = o.cell_type.astype(str).to_numpy(); sl = o.slice_id.astype(str).to_numpy(); pt = o.perturbation.astype(str).to_numpy()
xy = np.asarray(a.obsm['spatial'], float); cats = json.load(open(MD + '/meta.json'))['cell_type_categories']; K = 20
assert (np.load(MD + '/obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
log = open(f'{OUT}/cf_te.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

code = pd.Categorical(ct, categories=cats).codes; obs = np.zeros((len(ct), len(cats)), np.float32)
for m in np.unique(sl):
    i = np.where(sl == m)[0]; _, nb = cKDTree(xy[i]).query(xy[i], k=K + 1)
    for c in range(1, K + 1): np.add.at(obs, (i, code[i[nb[:, c]]]), 1.0)
obs /= K
pbg = np.load(MD + '/background_composition.npy'); pf = np.load(MD + '/condition_composition.npy'); dp = np.load(MD + '/salient_composition_shift.npy')
ko = pt != 'NT'
P(f'SPAC-seq lung: {len(ct)} bins, {ko.sum()} KO / {(~ko).sum()} NT, slices {sorted(set(sl))}; neighbour types {cats}')
P('dp == p_full - p_bg on KO bins: %.1e | dp on NT bins max |.| %.1e' % (np.abs(dp[ko] - (pf - pbg)[ko]).max(), np.abs(dp[~ko]).max()))
def r2(y, f): return 1 - ((y - f) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
P('R2 of observed composition: NT p_bg %.3f | KO p_bg %.3f  KO p_full %.3f' % (r2(obs[~ko], pbg[~ko]), r2(obs[ko], pbg[ko]), r2(obs[ko], pf[ko])))
P('mean |dp| per KO bin (L1/2) %.4f vs mean TV(obs, p_bg) on KO bins %.4f' % (0.5 * np.abs(dp[ko]).sum(1).mean(), 0.5 * np.abs(obs - pbg)[ko].sum(1).mean()))

targets = sorted(set(pt) - {'NT'}); slices = sorted(set(sl)); rows = []
for t in targets:
    for m in slices:
        acc = {k: np.zeros(len(cats)) for k in ['obs', 'bg', 'sal']}; n = 0
        for c in np.unique(ct):
            T = (pt == t) & (sl == m) & (ct == c); R = (pt == 'NT') & (sl == m) & (ct == c)
            if T.sum() < 5 or R.sum() < 10: continue
            ref = obs[R].mean(0); w = T.sum(); n += w
            acc['obs'] += w * (obs[T].mean(0) - ref); acc['bg'] += w * (pbg[T].mean(0) - ref); acc['sal'] += w * dp[T].mean(0)
        if n == 0: continue
        for k in acc:
            rows.append(dict(target=t, slice=m, n=n, kind=k, **dict(zip(cats, acc[k] / n))))
E = pd.DataFrame(rows); E.to_csv(f'{OUT}/te_by_target_slice.csv', index=False)
pool = E.groupby(['kind', 'target']).apply(lambda x: pd.Series(np.average(x[cats], axis=0, weights=x.n), index=cats))

P('\n-- 1. TE pooled over slices (change in neighbour fraction vs matched NT) --')
for k, lab in [('obs', 'observed'), ('bg', 'background p([z,0])'), ('sal', 'salient shift dp')]:
    P(f'   {lab}'); P('   ' + pool.loc[k].round(4).to_string().replace('\n', '\n   '))
mean_ = {k: pool.loc[k].mean(0) for k in ['obs', 'bg', 'sal']}
P('\n   common KO effect (mean over targets):'); P('   ' + pd.DataFrame(mean_).T.round(4).to_string().replace('\n', '\n   '))
tot = {k: np.abs(pool.loc[k].to_numpy()).sum() for k in ['obs', 'bg', 'sal']}
P('   sum |TE|: observed %.3f  background %.3f  salient %.3f' % (tot['obs'], tot['bg'], tot['sal']))

P('\n-- 2. target-specific salient TE (deviation from the mean over targets): replication across slices --')
def dev(k, m):
    X = E[(E.kind == k) & (E.slice == m)].set_index('target')[cats].reindex(targets); return X - X.mean(0)
rng = np.random.default_rng(0)
for k in ['sal', 'obs', 'bg']:
    for m1, m2 in itertools.combinations(slices, 2):
        A, B = dev(k, m1), dev(k, m2); ok = A.notna().all(1) & B.notna().all(1); A, B = A[ok].to_numpy(), B[ok].to_numpy()
        r = np.corrcoef(A.ravel(), B.ravel())[0, 1]
        null = [np.corrcoef(A.ravel(), B[rng.permutation(len(B))].ravel())[0, 1] for _ in range(5000)]
        P(f'   {k:4s} {m1} vs {m2}: r = {r:+.2f}  (target-label permutation p = {np.mean(np.array(null) >= r):.3f}, {ok.sum()} targets)')

P('\n-- 3. does the salient TE agree with the observed TE the background leaves (obs - bg), target-specific, pooled --')
S_ = pool.loc['sal'] - pool.loc['sal'].mean(0); Rr = (pool.loc['obs'] - pool.loc['bg']); Rr = Rr - Rr.mean(0)
O_ = pool.loc['obs'] - pool.loc['obs'].mean(0)
P('   corr(salient, obs - bg) over target x type: %+.2f | corr(salient, observed) %+.2f' %
  (np.corrcoef(S_.to_numpy().ravel(), Rr.to_numpy().ravel())[0, 1], np.corrcoef(S_.to_numpy().ravel(), O_.to_numpy().ravel())[0, 1]))
for j in cats:
    P(f'   {j:16s} per-target corr(salient, obs-bg) {np.corrcoef(S_[j], Rr[j])[0, 1]:+.2f}   salient range {S_[j].min():+.4f}..{S_[j].max():+.4f}')

P('\n-- 4. per target x neighbour type: salient TE sign agreement across slices (3/3 = consistent) --')
sg = E[E.kind == 'sal'].groupby('target')[cats].agg(lambda v: int(max((v > 0).sum(), (v < 0).sum())))
P('   ' + sg.to_string().replace('\n', '\n   '))
P('done')
