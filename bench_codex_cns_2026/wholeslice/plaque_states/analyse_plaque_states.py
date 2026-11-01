#!/usr/bin/env python3
"""What does s say about plaques?
  Q1  cell level: does z (or s, or protein) recognise 'near a plaque' when trained on AD and
      tested on Ctl-3's plaques?  (Ctl-3 was background during training)
  Q2  plaque level: size of the local plaque effect (delta vs same donor's distal tissue) in
      z, s and protein, AD donors vs Ctl-3
  Q3  heterogeneity: cluster AD plaque niches on delta-s; are the states donor-independent,
      reproducible on a held-out donor, interpretable, AD-specific (absent at Ctl-3 plaques),
      and do they add anything beyond the same clustering on delta-protein?
"""
import sys, warnings
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score, silhouette_score, adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from sklearn.neighbors import NearestCentroid
SEED = 0
M = ('/home/yancui/haruka_v2/results_codex_wholeslice/codex_wholeslice_all7_no_oligo/'
     'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
T = pd.read_csv('plaque_niches.csv'); F = np.load('plaque_niches.npz', allow_pickle=True)
prot, CT, MS = list(F['prot']), list(F['ct']), list(F['mstates'])
D = {'z': F['dz'], 's': F['ds'], 'protein': F['dp']}
gmp = (T.wm_fraction == 0).to_numpy()                      # niches entirely in grey matter
T = T[gmp].reset_index(drop=True); D = {k: v[gmp] for k, v in D.items()}
F = {k: (F[k][gmp] if k in ('dz', 'ds', 'dp', 'dct', 'ms') else F[k]) for k in F.files}
ad_ = (T.condition == 'AD').to_numpy(); donor = T.donor.to_numpy()
print(f'GM-only plaque niches: {len(T)} ({ad_.sum()} AD, {(~ad_).sum()} Ctl-3)')
# Q3 uses delta standardised WITHIN donor: heterogeneity among one donor's plaques, not donor scale
DW = {}
for k, v in D.items():
    w = v.copy()
    for d in np.unique(donor):
        m = donor == d; w[m] = (w[m] - w[m].mean(0)) / (w[m].std(0) + 1e-9)
    DW[k] = w

print('== Q1 cell level: trained on AD donors, tested on Ctl-3 (<=10 um vs >100 um of any Abeta object) ==')
a = ad.read_h5ad('../codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
dist = np.load('cell_dist_any_abeta_um.npy'); don = o.donor_id.astype(str).to_numpy()
X = {'z': np.load(f'{M}/background.npy'), 's': np.load(f'{M}/salient.npy'), 'protein': np.asarray(a.X)}
keep = (dist <= 10) | (dist > 100); y = (dist <= 10).astype(int)
tr = keep & np.isin(don, ['AD-1', 'AD-2', 'AD-3', 'AD-4']); te = keep & (don == 'Ctl-3')
print(f'  train {tr.sum()} cells ({y[tr].mean():.1%} pos), Ctl-3 test {te.sum()} ({y[te].mean():.1%} pos)')
for k, x in X.items():
    sc = StandardScaler().fit(x[tr])
    clf = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED).fit(sc.transform(x[tr]), y[tr])
    print(f'  {k:8s} AUROC on Ctl-3 plaques {roc_auc_score(y[te], clf.predict_proba(sc.transform(x[te]))[:, 1]):.3f}')

print('\n== Q2 plaque effect size: ||delta|| in units of each feature\'s cell-level SD ==')
sd = {k: X[k][np.isin(don, list(set(donor)))].std(0) for k in X}
for k in D:
    n = np.linalg.norm(D[k] / sd[k], axis=1) / np.sqrt(D[k].shape[1])
    print(f'  {k:8s} ' + '  '.join(f'{d} {np.median(n[donor == d]):.3f}' for d in sorted(set(donor))))
dp = pd.DataFrame(D['protein'], columns=prot)
top = ['ApoE', 'CD68', 'CD11c', 'HLA-DR', 'TMEM119', 'IBA1', 'GFAP', 'AQP4', 'Gal3', 'NeuN', 'MAP2', 'Synaptophysin', 'PSD95']
print('  median delta protein (z units) by donor:'); print(dp.groupby(T.donor)[top].median().round(2).to_string())

print('\n== Q3 plaque states from delta-s (AD plaques) ==')
def embed(x, fit_mask):
    sc = StandardScaler().fit(x[fit_mask]); p = PCA(10, random_state=SEED).fit(sc.transform(x[fit_mask]))
    return lambda m: p.transform(sc.transform(x[m]))
res = {}
for k in ['s', 'protein', 'z']:
    f = embed(DW[k], ad_); E = f(ad_)
    sil = {kk: silhouette_score(E, KMeans(kk, n_init=10, random_state=SEED).fit_predict(E), sample_size=4000, random_state=SEED)
           for kk in range(2, 9)}
    kbest = max(sil, key=sil.get); lab = KMeans(kbest, n_init=20, random_state=SEED).fit_predict(E)
    # reproducibility: fit on 3 donors, assign held-out donor by nearest centroid, compare with its own clustering
    rep = []
    for d in ['AD-1', 'AD-2', 'AD-3', 'AD-4']:
        trm = ad_ & (donor != d); tem = donor == d
        g = embed(DW[k], trm); km = KMeans(kbest, n_init=20, random_state=SEED).fit(g(trm))
        own = KMeans(kbest, n_init=20, random_state=SEED).fit_predict(g(tem))
        rep.append(ari(km.predict(g(tem)), own))
    res[k] = (E, lab, kbest)
    print(f'  {k:8s} best k {kbest} (silhouette {sil[kbest]:.3f}; all {{{", ".join(f"{a}:{b:.2f}" for a, b in sil.items())}}})  '
          f'donor NMI {nmi(donor[ad_], lab):.3f}  held-out-donor ARI {np.mean(rep):.3f} ({", ".join(f"{r:.2f}" for r in rep)})')
ls, lp = res['s'][1], res['protein'][1]
print(f'  agreement s-states vs protein-states ARI {ari(ls, lp):.3f}')
acc = (cross_val_predict(LogisticRegression(max_iter=3000, random_state=SEED), StandardScaler().fit_transform(DW['protein'][ad_]),
                         ls, groups=donor[ad_], cv=GroupKFold(4)) == ls).mean()
print(f'  s-state predictable from delta-protein (leave-one-donor-out acc) {acc:.3f}  (chance {pd.Series(ls).value_counts(normalize=True).max():.3f})')

print('\n  s-state profiles (AD plaques):')
A = T[ad_].assign(state=ls)
prof = A.groupby('state').agg(n=('plaque', 'size'), dense_frac=('cls', lambda c: (c == 'dense').mean()),
                              area_med=('area_um2', 'median'), abeta_med=('abeta', 'median'),
                              apoe_in_plaque=('apoe_in_plaque', 'median'), wm_frac=('wm_fraction', 'mean'))
prof = prof.join(pd.crosstab(A.state, A.donor, normalize='index').round(2))
print(prof.round(2).to_string())
print('  delta protein (top |mean|) per state:')
dpa = dp[ad_].assign(state=ls).groupby('state').mean()
for st, r in dpa.iterrows():
    r = r.sort_values(); print(f'    state {st}: up ' + ', '.join(f'{g} {v:+.2f}' for g, v in r[::-1][:6].items())
                               + ' | down ' + ', '.join(f'{g} {v:+.2f}' for g, v in r[:4].items()))
ct = pd.DataFrame(F['dct'][ad_], columns=CT).assign(state=ls).groupby('state').mean()
ms = pd.DataFrame(F['ms'][ad_], columns=MS).assign(state=ls).groupby('state').mean()
print('  delta cell-type fraction per state:'); print(ct.round(3).to_string())
print('  myeloid-state fractions among niche microglia per state:'); print(ms.round(2).to_string())

print('\n  Ctl-3 plaques projected onto the AD s-states (nearest centroid in the AD embedding):')
f = embed(DW['s'], ad_); nc = NearestCentroid().fit(f(ad_), ls)
pc = pd.Series(nc.predict(f(~ad_))).value_counts(normalize=True).sort_index()
pa = pd.Series(ls).value_counts(normalize=True).sort_index()
print(pd.DataFrame({'AD share': pa, 'Ctl-3 share': pc}).fillna(0).round(3).to_string())
pd.DataFrame({'donor': donor[ad_], 'plaque': T.plaque[ad_], 's_state': ls, 'protein_state': lp}).to_csv('ad_plaque_states.csv', index=False)
