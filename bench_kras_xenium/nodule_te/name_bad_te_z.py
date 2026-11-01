#!/usr/bin/env python3
"""Name the background (z) axes linked to the bad treatment effect (bad_te_background.py: zPC2, zPC3), and test whether
they PREDICT the outcome on held-out data.  Nodules / fit as bad_te_background.py (results_kras_te_null, 132 treated nodules).
1 genes: nodule mean expression of its tumour cells (h5ad X is already normalize+log1p), centred per section; Spearman with
  each z axis -> top genes both ends.  Niche: Spearman with D([z,0]) neighbour shares.
2 held-out prediction: for each treated section, PCA (5) of section-centred nodule mean z fitted on the OTHER 3 sections;
  the training axis matched to the original zPC2 / zPC3 by |cosine|, oriented so + = more s-specific bad TE in training;
  held-out Spearman of the projected axis with (a) the s-specific Itgb6+ TE and (b) the independent responder rate / MPAS.
3 presence before treatment: project the 124 control nodules (centred per control section) on the same axes;
  SD ratio control / treated, and the control axis range covered.
"""
import os, json, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.neighbors import radius_neighbors_graph
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_kras_xenium/nodule_te'
MD = H / 'results_kras_te_null/mouselung_kras_xenium_clean_noC3/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r'); o = a.obs
types = json.load(open(MD / 'meta.json'))['cell_type_categories']; genes = np.asarray(a.var_names).astype(str)
TUMOR = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']
ct = o.celltype.astype(str).to_numpy(); sl = o.slice_id.astype(str).to_numpy(); treated = (o.condition_label.astype(str) == 'KRASi_treated').to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]; tum = np.isin(ct, TUMOR)
Z = np.load(MD / 'background.npy', mmap_mode='r'); pbg = np.load(MD / 'background_composition.npy', mmap_mode='r')
log = open(OUT / 'name_bad_te_z.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
# nodules in the same order as bad_te_background.py
nod = []
for s_ in np.unique(sl):
    q = np.flatnonzero((sl == s_) & tum); _, lab = connected_components(radius_neighbors_graph(xy[q], 25.0, mode='connectivity'), directed=False)
    for l in np.flatnonzero(np.bincount(lab) >= 50): nod.append((s_, bool(treated[q[0]]), q[lab == l]))
N = pd.read_csv(OUT / 'bad_te_nodules.csv'); N['te_spec'] = N.te - N[['null0', 'null1', 'null2']].mean(1)
TR = [x for x in nod if x[1]]; assert len(TR) == len(N) and all(len(x[2]) == n for x, n in zip(TR, N.n))
CO = [x for x in nod if not x[1]]
zm = lambda L: np.stack([np.asarray(Z[i]).mean(0) for _, _, i in L])
def cen(X, g):
    X = X.copy()
    for u in np.unique(g): X[g == u] -= X[g == u].mean(0)
    return X
sec = N.section.to_numpy(); Zt = cen(zm(TR), sec)
pca = PCA(5, random_state=0).fit(Zt); sc = pca.transform(Zt)
assert np.allclose(np.abs(np.corrcoef(sc[:, 1], N.zPC2)[0, 1]), 1, atol=1e-6)
AX = {'zPC2': 1, 'zPC3': 2}
for name, j in AX.items():
    if np.sign(np.corrcoef(sc[:, j], N[name])[0, 1]) < 0: sc[:, j] *= -1; pca.components_[j] *= -1
    if np.sign(spearmanr(sc[:, j], N.te_spec).statistic) < 0: sc[:, j] *= -1; pca.components_[j] *= -1   # + = more bad TE
    P(f'{name}: {pca.explained_variance_ratio_[j]:.0%} of nodule z variance; Spearman with s-specific bad TE {spearmanr(sc[:, j], N.te_spec).statistic:+.2f}, responder rate {spearmanr(sc[:, j], N.resp).statistic:+.2f} (pooled, section-centred)')
# 1 genes + niche
X = np.stack([np.asarray(a[i].X.mean(0)).ravel() if not sp.issparse(a[i].X) else np.asarray(a[i].X.mean(0)).ravel() for _, _, i in TR])
X = cen(X, sec); keep = X.std(0) > 0
for name, j in AX.items():
    r = np.full(len(genes), np.nan); r[keep] = [spearmanr(X[:, g], sc[:, j]).statistic for g in np.flatnonzero(keep)]
    pd.DataFrame({'gene': genes, 'rho': r}).sort_values('rho').to_csv(OUT / f'name_{name}_genes.csv', index=False)
    o_ = np.argsort(np.nan_to_num(r))
    P(f'\n{name} genes up with bad TE (+ end):', ', '.join(f'{genes[g]} {r[g]:+.2f}' for g in o_[::-1][:15]))
    P(f'{name} genes down (- end):', ', '.join(f'{genes[g]} {r[g]:+.2f}' for g in o_[:12]))
    rn = {t: spearmanr(N[f'pbg:{t}'], sc[:, j]).statistic for t in types if f'pbg:{t}' in N}
    rn = pd.Series(rn).sort_values()
    P(f'{name} predicted neighbours + end:', ', '.join(f'{k} {v:+.2f}' for k, v in rn.tail(6)[::-1].items()), '| - end:', ', '.join(f'{k} {v:+.2f}' for k, v in rn.head(4).items()))
# 2 held-out prediction
P('\n== held-out section: axis fitted on the other 3 treated sections ==')
rows = []
for u in np.unique(sec):
    tr_, te_ = sec != u, sec == u
    p2 = PCA(5, random_state=0).fit(Zt[tr_])
    for name, j in AX.items():
        k = int(np.argmax(np.abs(p2.components_ @ pca.components_[j]))); v = p2.components_[k]
        s_tr = Zt[tr_] @ v
        if spearmanr(s_tr, N.te_spec[tr_]).statistic < 0: v = -v
        s_te = Zt[te_] @ v
        rows.append(dict(held_out=u, axis=name, matched_pc=k + 1, cos=abs(float(v @ pca.components_[j])), n=int(te_.sum()),
                         rho_bad_te=spearmanr(s_te, N.te_spec[te_]).statistic, rho_resp=spearmanr(s_te, N.resp[te_]).statistic, rho_mpas=spearmanr(s_te, N.mpas[te_]).statistic))
Hd = pd.DataFrame(rows); Hd.to_csv(OUT / 'name_bad_te_heldout.csv', index=False); P(Hd.round(2).to_string(index=False))
for name in AX:
    t = Hd[Hd.axis == name]
    P(f'  {name}: held-out rho with bad TE median {t.rho_bad_te.median():+.2f} ({(t.rho_bad_te > 0).sum()}/4 > 0) | with responder rate median {t.rho_resp.median():+.2f} ({(t.rho_resp < 0).sum()}/4 < 0) | with MPAS {t.rho_mpas.median():+.2f}')
# 3 presence in controls
Zc = cen(zm(CO), np.array([s for s, _, _ in CO]))
for name, j in AX.items():
    a_t = Zt @ pca.components_[j]; a_c = Zc @ pca.components_[j]
    P(f'\n{name} in untreated control nodules ({len(CO)}): SD control/treated {a_c.std() / a_t.std():.2f}; control 5-95% range {np.percentile(a_c, 5):+.2f}..{np.percentile(a_c, 95):+.2f} vs treated {np.percentile(a_t, 5):+.2f}..{np.percentile(a_t, 95):+.2f}')
# figure export: per-nodule z-PC2 scores (treated + control projections), neighbour correlations, example-section membership
csec = np.array([s for s, _, _ in CO]); a2t = Zt @ pca.components_[1]; a2c = Zc @ pca.components_[1]
ex = 'JB_M0'; exn = [(k, i) for k, (s_, _, i) in enumerate(TR) if s_ == ex]
rn2 = pd.Series({t: spearmanr(N[f'pbg:{t}'], sc[:, 1]).statistic for t in types if f'pbg:{t}' in N}).sort_values()
rn2.to_csv(OUT / 'name_zPC2_neighbours.csv', header=['rho'])
np.savez(OUT / 'name_zPC2_fig.npz', treated_score=a2t, treated_section=sec, control_score=a2c, control_section=csec,
         ex_section=ex, ex_cells=np.concatenate([i for _, i in exn]), ex_score=np.concatenate([np.full(len(i), a2t[k]) for k, i in exn]))
P('done')
