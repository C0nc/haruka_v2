#!/usr/bin/env python3
"""KRAS: nodule-level treatment effect (TE) of the counterfactual neighbourhood composition.

TE of a treated tumour cell = D([z,s]) - D([z,0]) over the 34 neighbour cell types (saved salient_composition_shift).
Unit = tumour nodule: connected component of tumour cells (radius graph R_UM within a section), >= MIN_CELLS cells.
  Step 1  nodule TE = mean cell TE; its structure (PCA) and replication across the 4 treated sections.
  Step 2  TE vs data Haruka never used: observed - p_bg bookkeeping; MAPK MPAS (drug PD), responder rate,
          Itgb6+ persister fraction, proliferation -- Spearman WITHIN each section, and partial on nodule size + density.
  Step 3 (preliminary, no null yet)  leave-section-out ridge: nodule TE from baseline z, from p_bg, vs from s (ceiling).
usage: nodule_te.py <model_dir> <tag> [null_shift.npy ...]
"""
import os, sys, json, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, rankdata
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_kras_xenium/nodule_te'
MD = Path(sys.argv[1]); TAG = sys.argv[2]; NULLS = sys.argv[3:]
R_UM, MIN_CELLS, K = 25.0, 50, 20
OWN = os.environ.get('OWN_TYPE')  # if set: nodule summaries use only tumour cells of this type (removes own-type mix)
TUMOR = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']
log = open(OUT / f'nodule_te_{TAG}.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

a = ad.read_h5ad('/home/yancui/mouselung_kras_xenium_clean_noC3.h5ad', backed='r'); o = a.obs
names = np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str)
assert (names == np.asarray(a.obs_names).astype(str)).all()
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
ct = o.celltype.astype(str).to_numpy(); sl = o.slice_id.astype(str).to_numpy()
treated = (o.condition_label.astype(str) == 'KRASi_treated').to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]
tum = np.isin(ct, TUMOR)
Z = np.load(MD / 'background.npy'); S = np.load(MD / 'salient.npy')
pbg = np.load(MD / 'background_composition.npy'); dp = np.load(MD / 'salient_composition_shift.npy')
assert pbg.shape[1] == len(types)
nulls = [np.load(f) for f in NULLS]

# signatures (define_biological_responder.py, rows = tumour cells in h5ad order)
sig = pd.read_csv(H / 'bench_kras_xenium/biological_responder/all_tumor_signature_scores.csv.gz')
assert len(sig) == tum.sum() and (sig.ct.to_numpy() == ct[tum]).all()
SIG = {c: np.full(len(ct), np.nan) for c in ['MAPK_MPAS', 'Proliferation', 'EMT', 'Stress_p21']}
for c in SIG: SIG[c][tum] = sig[c].to_numpy()
p25 = {t: np.nanpercentile(SIG['MAPK_MPAS'][tum & ~treated & (ct == t)], 25) for t in TUMOR}
thr = np.array([p25.get(c, -np.inf) for c in ct]); resp = tum & (np.nan_to_num(SIG['MAPK_MPAS'], nan=np.inf) < thr)

# observed k=20 neighbour composition of tumour cells (within section, self excluded)
tix = {t: i for i, t in enumerate(types)}; code = np.array([tix.get(c, -1) for c in ct])
obs = np.zeros((len(ct), len(types)), np.float32)
rows = []
for s_ in np.unique(sl):
    m = np.flatnonzero(sl == s_); q = m[tum[m]]
    nn = NearestNeighbors(n_neighbors=K + 1).fit(xy[m]); _, ind = nn.kneighbors(xy[q])
    nb = code[m][ind[:, 1:]]
    for j in range(len(types)): obs[q, j] = (nb == j).mean(1)
    # nodules
    G = radius_neighbors_graph(xy[q], R_UM, mode='connectivity')
    _, lab = connected_components(G, directed=False)
    for l, cnt in zip(*np.unique(lab, return_counts=True)):
        if cnt >= MIN_CELLS: rows.append((s_, l, q[lab == l]))
P(f'{TAG}: nodules >= {MIN_CELLS} tumour cells (radius {R_UM:.0f} um): ' +
  ', '.join(f'{s_} {sum(r[0] == s_ for r in rows)}' for s_ in np.unique(sl)))
tumour_cols = [tix[t] for t in TUMOR]

N = []
for s_, l, idx0 in rows:
    idx = idx0[ct[idx0] == OWN] if OWN else idx0
    if len(idx) < 30: continue
    rec = dict(section=s_, treated=bool(treated[idx[0]]), n=len(idx),
               itgb6=np.mean(ct[idx0] == 'Tumor_epithelial_Itgb6'), resp_rate=resp[idx].mean(),
               density=obs[idx][:, tumour_cols].sum(1).mean(), **{c: np.nanmean(SIG[c][idx]) for c in SIG})
    N.append((rec, idx))
meta = pd.DataFrame([r for r, _ in N])
agg = lambda M: np.stack([M[idx].mean(0) for _, idx in N])
TE, PBG, OBS, ZN, SN = agg(dp), agg(pbg), agg(obs), agg(Z), agg(S)
TEn = [agg(x) for x in nulls]
tr = meta.treated.to_numpy(); sec = meta.section.to_numpy()
P(f'   treated nodules {tr.sum()} (median {int(meta.n[tr].median())} cells), control nodules {(~tr).sum()}')
meta.to_csv(OUT / f'nodules_{TAG}.csv', index=False)
np.savez(OUT / f'nodules_{TAG}.npz', TE=TE, PBG=PBG, OBS=OBS, Z=ZN, S=SN, types=np.array(types))

def centre(X, g):
    X = np.asarray(X, float).copy()
    for u in np.unique(g): X[g == u] -= X[g == u].mean(0)
    return X

T = TE[tr]; st = sec[tr]; mt = meta[tr].reset_index(drop=True)
P('\n== Step 1: treatment-effect structure over treated nodules ==')
mean_te = pd.Series(T.mean(0), index=types).sort_values()
P('   mean TE (pooled; includes any run-level offset): most negative ' +
  ', '.join(f'{k} {v:+.4f}' for k, v in mean_te.head(4).items()) + ' | most positive ' +
  ', '.join(f'{k} {v:+.4f}' for k, v in mean_te.tail(4)[::-1].items()))
secmean = np.stack([T[st == u].mean(0) for u in np.unique(st)])
cos = [np.dot(a_, b_) / np.linalg.norm(a_) / np.linalg.norm(b_) for i, a_ in enumerate(secmean) for b_ in secmean[i + 1:]]
P(f'   section-mean TE direction, pairwise cosine across sections: median {np.median(cos):+.2f} (range {min(cos):+.2f} to {max(cos):+.2f})')
Tc = centre(T, st)
pca = PCA(4).fit(Tc); sc_ = pca.transform(Tc)
P('   within-section heterogeneity (section-centred PCA), variance explained ' + ', '.join(f'{v:.0%}' for v in pca.explained_variance_ratio_))
for k in range(2):
    ld = pd.Series(pca.components_[k], index=types).sort_values()
    P(f'   PC{k + 1}: - ' + ', '.join(f'{i} {v:+.2f}' for i, v in ld.head(4).items()) + ' | + ' +
      ', '.join(f'{i} {v:+.2f}' for i, v in ld.tail(4)[::-1].items()))
reps = []
for u in np.unique(st):
    pc = PCA(1).fit(Tc[st == u]).components_[0]; reps.append(pc)
rc = [abs(np.dot(a_, b_)) for i, a_ in enumerate(reps) for b_ in reps[i + 1:]]
P(f'   PC1 fitted separately per section, |cosine| between sections: median {np.median(rc):.2f} (range {min(rc):.2f}-{max(rc):.2f})')

P('\n== Step 2a: bookkeeping -- does the TE track observed - p_bg across nodules (within section)? ==')
D = OBS[tr] - PBG[tr]
rr = []
for j, t in enumerate(types):
    if np.std(T[:, j]) == 0: continue
    rr.append((t, spearmanr(centre(T[:, [j]], st).ravel(), centre(D[:, [j]], st).ravel()).statistic, abs(T[:, j]).mean() / max(abs(D[:, j]).mean(), 1e-9)))
R = pd.DataFrame(rr, columns=['type', 'rho', 'size_ratio']).sort_values('rho', ascending=False)
P(f'   per neighbour type, Spearman(TE, observed - p_bg): median {R.rho.median():+.2f}; '
  f'top {", ".join(f"{r.type} {r.rho:+.2f}" for r in R.head(5).itertuples())}')
P(f'   |TE| / |observed - p_bg|: median {R.size_ratio.median():.2f}')

def partial(x, y, C):
    g = st
    def res(v):
        r = np.zeros(len(v))
        for u in np.unique(g):
            m = g == u; rv = rankdata(v[m]); Cm = np.c_[np.ones(m.sum()), np.column_stack([rankdata(c[m]) for c in C.T])]
            r[m] = rv - Cm @ np.linalg.lstsq(Cm, rv, rcond=None)[0]
        return r
    return np.corrcoef(res(x), res(y))[0, 1]

P('\n== Step 2b: TE vs independent biology (treated nodules; Spearman within each section, then partial on size + density) ==')
cov = np.c_[np.log(mt.n), mt.density]
feats = {'TE PC1': sc_[:, 0], 'TE PC2': sc_[:, 1], '|TE|': np.linalg.norm(T, axis=1)}
for t in ['CD8_T', 'T_cell', 'NK', 'Macrophage_alveolar', 'Macrophage_interstitial', 'moMacrophage', 'Neutrophil',
          'Fibroblast_CAF', 'B_cell', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']:
    feats[f'TE {t}'] = T[:, tix[t]]
bio = {'MAPK MPAS': mt.MAPK_MPAS.to_numpy(), 'responder rate': mt.resp_rate.to_numpy(),
       'Itgb6 fraction': mt.itgb6.to_numpy(), 'proliferation': mt.Proliferation.to_numpy(), 'EMT': mt.EMT.to_numpy()}
rows2 = []
for fn, fv in feats.items():
    for bn, bv in bio.items():
        per = [spearmanr(fv[st == u], bv[st == u]).statistic for u in np.unique(st)]
        rows2.append(dict(te=fn, biology=bn, **{u: r for u, r in zip(np.unique(st), per)}, median=np.median(per),
                          same_sign=int(max(np.sum(np.array(per) > 0), np.sum(np.array(per) < 0))), partial=partial(fv, bv, cov)))
B = pd.DataFrame(rows2); B.to_csv(OUT / f'te_vs_biology_{TAG}.csv', index=False)
P('   ' + B.round(2).to_string(index=False).replace('\n', '\n   '))
P('   covariates alone: ' + ', '.join(f'{bn} vs log size {np.median([spearmanr(np.log(mt.n)[st == u], bv[st == u]).statistic for u in np.unique(st)]):+.2f}'
                                     f' / density {np.median([spearmanr(mt.density[st == u], bv[st == u]).statistic for u in np.unique(st)]):+.2f}'
                                     for bn, bv in bio.items()))

P('\n== Step 3 (preliminary): can the baseline state predict the nodule TE? leave-section-out ridge, section-centred ==')
def lso(X, Y):
    X, Y = centre(X, st), centre(Y, st); pred = np.zeros_like(Y)
    for u in np.unique(st):
        te_, tr_ = st == u, st != u
        ss = StandardScaler().fit(X[tr_]); m = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(ss.transform(X[tr_]), Y[tr_])
        pred[te_] = m.predict(ss.transform(X[te_]))
    per = []
    for u in np.unique(st):
        m = st == u; per.append(np.mean([spearmanr(pred[m, j], Y[m, j]).statistic for j in range(Y.shape[1])]))
    return np.array(per)
targets = {'TE PC1-2': sc_[:, :2], 'TE all types': T}
for tn, Y in targets.items():
    out = {k: lso(X, Y) for k, X in [('z (baseline)', ZN[tr]), ('p_bg (baseline)', PBG[tr]), ('s (ceiling)', SN[tr])]}
    P(f'   {tn:13s} ' + ' | '.join(f'{k} {v.mean():+.2f} [' + ' '.join(f'{x:+.2f}' for x in v) + ']' for k, v in out.items()))
for k, Tn in enumerate(TEn):
    Yn = Tn[tr]; scn = pca.transform(centre(Yn, st))[:, :2]
    P(f'   NULL shuffled-s {k}: z -> null TE PC1-2 {lso(ZN[tr], scn).mean():+.2f} | z -> null TE all types {lso(ZN[tr], Yn).mean():+.2f}'
      f' | s -> null TE PC1-2 {lso(SN[tr], scn).mean():+.2f}')
P('done')
