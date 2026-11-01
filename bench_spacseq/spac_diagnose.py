#!/usr/bin/env python3
"""SPAC-seq lung: is the missing knockout effect a FIT problem or a DATASET problem?

Dataset (model-free, full tissue from the raw slices):
  D1 guide calls: bins with exactly one guide read in tumour bins (the anchors), their guide UMIs
  D2 are same-target guide bins spatially clustered (clonal lesions) beyond a label-permutation null?
  D3 construction: the fitted h5ad keeps only anchor contexts (anchor + 8 NN); how far is its
     20-NN "neighbourhood" vs the real tissue, and how similar are the two compositions?
  D4 full-tissue KO effect on neighbourhood composition (all eligible anchors, 20 NN in full tissue):
     target-specific profile replication (a) across slices and (b) between the 2 guides of a target
Fit:
  D5 can raw expression / z / s of anchors tell the target apart (leave-one-slice-out)?
"""
import os, sys, json, itertools, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); B = H / 'bench_spacseq'; sys.path.insert(0, str(B))
import prepare_multislice_all_guides as base
OUT = B / 'results_multislice/cf_te'; log = open(OUT / 'diagnose.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
a = ad.read_h5ad(B / 'data/spac_lung_functional_guides_k8_hvg3000.h5ad')
TARGETS = a.uns['functional_guide_design']['targets']; GUIDES = a.uns['functional_guide_design']['guides_by_target']
rng = np.random.default_rng(0); K = 20
full = {}
for sid in base.SLICES:
    d = base.load_slice(B / 'raw', sid)
    ct = d['cells']['cell_type'].astype(str).to_numpy(); xy = d['coords']
    tgt = np.full(len(ct), '', object); gname = np.full(len(ct), '', object); v = d['single_feature'] >= 0
    tgt[v] = d['feature_target'][d['single_feature'][v]]; gname[v] = d['names'][d['single_feature'][v]]
    umi = np.zeros(len(ct)); umi[v] = np.asarray(d['counts'][np.where(v)[0]].sum(1)).ravel()
    full[sid] = dict(ct=ct, xy=xy, tgt=tgt, guide=gname, umi=umi, single=d['single'], tumor=d['tumor'], nnz=d['nnz'],
                     names=np.asarray(d['guide'].obs_names).astype(str))

P('-- D1 guide calls (full slices) --')
for sid, f in full.items():
    elig = f['single'] & f['tumor']; anyg = f['nnz'] > 0
    u = f['umi'][elig]
    P(f'   {sid}: {len(f["ct"])} bins | any guide {anyg.mean():.1%} | >1 guide feature {(f["nnz"] > 1).mean():.1%} | '
      f'eligible single-guide tumour bins {elig.sum()} (UMI = 1: {np.mean(u == 1):.0%}, median {np.median(u):.0f}) | '
      f'targeting {np.isin(f["tgt"][elig], TARGETS).sum()}, NT {(f["tgt"][elig] == "NT").sum()}')

P('\n-- D2 spatial clustering of guide labels (eligible bins, 10 nearest other eligible bins) --')
for sid, f in full.items():
    e = np.where(f['single'] & f['tumor'] & (np.isin(f['tgt'], TARGETS) | (f['tgt'] == 'NT')))[0]
    _, nb = cKDTree(f['xy'][e]).query(f['xy'][e], k=11); nb = nb[:, 1:]
    dist10 = np.median(np.linalg.norm(f['xy'][e][nb[:, -1]] - f['xy'][e], axis=1))
    for key in ['tgt', 'guide']:
        lab = f[key][e]; obs_ = np.mean(lab[nb] == lab[:, None])
        null = [np.mean(l[nb] == l[:, None]) for l in (rng.permutation(lab) for _ in range(200))]
        P(f'   {sid} same-{"target" if key == "tgt" else "guide "} among 10 NN: {obs_:.3f} vs permuted {np.mean(null):.3f} '
          f'(z = {(obs_ - np.mean(null)) / np.std(null):.1f}); median distance to 10th NN {dist10:.0f} bins')

P('\n-- D3 construction: fitted subset vs full tissue (k = 20) --')
cats = sorted(set(a.obs.cell_type.astype(str)))
sub_comp, full_comp = [], []
for sid, f in full.items():
    m = a.obs.slice_id.astype(str).to_numpy() == sid
    names = np.array([n.split(':', 1)[1] for n in a.obs_names[m]]); idx = pd.Index(f['names']).get_indexer(names)
    xs = f['xy'][idx]; dsub, nsub = cKDTree(xs).query(xs, k=K + 1); dful, nful = cKDTree(f['xy']).query(xs, k=K + 1)
    code_s = pd.Categorical(a.obs.cell_type.astype(str).to_numpy()[m], categories=cats).codes
    code_f = pd.Categorical(f['ct'], categories=cats).codes
    cs = np.stack([(code_s[nsub[:, 1:]] == j).mean(1) for j in range(len(cats))], 1)
    cf = np.stack([(code_f[nful[:, 1:]] == j).mean(1) for j in range(len(cats))], 1)
    r = [np.corrcoef(cs[:, j], cf[:, j])[0, 1] for j in range(len(cats))]
    P(f'   {sid}: distance to 20th NN median subset {np.median(dsub[:, -1]):.1f} vs full tissue {np.median(dful[:, -1]):.1f} bins | '
      f'shared neighbours {np.mean([len(set(nsub[i, 1:]) & set(idx[nful[i, 1:]].tolist() and [])) for i in range(1)]) if False else ""}'
      f'per-type corr(subset comp, full-tissue comp): ' + ', '.join(f'{c} {v:+.2f}' for c, v in zip(cats, r)))

P('\n-- D4 full-tissue KO effect on neighbourhood composition (all eligible anchors, own bin excluded) --')
rows = []
for sid, f in full.items():
    code_f = pd.Categorical(f['ct'], categories=cats).codes
    e = np.where(f['single'] & f['tumor'] & (np.isin(f['tgt'], TARGETS) | (f['tgt'] == 'NT')))[0]
    _, nb = cKDTree(f['xy']).query(f['xy'][e], k=K + 1)
    comp = np.stack([(code_f[nb[:, 1:]] == j).mean(1) for j in range(len(cats))], 1)
    nt = comp[f['tgt'][e] == 'NT'].mean(0)
    for t in TARGETS:
        for g in list(GUIDES[t]) + ['both']:
            mm = (f['tgt'][e] == t) & ((f['guide'][e] == g) if g != 'both' else True)
            if mm.sum() >= 10:
                # independent lesions = connected components of these anchors at <= 4 bins
                pts = f['xy'][e[mm]]; pairs = cKDTree(pts).query_pairs(4.0, output_type='ndarray')
                from scipy.sparse.csgraph import connected_components
                Adj = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts))) if len(pairs) else sp.coo_matrix((len(pts), len(pts)))
                ncomp, lab_ = connected_components(Adj, directed=False); big = int((np.bincount(lab_) >= 5).sum())
                rows.append(dict(slice=sid, target=t, guide=g, n=int(mm.sum()), lesions=big, **dict(zip(cats, comp[mm].mean(0) - nt))))
F = pd.DataFrame(rows); F.to_csv(OUT / 'fulltissue_te.csv', index=False)
def specific(X): return X - X.mean(0)
both = F[F.guide == 'both']
P('   anchors per target x slice: ' + both.pivot(index='target', columns='slice', values='n').to_string().replace('\n', '\n   '))
P('   lesions (>= 5 anchors within 4 bins) per target x slice: ' + both.pivot(index='target', columns='slice', values='lesions').to_string().replace('\n', '\n   '))
for s1, s2 in itertools.combinations(full, 2):
    A_ = specific(both[both.slice == s1].set_index('target')[cats]); B_ = specific(both[both.slice == s2].set_index('target')[cats])
    A_, B_ = A_.align(B_, join='inner', axis=0); A_, B_ = A_.to_numpy(), B_.to_numpy()
    r = np.corrcoef(A_.ravel(), B_.ravel())[0, 1]; null = [np.corrcoef(A_.ravel(), B_[rng.permutation(len(B_))].ravel())[0, 1] for _ in range(5000)]
    P(f'   (a) across slices {s1} vs {s2}: r = {r:+.2f}, permutation p = {np.mean(np.array(null) >= r):.3f}')
G = F[F.guide != 'both']; gr = []
for sid in full:
    X = G[G.slice == sid]; X = X[X.target.map(X.target.value_counts()) == 2]
    if X.target.nunique() < 3: continue
    X = X.assign(**{c: X[c] - X[c].mean() for c in cats})                     # target-specific
    g1 = X.groupby('target').nth(0).set_index('target')[cats]; g2 = X.groupby('target').nth(1).set_index('target')[cats]
    g1, g2 = g1.align(g2, join='inner', axis=0)
    r = np.corrcoef(g1.to_numpy().ravel(), g2.to_numpy().ravel())[0, 1]
    null = [np.corrcoef(g1.to_numpy().ravel(), g2.to_numpy()[rng.permutation(len(g2))].ravel())[0, 1] for _ in range(5000)]
    gr.append(r); P(f'   (b) guide 1 vs guide 2 of the same target, {sid} ({len(g1)} targets): r = {r:+.2f}, permutation p = {np.mean(np.array(null) >= r):.3f}')

P('\n-- D5 target identity of anchor bins, leave-one-slice-out balanced accuracy (chance 1/8 = 0.125); KO vs NT AUROC --')
MD = H / 'results_real_compgated_nb_final/spac_lung_functional_guides_k8_hvg3000/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
anc = a.obs.is_anchor.to_numpy(bool); pt = a.obs.perturbation.astype(str).to_numpy(); sl = a.obs.slice_id.astype(str).to_numpy()
C = a.layers['counts']; C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
L = C.multiply(1e4 / np.asarray(C.sum(1)).ravel().clip(1)[:, None]).tocsr(); L.data = np.log1p(L.data)
Xs = {'expression PCA50': PCA(50, random_state=0).fit_transform(L[anc].toarray()),
      'z': np.load(MD / 'background.npy')[anc], 's': np.load(MD / 's'.join(['', 'alient.npy']))[anc] if False else np.load(MD / 'salient.npy')[anc]}
y = pt[anc]; g = sl[anc]
for k, X in Xs.items():
    pr = np.empty(len(y), object); sc_ = np.zeros(len(y)); ko = y != 'NT'
    for h in np.unique(g):
        tr, te = g != h, g == h
        ss = StandardScaler().fit(X[tr])
        m = tr & ko
        pr[te & ko] = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced').fit(ss.transform(X[m]), y[m]).predict(ss.transform(X[te & ko]))
        sc_[te] = LogisticRegression(max_iter=500, C=0.1, class_weight='balanced').fit(ss.transform(X[tr]), ko[tr]).decision_function(ss.transform(X[te]))
    P(f'   {k:18s} target bal. acc {balanced_accuracy_score(y[ko], pr[ko].astype(str)):.3f} | KO vs NT AUROC {roc_auc_score(ko, sc_):.3f}')
P('done')
