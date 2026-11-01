#!/usr/bin/env python3
"""Is s an intra-fibrosis inflammatory-state axis, and is it a main source of within-patient heterogeneity?
Fibrosis cores, cells with one H&E region label.  Latents centred per core x cell type (removes patient, core,
stage and cell identity; what is left is state variation of a cell type inside one tissue sample).
a  share of that within-core-and-type variance explained by the H&E region (one-hot), for z and s.
b  inflammation axis: logistic (C=0.1, balanced) on the centred latent, inflamed (mixed_inflammation + TLS +
   granuloma) vs fibrosis, GroupKFold(5) over donors -> out-of-fold score for every labelled cell;
   held-out within-core AUROC, and the share of centred-latent variance lying along the axis.
c  gene check (expression is the validation, not a competitor): within cell type, Spearman of the out-of-fold score
   with inflammatory genes vs fibrotic genes (log1p CP10k, centred per core x cell type).
d  heterogeneity of the s score on ALL fibrosis cells (axis refit on all labelled cells, applied to uncentred s
   minus the cell-type mean): variance between donors / between cores within donor / within core.
"""
import os, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/he_region'
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
pos = pd.Series(np.arange(len(o)), index=o.index)
Zm = np.load(MD / 'background.npy', mmap_mode='r'); Sm = np.load(MD / 'salient.npy', mmap_mode='r')
log = open(OUT / 's_inflammation.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
d = pd.read_csv('/project/zhihuanglab/yancui/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type', 'stage', 'core'])
d = d[d.stage != 'Unaffected'].drop_duplicates(['obs', 'annotation_type']); d = d[~d.obs.duplicated(keep=False)]
d['i'] = pos.reindex(d.obs).to_numpy(); d = d.sort_values('i')
idx = d.i.to_numpy(); ct = o.cell_type.astype(str).to_numpy()[idx]; core = d.core.to_numpy(); don = o.donor.astype(str).to_numpy()[idx]
reg = d.annotation_type.to_numpy(); grp = pd.Series(core).str.cat(ct, sep='|').to_numpy()
def centre(M, g):
    M = np.asarray(M, np.float64).copy(); s = pd.Series(np.arange(len(g))).groupby(g).indices
    for k in s.values(): M[k] -= M[k].mean(0)
    return M
L = {'z': centre(StandardScaler().fit_transform(np.asarray(Zm[idx])), grp), 's': centre(StandardScaler().fit_transform(np.asarray(Sm[idx])), grp)}
P(f'labelled fibrosis cells {len(idx)}, cores {len(set(core))}, donors {len(set(don))}')

# a
D = pd.get_dummies(reg).to_numpy(float); D = centre(D, grp)
P('\n== a. share of within-core x cell-type latent variance explained by H&E region ==')
for k, X in L.items():
    B, *_ = np.linalg.lstsq(D, X, rcond=None); P(f'  {k}: {1 - ((X - D @ B) ** 2).sum() / (X ** 2).sum():.3f}')

# b
INF = ['mixed_inflammation', 'TLS', 'granuloma']
m = np.isin(reg, INF + ['fibrosis']); y = np.isin(reg[m], INF)
P(f'\n== b. inflammation axis: inflamed {y.sum()} vs fibrosis {(~y).sum()} cells, {len(set(core[m]))} cores ==')
oof = {}; W = {}
for k, X in L.items():
    Xm = X[m]; sc = np.zeros(m.sum())
    for tr, te in GroupKFold(5).split(Xm, y, don[m]):
        sc[te] = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced').fit(Xm[tr], y[tr]).decision_function(Xm[te])
    cm = core[m]; aucs = [roc_auc_score(y[cm == c], sc[cm == c]) for c in np.unique(cm) if 0 < y[cm == c].sum() < (cm == c).sum() and min(y[cm == c].sum(), (~y[cm == c]).sum()) >= 20]
    w = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced').fit(Xm, y).coef_[0]; w /= np.linalg.norm(w); W[k] = w
    share = np.var(X @ w) / X.var(0).sum()
    oof[k] = sc
    P(f'  {k}: held-out-donor within-core AUROC median {np.median(aucs):.3f} over {len(aucs)} cores; axis carries {share:.1%} of centred {k} variance'
      f' (a random direction carries {1/X.shape[1]:.1%})')
# where the s axis sits among the s PCs
from sklearn.decomposition import PCA
pc = PCA(10, random_state=0).fit(L['s']); cos = np.abs(pc.components_ @ W['s'])
P('  |cos| of the s inflammation axis with within-core s PC1..10:', np.round(cos, 2).tolist(), '| PC var share', np.round(pc.explained_variance_ratio_, 3).tolist())

# c genes
genes = np.asarray(a.var_names).astype(str); im = idx[m]
Xr = a[im].layers['counts']; Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
Xn = centre(np.log1p(Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e4), grp[m])
GS = {'inflammatory': ['CXCL9', 'CXCL10', 'CXCL13', 'STAT1', 'IFIT1', 'IFIT3', 'CCL19', 'CCL21', 'CD74', 'HLA-DRA', 'CCL5', 'GZMB', 'IL7R', 'CD52'],
      'fibrotic': ['COL1A1', 'COL1A2', 'COL3A1', 'POSTN', 'CTHRC1', 'FN1', 'SPP1', 'MMP7', 'KRT17', 'TGFB1']}
P('\n== c. within cell type, Spearman(out-of-fold inflammation score, gene), median over cell types with >= 500 cells ==')
cts = [c for c in np.unique(ct[m]) if (ct[m] == c).sum() >= 500]
rows = []
for g in sum(GS.values(), []):
    if g not in genes: continue
    j = list(genes).index(g); rec = dict(gene=g, set=[k for k, v in GS.items() if g in v][0])
    for k in L:
        rec[k] = np.nanmedian([spearmanr(oof[k][ct[m] == c], Xn[ct[m] == c, j]).statistic for c in cts])
    rows.append(rec)
G = pd.DataFrame(rows); G.to_csv(OUT / 's_inflammation_genes.csv', index=False); P(G.round(3).to_string(index=False))
P(G.groupby('set')[['z', 's']].median().round(3).to_string())

# d heterogeneity on all fibrosis cells
F = np.flatnonzero(o.sample_affect.astype(str).to_numpy() != 'Unaffected'); rng = np.random.default_rng(0); F = np.sort(rng.choice(F, 400000, replace=False))
ctF = o.cell_type.astype(str).to_numpy()[F]; coreF = o['sample'].astype(str).to_numpy()[F]; donF = o.donor.astype(str).to_numpy()[F]
P('\n== d. where does the inflammation score vary? (400k fibrosis cells, score minus cell-type mean) ==')
for k, M in [('z', Zm), ('s', Sm)]:
    X = StandardScaler().fit_transform(np.asarray(M[F])); sc = centre((X @ W[k])[:, None], ctF)[:, 0]
    df = pd.DataFrame({'s': sc, 'd': donF, 'c': coreF}); tot = df.s.var()
    bd = df.groupby('d').s.transform('mean'); bc = df.groupby('c').s.transform('mean')
    P(f'  {k}: between donor {bd.var() / tot:.1%} | between cores within donor {(bc - bd).var() / tot:.1%} | within core {(df.s - bc).var() / tot:.1%}')
    # within-core: how much of it is the H&E region (labelled subset)
P('done')
