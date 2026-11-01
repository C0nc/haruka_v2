#!/usr/bin/env python3
"""Lung fibrosis: does s (and not z) follow the gradient towards a pathologist-annotated lesion?
Fit: Unaffected vs all fibrosis, batch = core, one-way penalty 10.  Fibrosis cores only (condition cells).
GT: GSE250346 H&E annotations.  For lesion type L and each fibrosis core with >= 30 L cells: distance of every cell
NOT annotated L to the nearest L cell (um), cells within 0-400 um kept.
Latents standardised and centred per core x cell type (removes patient, stage, cell identity).
A  supervised: per L x cell type, ridge on centred z or s -> log1p(distance), leave-one-donor-out;
   readout = median over cores of the within-core Spearman(held-out prediction, distance) (sign: closer = lower).
B  unsupervised: PCA of the centred latent (per L, pooled over cell types); |Spearman| of PC1..PC5 with distance,
   computed within core x cell type and averaged.  PC1 reported, and the best of PC1-5.
C  biology: per L, genes whose within-core-and-type expression best tracks distance (Spearman, pooled); and how well
   the held-out s / z predictions track the 10 most distance-dependent genes.
"""
import os, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/lesion_gradient'; OUT.mkdir(exist_ok=True)
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
log = open(OUT / 'lesion_gradient.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
pd.set_option('display.width', 250)
stage = o.sample_affect.astype(str).to_numpy(); F = np.flatnonzero(stage != 'Unaffected')
core = o['sample'].astype(str).to_numpy(); ctA = o.cell_type.astype(str).to_numpy(); donA = o.donor.astype(str).to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]
pos = pd.Series(np.arange(len(o)), index=o.index)
he = pd.read_csv('/project/zhihuanglab/yancui/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type'])
he['i'] = pos.reindex(he.obs).to_numpy()
Zm = np.load(MD / 'background.npy', mmap_mode='r'); Sm = np.load(MD / 'salient.npy', mmap_mode='r')
genes = np.asarray(a.var_names).astype(str)
def centre(M, g):
    M = np.asarray(M, np.float64).copy()
    for k in pd.Series(np.arange(len(g))).groupby(g).indices.values(): M[k] -= M[k].mean(0)
    return M
LES = ['fibroblastic_focus', 'mixed_inflammation', 'TLS', 'remodeled_epithelium', 'microscopic_honeycombing', 'severe_fibrosis']
fibcores = set(core[F]); rng = np.random.default_rng(0)
A_rows, B_rows, C_rows = [], [], []
for L in LES:
    li = he.i[he.annotation_type == L].to_numpy()
    idx_all, dist_all = [], []
    for c in sorted(fibcores):
        lc = li[core[li] == c]
        if len(lc) < 30: continue
        ci = np.flatnonzero(core == c); ci = ci[~np.isin(ci, lc)]
        d, _ = cKDTree(xy[lc]).query(xy[ci]); k = d <= 400
        idx_all.append(ci[k]); dist_all.append(d[k])
    if not idx_all: continue
    idx = np.concatenate(idx_all); dist = np.concatenate(dist_all); o_ = np.argsort(idx); idx, dist = idx[o_], dist[o_]
    if len(idx) > 400000: k = np.sort(rng.choice(len(idx), 400000, replace=False)); idx, dist = idx[k], dist[k]
    cc, ct, dn = core[idx], ctA[idx], donA[idx]; grp = pd.Series(cc).str.cat(ct, sep='|').to_numpy()
    y = np.log1p(dist)
    Lat = {'z': centre(StandardScaler().fit_transform(np.asarray(Zm[idx])), grp), 's': centre(StandardScaler().fit_transform(np.asarray(Sm[idx])), grp)}
    P(f'\n### {L}: {len(set(cc))} cores, {len(set(dn))} donors, {len(idx)} cells within 400 um (median distance {np.median(dist):.0f} um)')
    # A supervised per cell type
    preds = {k: np.full(len(idx), np.nan) for k in Lat}
    for t in np.unique(ct):
        m = np.flatnonzero(ct == t)
        if len(m) < 1000 or len(np.unique(dn[m])) < 4: continue
        rec = dict(lesion=L, cell_type=t, n=len(m), cores=len(np.unique(cc[m])))
        for k, X in Lat.items():
            pr = np.full(len(m), np.nan)
            for d in np.unique(dn[m]):
                te = dn[m] == d
                pr[te] = Ridge(alpha=10).fit(X[m][~te], y[m][~te]).predict(X[m][te])
            preds[k][m] = pr
            rs = [spearmanr(pr[cc[m] == q], y[m][cc[m] == q]).statistic for q in np.unique(cc[m]) if (cc[m] == q).sum() >= 50]
            rec[k] = np.nanmedian(rs)
        A_rows.append(rec)
    t = pd.DataFrame([r for r in A_rows if r['lesion'] == L])
    if len(t):
        P(f'  A supervised: {len(t)} cell types | median within-core rho z {t.z.median():.3f}  s {t.s.median():.3f} | s > z in {(t.s > t.z).sum()}/{len(t)}'
          + (f' | Wilcoxon p {wilcoxon(t.s, t.z).pvalue:.2g}' if len(t) >= 6 else ''))
        P(t.sort_values('n', ascending=False).head(8)[['cell_type', 'n', 'cores', 'z', 's']].round(3).to_string(index=False))
    # B unsupervised PCs
    for k, X in Lat.items():
        pcs = PCA(5, random_state=0).fit_transform(X)
        r = []
        for j in range(5):
            v = pd.DataFrame({'p': pcs[:, j], 'y': y, 'g': grp}).groupby('g').filter(lambda q: len(q) >= 50)
            rr = v.groupby('g').apply(lambda q: spearmanr(q.p, q.y).statistic)
            w = v.groupby('g').size().reindex(rr.index)
            r.append(np.abs(np.average(rr.fillna(0), weights=w)))   # sign-consistent average, then |.|
        B_rows.append(dict(lesion=L, latent=k, PC1=r[0], best=max(r), best_pc=int(np.argmax(r)) + 1))
    b = pd.DataFrame([r for r in B_rows if r['lesion'] == L]).set_index('latent')
    P(f'  B unsupervised |rho(PC, distance)| (sign-consistent, within core x type): PC1 z {b.loc["z","PC1"]:.3f} s {b.loc["s","PC1"]:.3f}'
      f' | best of PC1-5 z {b.loc["z","best"]:.3f} (PC{b.loc["z","best_pc"]}) s {b.loc["s","best"]:.3f} (PC{b.loc["s","best_pc"]})')
    # C genes along the gradient
    sub = np.sort(rng.choice(len(idx), min(60000, len(idx)), replace=False))
    Xr = a[idx[sub]].layers['counts']; Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
    Xn = centre(np.log1p(Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e4), grp[sub])
    yc = centre(y[sub][:, None], grp[sub])[:, 0]
    rg = np.array([spearmanr(Xn[:, j], yc).statistic for j in range(Xn.shape[1])]); rg = np.nan_to_num(rg)
    top_near = genes[np.argsort(rg)[:10]]; top_far = genes[np.argsort(-rg)[:5]]
    P('  C genes up NEAR the lesion (within core x type):', ', '.join(f'{g} {rg[list(genes).index(g)]:+.3f}' for g in top_near))
    P('    genes up far from it:', ', '.join(f'{g} {rg[list(genes).index(g)]:+.3f}' for g in top_far))
    near_score = Xn[:, np.argsort(rg)[:10]].mean(1)
    for k in Lat:
        p = preds[k][sub]; ok = np.isfinite(p)
        C_rows.append(dict(lesion=L, latent=k, rho_pred_vs_nearGenes=spearmanr(p[ok], near_score[ok]).statistic))
    c = pd.DataFrame([r for r in C_rows if r['lesion'] == L]).set_index('latent')
    P(f'    held-out distance prediction vs near-lesion gene score: z {c.loc["z","rho_pred_vs_nearGenes"]:+.3f}  s {c.loc["s","rho_pred_vs_nearGenes"]:+.3f} (negative = tracks it)')
    pd.DataFrame({'gene': genes, 'rho_distance': rg}).to_csv(OUT / f'genes_{L}.csv', index=False)
pd.DataFrame(A_rows).to_csv(OUT / 'A_supervised.csv', index=False); pd.DataFrame(B_rows).to_csv(OUT / 'B_pcs.csv', index=False); pd.DataFrame(C_rows).to_csv(OUT / 'C_genes.csv', index=False)
A = pd.DataFrame(A_rows)
P(f'\nALL lesion x cell type ({len(A)}): median z {A.z.median():.3f}  s {A.s.median():.3f} | s > z in {(A.s > A.z).sum()}/{len(A)} | Wilcoxon p {wilcoxon(A.s, A.z).pvalue:.2g}')
P('done')
