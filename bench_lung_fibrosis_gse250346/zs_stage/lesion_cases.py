#!/usr/bin/env python3
"""Deep dive on the strongest unsupervised lesion-gradient cases (lesion_gradient_unsup.py).
Each case = (lesion L, cell type, axis).  Cells, centring and axes exactly as in lesion_gradient_unsup.py
(non-L cells within 400 um of L in fibrosis cores with >= 30 L cells; latent standardised + centred per core;
PCA fit on that cell type).  Per case, for the s axis and the same-numbered z axis:
 1 robustness: per-core Pearson r(axis, distance); share of cores with the pooled sign; leave-one-core-out signed mean;
   bootstrap over cores (2000) 95% CI.
 2 spatial null: per core, the lesion cell coordinates are translated by a random offset with wrap-around inside the core's
   bounding box (200 draws); cells fixed, axis fixed; null distribution of the sign-consistent |mean r|.
 3 distance profile: mean axis value (oriented so that + = towards the lesion) in bins 0-25-50-100-200-400 um, mean +- SE over cores.
 4 composition control: partial r(axis, distance | observed k=20 neighbour cell-type shares), residualised within core.
 5 genes: Spearman of within-core-centred log1p CP10k with the s axis (top 10 each end) and with distance.
 6 decoded salient composition shift: Spearman of each neighbour-type shift with the s axis (top 4 each end).
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
_HW = _os.environ.get('HARUKA_WORK') or ('/project/zhihuanglab/yancui' if _os.path.isdir('/project/zhihuanglab/yancui') else _HR + '/work')
import os, json, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')
H = Path(f'{_HR}'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/lesion_gradient/cases'; OUT.mkdir(parents=True, exist_ok=True)
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
types = json.load(open(MD / 'meta.json'))['cell_type_categories']; genes = np.asarray(a.var_names).astype(str)
log = open(OUT / 'lesion_cases.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
stage = o.sample_affect.astype(str).to_numpy(); core = o['sample'].astype(str).to_numpy(); ctA = o.cell_type.astype(str).to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]
pos = pd.Series(np.arange(len(o)), index=o.index)
he = pd.read_csv(f'{_HW}/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type']); he['i'] = pos.reindex(he.obs).to_numpy()
Zm = np.load(MD / 'background.npy', mmap_mode='r'); Sm = np.load(MD / 'salient.npy', mmap_mode='r'); SH = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r')
cti = pd.Categorical(ctA, categories=types).codes
def centre(M, g):
    M = np.asarray(M, np.float64).copy()
    for k in pd.Series(np.arange(len(g))).groupby(g).indices.values(): M[k] -= M[k].mean(0)
    return M
def per_core_r(ax, d, cc):
    out = {}
    for q in np.unique(cc):
        m = cc == q
        if m.sum() >= 50 and ax[m].std() > 0: out[q] = (np.corrcoef(ax[m], d[m])[0, 1], m.sum())
    return out
def smean(rr, keys=None):
    keys = list(rr) if keys is None else keys
    return np.average([rr[k][0] for k in keys], weights=[rr[k][1] for k in keys])
CASES = [('TLS', 'CD4+ T-cells', 0), ('fibroblastic_focus', 'Alveolar FBs', 0), ('microscopic_honeycombing', 'Alveolar FBs', 0),
         ('microscopic_honeycombing', 'Interstitial Macrophages', 0), ('fibroblastic_focus', 'Capillary', 1)]
fibcores = set(core[stage != 'Unaffected']); rng = np.random.default_rng(0); summary = []
for L, T, pcj in CASES:
    li = he.i[he.annotation_type == L].to_numpy(); ii, dd, lesion_pts = [], [], {}
    for c in sorted(fibcores):
        lc = li[core[li] == c]
        if len(lc) < 30: continue
        ci = np.flatnonzero(core == c); ci = ci[~np.isin(ci, lc)]
        d, _ = cKDTree(xy[lc]).query(xy[ci]); k = (d <= 400) & (ctA[ci] == T)
        ii.append(ci[k]); dd.append(d[k]); lesion_pts[c] = xy[lc]
    ii = np.concatenate(ii); dd = np.concatenate(dd)
    if len(ii) > 15000:   # same subsample rule as lesion_gradient_unsup.py
        m = np.sort(np.random.default_rng(0).choice(len(ii), 15000, replace=False)); ii, dd = ii[m], dd[m]
    o_ = np.argsort(ii); ii, dd = ii[o_], dd[o_]; cc = core[ii]
    P(f'\n######## {T} -> {L}: axis PC{pcj+1}; {len(ii)} cells, {len(np.unique(cc))} cores, median distance {np.median(dd):.0f} um')
    axes = {}
    for k, M in [('z', Zm), ('s', Sm)]:
        X = centre(StandardScaler().fit_transform(np.asarray(M[ii])), cc); axes[k] = PCA(2, random_state=0).fit_transform(X)[:, pcj]
    # orient each axis so that + = towards the lesion (negative r with distance)
    for k in axes:
        if smean(per_core_r(axes[k], dd, cc)) > 0: axes[k] = -axes[k]
    rec = dict(lesion=L, cell_type=T, axis=f'PC{pcj+1}', n=len(ii), cores=len(np.unique(cc)))
    # 1 robustness
    for k, ax in axes.items():
        rr = per_core_r(ax, dd, cc); keys = list(rr); m0 = smean(rr)
        same = np.mean([np.sign(rr[q][0]) == np.sign(m0) for q in keys])
        loo = [smean(rr, [q for q in keys if q != x]) for x in keys]
        bs = [smean(rr, list(rng.choice(keys, len(keys), replace=True))) for _ in range(2000)]
        # 2 spatial null
        null = []
        for _ in range(200):
            rn = {}
            for q in keys:
                m = cc == q; lo, hi = xy[core == q].min(0), xy[core == q].max(0); w = hi - lo
                pts = lo + np.mod(lesion_pts[q] - lo + rng.uniform(0, 1, 2) * w, w)
                dn_, _ = cKDTree(pts).query(xy[ii[m]]); rn[q] = (np.corrcoef(ax[m], dn_)[0, 1], m.sum())
            null.append(abs(smean(rn)))
        null = np.array(null)
        rec.update({f'r_{k}': abs(m0), f'cores_same_sign_{k}': same, f'loo_min_{k}': min(abs(x) for x in loo), f'ci_lo_{k}': abs(np.percentile(bs, 97.5)), f'ci_hi_{k}': abs(np.percentile(bs, 2.5)),
                    f'null_median_{k}': np.median(null), f'null_p_{k}': (1 + (null >= abs(m0)).sum()) / (1 + len(null))})
        P(f'  {k}: |r| {abs(m0):.3f} | cores with same sign {same:.0%} of {len(keys)} | leave-one-core-out min {rec[f"loo_min_{k}"]:.3f}'
          f' | bootstrap 95% CI {rec[f"ci_lo_{k}"]:.3f}-{rec[f"ci_hi_{k}"]:.3f} | spatial-shift null median {np.median(null):.3f}, p {rec[f"null_p_{k}"]:.3f}')
    # 3 distance profile
    bins = [0, 25, 50, 100, 200, 400]; lab = np.digitize(dd, bins[1:-1])
    prof = {k: pd.DataFrame({'b': lab, 'c': cc, 'v': ax}).groupby(['c', 'b']).v.mean().unstack() for k, ax in axes.items()}
    P('  profile (axis oriented + = towards lesion; mean over cores +- SE):')
    for k in prof:
        P(f'    {k}: ' + ' | '.join(f'{bins[b]}-{bins[b+1]}um {prof[k][b].mean():+.2f}+-{prof[k][b].std() / np.sqrt(prof[k][b].notna().sum()):.2f}' for b in range(5)))
    # 4 composition control
    OBSc = np.zeros((len(ii), len(types)))
    for q in np.unique(cc):
        cq = np.flatnonzero(core == q); _, nb = NearestNeighbors(n_neighbors=21).fit(xy[cq]).kneighbors(xy[ii[cc == q]])
        OBSc[cc == q] = np.stack([np.bincount(r, minlength=len(types)) for r in cti[cq][nb[:, 1:]]]) / 20.0
    Cc = centre(OBSc, cc); dc = centre(dd[:, None], cc)[:, 0]
    def resid(v): B, *_ = np.linalg.lstsq(np.c_[Cc, np.ones(len(Cc))], v, rcond=None); return v - np.c_[Cc, np.ones(len(Cc))] @ B
    dres = resid(dc)
    for k, ax in axes.items():
        rec[f'partial_r_{k}'] = abs(np.corrcoef(resid(centre(ax[:, None], cc)[:, 0]), dres)[0, 1])
    rc = abs(np.corrcoef(resid(centre(axes["s"][:, None], cc)[:, 0]), centre(axes["s"][:, None], cc)[:, 0])[0, 1])
    P(f'  composition control: partial |r| given observed k=20 neighbour shares: z {rec["partial_r_z"]:.3f}  s {rec["partial_r_s"]:.3f}'
      f' (share of s-axis variance NOT explained by composition: {rc**2:.0%})')
    # 5 genes
    Xr = a[ii].layers['counts']; Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
    Xn = centre(np.log1p(Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e4), cc)
    sa = centre(axes['s'][:, None], cc)[:, 0]
    rg = np.nan_to_num([spearmanr(Xn[:, j], sa).statistic for j in range(len(genes))]); rd = np.nan_to_num([spearmanr(Xn[:, j], -dc).statistic for j in range(len(genes))])
    P('  genes up towards the lesion end of the s axis:', ', '.join(f'{genes[j]} {rg[j]:+.2f}' for j in np.argsort(-rg)[:10]))
    P('  genes down along it:', ', '.join(f'{genes[j]} {rg[j]:+.2f}' for j in np.argsort(rg)[:8]))
    P('  genes up with real proximity (same cells):', ', '.join(f'{genes[j]} {rd[j]:+.3f}' for j in np.argsort(-rd)[:8]),
      f'| r(gene~s-axis, gene~proximity) over genes {np.corrcoef(rg, rd)[0, 1]:+.2f}')
    pd.DataFrame({'gene': genes, 'rho_s_axis': rg, 'rho_proximity': rd}).to_csv(OUT / f'genes_{T.replace("/", "_").replace(" ", "_")}_{L}.csv', index=False)
    # 6 decoded shift
    sh = np.asarray(SH[ii]); rs = np.nan_to_num([spearmanr(sh[:, j], axes['s']).statistic for j in range(len(types))])
    P('  decoded salient shift along the s axis, neighbours gained:', ', '.join(f'{types[j]} {rs[j]:+.2f}' for j in np.argsort(-rs)[:4]),
      '| lost:', ', '.join(f'{types[j]} {rs[j]:+.2f}' for j in np.argsort(rs)[:3]))
    summary.append(rec)
S = pd.DataFrame(summary); S.to_csv(OUT / 'cases_summary.csv', index=False)
pd.set_option('display.width', 250); P('\n' + S.round(3).to_string(index=False)); P('done')
