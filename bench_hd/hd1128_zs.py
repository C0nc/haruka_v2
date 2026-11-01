#!/usr/bin/env python3
"""HD MM-1128 Haruka fit (argv[1] = fit folder, default results_1128_sym): do z or s carry HD-specific cell states?  HD (condition) cells only.
GT (evaluation only, never model input): CAG (smFISH CAG-repeat signal, somatic expansion proxy), Agg (1 = intranuclear
huntingtin inclusion, 0 = cell nearest an extranuclear inclusion, NaN = none), m_gamma (gammaH2AX), lipo (lipofuscin).
Latents standardised, then centred per donor x cell type: removes germline CAG / donor / cell identity, so each test asks
about a cell's state relative to the same cell type in the same patient.
0  collapse check: ||s|| by condition, per-dim SD.
1  variance of HD-cell z / s explained by cell type, subtype, donor, batch.
2  continuous GT (CAG, m_gamma, lipo): per cell type, ridge on centred z or s, leave-one-HD-donor-out; Spearman of the
   held-out prediction with the donor x type centred GT (median over donors).
3  inclusions: per cell type, intranuclear inclusion (Agg 1) vs no inclusion (NaN), and nearest-to-extranuclear-inclusion
   (Agg 0) vs no inclusion; logistic, leave-one-HD-donor-out, AUROC within donor (median).
4  spatial salient: decoded composition shift vs observed k=20 neighbour shares, Agg-1 / Agg-0 cells minus same-type
   no-inclusion cells of the same donor: alignment with D([z,0]), D([z,s]) and the shift alone.
5  genes: within donor x type, share of each gene's variance (log1p CP10k) linearly explained by z vs s; top s-specific.
"""
import os, sys, glob, json, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')
RES = sys.argv[1] if len(sys.argv) > 1 else 'results_1128_sym'   # fit folder under haruka_hd
D = Path('/project/zhihuanglab/yancui/haruka_hd'); MD = Path(glob.glob(str(D / RES / '*/*/'))[0])
OUT = Path('/home/yancui/haruka_v2/bench_hd') / ('hd1128_zs' if RES == 'results_1128_sym' else f'hd1128_zs_{RES.replace("results_1128_", "")}'); OUT.mkdir(exist_ok=True)
a = ad.read_h5ad(D / 'hd1128_haruka.h5ad', backed='r'); o = a.obs
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
log = open(OUT / 'hd1128_zs.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
pd.set_option('display.width', 250)
cond = o.condition.astype(str).to_numpy(); Sall = np.load(MD / 'salient.npy', mmap_mode='r'); Zall = np.load(MD / 'background.npy', mmap_mode='r')
P('== 0. collapse check ==')
for g in ['control', 'disease']:
    i = np.flatnonzero(cond == g); x = np.asarray(Sall[i]); P(f'  {g}: mean ||s|| {np.linalg.norm(x, axis=1).mean():.3f}, median per-dim SD {np.median(x.std(0)):.3f}')
P(f'  z median per-dim SD (disease) {np.median(np.asarray(Zall[cond == "disease"]).std(0)):.3f}')
H = np.flatnonzero(cond == 'disease'); oh = o.iloc[H]
ct = oh.cell_type.astype(str).to_numpy(); dn = oh.slice_id.astype(str).to_numpy(); grp = pd.Series(dn).str.cat(ct, sep='|').to_numpy()
def centre(M, g):
    M = np.asarray(M, np.float64).copy()
    for k in pd.Series(np.arange(len(g))).groupby(g).indices.values(): M[k] -= M[k].mean(0)
    return M
Zs = StandardScaler().fit_transform(np.asarray(Zall[H])); Ss = StandardScaler().fit_transform(np.asarray(Sall[H]))
L = {'z': centre(Zs, grp), 's': centre(Ss, grp)}

# 1
P('\n== 1. share of HD-cell latent variance explained by each annotation alone ==')
rows = {}
for f in ['cell_type', 'subtype', 'slice_id', 'batch']:
    Dm = pd.get_dummies(oh[f].astype(str)).to_numpy(float)
    rows[f] = {k: 1 - ((X - Dm @ np.linalg.lstsq(Dm, X, rcond=None)[0]) ** 2).sum() / (X ** 2).sum() for k, X in [('z', Zs), ('s', Ss)]}
VP = pd.DataFrame(rows).T; VP.to_csv(OUT / 'variance_partition.csv'); P(VP.round(3).to_string())
np.save(OUT / 's_scale.npy', np.array([np.linalg.norm(np.asarray(Sall[cond == g]), axis=1).mean() for g in ['control', 'disease']] + [np.median(np.asarray(Sall[cond == 'disease']).std(0)), np.median(np.asarray(Zall[cond == 'disease']).std(0))]))

# 2
P('\n== 2. continuous HD state, per cell type, leave-one-HD-donor-out, Spearman (median over held-out donors) ==')
res = []
for gt in ['CAG', 'm_gamma', 'lipo']:
    y_all = pd.to_numeric(oh[gt], errors='coerce').to_numpy()
    for c in np.unique(ct):
        m = (ct == c) & np.isfinite(y_all)
        if m.sum() < 500 or len(np.unique(dn[m])) < 4: continue
        y = centre(y_all[m][:, None], grp[m])[:, 0]; d_ = dn[m]; rec = dict(gt=gt, cell_type=c, n=int(m.sum()))
        for k, X in L.items():
            Xm = X[m]; r = []
            for d in np.unique(d_):
                te = d_ == d
                if te.sum() < 50: continue
                pr = Ridge(alpha=10).fit(Xm[~te], y[~te]).predict(Xm[te]); r.append(spearmanr(pr, y[te]).statistic)
            rec[k] = np.nanmedian(r)
        res.append(rec)
R2 = pd.DataFrame(res); R2.to_csv(OUT / 'continuous_gt.csv', index=False)
for gt, t in R2.groupby('gt', sort=False):
    P(f'\n {gt}: {len(t)} cell types | median z {t.z.median():+.3f}  s {t.s.median():+.3f} | s > z in {(t.s > t.z).sum()}/{len(t)}'
      + (f' | Wilcoxon p {wilcoxon(t.s, t.z).pvalue:.2g}' if len(t) >= 5 else ''))
    P(t.sort_values('n', ascending=False)[['cell_type', 'n', 'z', 's']].round(3).to_string(index=False))

# 3
P('\n== 3. huntingtin inclusions, per cell type, leave-one-HD-donor-out, within-donor AUROC (median) ==')
agg = pd.to_numeric(oh.Agg, errors='coerce').to_numpy(); res = []
for lab, pos in [('intranuclear (Agg 1) vs none', agg == 1), ('nearest extranuclear (Agg 0) vs none', agg == 0)]:
    for c in np.unique(ct):
        m = (ct == c) & (pos | np.isnan(agg)); y = pos[m]; d_ = dn[m]
        if y.sum() < 50: continue
        rec = dict(test=lab, cell_type=c, n_pos=int(y.sum()), n=int(m.sum()))
        for k, X in L.items():
            Xm = X[m]; au = []
            for d in np.unique(d_):
                te = d_ == d
                if y[te].sum() < 10 or y[~te].sum() < 10: continue
                sc = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced').fit(Xm[~te], y[~te]).decision_function(Xm[te])
                au.append(roc_auc_score(y[te], sc))
            rec[k] = np.nanmedian(au) if au else np.nan; rec['donors'] = len(au)
        res.append(rec)
R3 = pd.DataFrame(res).dropna(subset=['z', 's']); R3.to_csv(OUT / 'inclusions.csv', index=False)
for lab, t in R3.groupby('test', sort=False):
    P(f'\n {lab}: {len(t)} cell types | median z {t.z.median():.3f}  s {t.s.median():.3f} | s > z in {(t.s > t.z).sum()}/{len(t)}')
    P(t[['cell_type', 'n_pos', 'n', 'donors', 'z', 's']].round(3).to_string(index=False))

# 4 spatial salient around inclusions
P('\n== 4. decoded niche vs observed niche: inclusion-associated cells minus same-type no-inclusion cells, same donor ==')
xy = np.asarray(a.obsm['spatial'])[H]; cti = pd.Categorical(ct, categories=types).codes; OBS = np.zeros((len(H), len(types)), np.float32)
for d in np.unique(dn):
    i = np.flatnonzero(dn == d); _, nb = NearestNeighbors(n_neighbors=21).fit(xy[i]).kneighbors(xy[i])
    OBS[i] = np.stack([np.bincount(r, minlength=len(types)) for r in cti[i][nb[:, 1:]]]) / 20.0
BG = np.asarray(np.load(MD / 'background_composition.npy', mmap_mode='r')[H]); SH = np.asarray(np.load(MD / 'salient_composition_shift.npy', mmap_mode='r')[H])
M4 = {'obs': OBS, 'bg': BG, 'cond': BG + SH, 'shift': SH}; rows = []
for lab, pos in [('Agg 1', agg == 1), ('Agg 0', agg == 0)]:
    for c in np.unique(ct):
        diffs = {k: [] for k in M4}
        for d in np.unique(dn):
            p_ = (ct == c) & (dn == d) & pos; n_ = (ct == c) & (dn == d) & np.isnan(agg)
            if p_.sum() < 10 or n_.sum() < 10: continue
            for k, v in M4.items(): diffs[k].append(v[p_].mean(0) - v[n_].mean(0))
        if not diffs['obs']: continue
        dm = {k: np.mean(v, 0) for k, v in diffs.items()}
        top = np.argsort(-np.abs(dm['obs']))[:2]
        rows.append(dict(test=lab, cell_type=c, donors=len(diffs['obs']), align_bg=np.corrcoef(dm['obs'], dm['bg'])[0, 1],
                         align_cond=np.corrcoef(dm['obs'], dm['cond'])[0, 1], align_shift=np.corrcoef(dm['obs'], dm['shift'])[0, 1],
                         obs_L1=np.abs(dm['obs']).sum(), shift_L1=np.abs(dm['shift']).sum(),
                         top_obs='; '.join(f'{types[j]} {dm["obs"][j]:+.3f} (shift {dm["shift"][j]:+.3f})' for j in top)))
R4 = pd.DataFrame(rows); R4.to_csv(OUT / 'inclusion_niche.csv', index=False)
if len(R4): P(R4.round(3).to_string(index=False))

# 5 genes
P('\n== 5. within donor x cell type gene variance explained (HD cells, 60k subsample) ==')
sub = np.sort(np.random.default_rng(0).choice(len(H), min(60000, len(H)), replace=False))
Xr = a[H[sub]].layers['counts']; Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
Xn = centre(np.log1p(Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e4), grp[sub]); genes = np.asarray(a.var_names).astype(str)
def r2(M):
    B, *_ = np.linalg.lstsq(M, Xn, rcond=None); return 1 - ((Xn - M @ B) ** 2).sum(0) / np.maximum((Xn ** 2).sum(0), 1e-12)
G = pd.DataFrame({'gene': genes, 'R2_z': r2(L['z'][sub]), 'R2_s': r2(L['s'][sub])}); G['s_minus_z'] = G.R2_s - G.R2_z
G.to_csv(OUT / 'gene_r2.csv', index=False)
P(f'  median over genes: z {G.R2_z.median():.3f}  s {G.R2_s.median():.3f}; s > z for {(G.s_minus_z > 0).mean():.0%} of genes')
P('  top s-specific:', ', '.join(f'{r.gene} {r.R2_s:.2f}/{r.R2_z:.2f}' for r in G.nlargest(20, 's_minus_z').itertuples()))
P('  top z-specific:', ', '.join(f'{r.gene} {r.R2_z:.2f}/{r.R2_s:.2f}' for r in G.nsmallest(12, 's_minus_z').itertuples()))
P('done')
