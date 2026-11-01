#!/usr/bin/env python3
"""Two layers inside the fibrosis (condition) arm: where does z separate Less vs More Affected, and where does s
split a microenvironment that z cannot?  Condition cells only (control s is the unused encoder output).
Fit: Unaffected vs all fibrosis, batch = sample (core).  35 cores (More 20 / Less 15), 26 donors.

PART 1  per-cell, per microenvironment stratum (cell type x published TNiche): logistic (C=0.1, balanced) on z or s,
        GroupKFold(5) over donors -> per-cell AUROC and core-level AUROC (mean held-out score per core).
        Stratum kept if >= 2000 cells, >= 8 donors, both stages.
PART 2  nested domains.  Background microenvironments = k-means (k=12) on z.  Inside each z-domain, split the cells
        4 ways by k-means on s (s-subdomains) and, as the matched control, 4 ways by k-means on z (z-subdomains).
        Per core: CLR of the 4 subdomain shares within that domain -> leave-one-donor-out logistic (C=0.1)
        AUROC Less vs More, and Spearman(held-out score, % pathology).  Also the domain's own share per core
        (does the background microenvironment's abundance already separate stage?).
        Mirror: s-domains (k=12 on s) split by z vs by s.
        Every s-subdomain inside a z-domain is described by its cell types, H&E regions (labelled cells), top
        within-type genes vs the rest of the domain, and its mean decoded salient composition shift.
"""
import os, warnings, json
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_lung_fibrosis_gse250346/zs_stage/layers'; OUT.mkdir(exist_ok=True)
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r'); o = a.obs
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
log = open(OUT / 'zs_layers.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
stage = o.sample_affect.astype(str).to_numpy(); F = np.flatnonzero(stage != 'Unaffected')
ct = o.cell_type.astype(str).to_numpy()[F]; core = o['sample'].astype(str).to_numpy()[F]; don = o.donor.astype(str).to_numpy()[F]
tn = o.TNiche.astype(str).to_numpy()[F]; yb = stage[F] == 'More Affected'
Z = StandardScaler().fit_transform(np.load(MD / 'background.npy', mmap_mode='r')[F]).astype(np.float32)
S = StandardScaler().fit_transform(np.load(MD / 'salient.npy', mmap_mode='r')[F]).astype(np.float32)
C = pd.DataFrame({'core': core, 'donor': don, 'y': yb}).groupby('core').first()
C['path'] = pd.to_numeric(o.percent_pathology, errors='coerce').groupby(o['sample'].astype(str)).first().reindex(C.index)
P(f'fibrosis cells {len(F)}, cores {len(C)} (More {C.y.sum()}, Less {(~C.y).sum()}), donors {C.donor.nunique()}')
rng = np.random.default_rng(0)

# ---------------- PART 1
def cv_cell(X, y, g):
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2: continue
        sc[te] = LogisticRegression(C=0.1, max_iter=500, class_weight='balanced').fit(X[tr], y[tr]).decision_function(X[te])
    return sc
rows = []
for (c, t), idx in pd.Series(np.arange(len(F))).groupby([ct, tn]).indices.items():
    if len(idx) < 2000: continue
    if len(idx) > 30000: idx = np.sort(rng.choice(idx, 30000, replace=False))
    y = yb[idx]; g = don[idx]
    if y.mean() in (0, 1) or len(np.unique(g)) < 8 or min(y.sum(), (~y).sum()) < 200: continue
    rec = dict(cell_type=c, TNiche=t, n=len(idx), donors=len(np.unique(g)))
    for k, M in [('z', Z), ('s', S)]:
        sc = cv_cell(M[idx], y, g); rec[f'cell_{k}'] = roc_auc_score(y, sc)
        d = pd.DataFrame({'s': sc, 'y': y, 'c': core[idx]}).groupby('c').agg(s=('s', 'mean'), y=('y', 'first'))
        rec[f'core_{k}'] = roc_auc_score(d.y, d.s) if d.y.nunique() == 2 else np.nan
    rows.append(rec)
T1 = pd.DataFrame(rows); T1['d_cell'] = T1.cell_s - T1.cell_z; T1['d_core'] = T1.core_s - T1.core_z
TNn = pd.crosstab(o.TNiche.astype(str), o.cell_type.astype(str), normalize='index')
T1['TNiche_top'] = T1.TNiche.map(lambda n: ' / '.join(f'{k} {v:.0%}' for k, v in TNn.loc[n].nlargest(2).items()))
T1.to_csv(OUT / 'part1_celltype_x_niche.csv', index=False)
P(f'\n== PART 1: cell type x TNiche strata ({len(T1)}), Less vs More, donor-held-out ==')
for m in ['cell', 'core']:
    P(f'  {m}: median z {T1[f"{m}_z"].median():.3f}  s {T1[f"{m}_s"].median():.3f} | s > z in {(T1[f"d_{m}"] > 0).sum()}/{len(T1)} | Wilcoxon p {wilcoxon(T1[f"{m}_s"], T1[f"{m}_z"]).pvalue:.2g}')
lay = np.where((T1.core_s >= 0.7) & (T1.d_core >= 0.1), 's-layer', np.where((T1.core_z >= 0.7) & (T1.d_core <= -0.1), 'z-layer', 'shared/neither'))
T1['layer'] = lay; P('  core-level layers (AUROC >= 0.7 and margin >= 0.1):', pd.Series(lay).value_counts().to_dict())
cols = ['cell_type', 'TNiche', 'TNiche_top', 'n', 'donors', 'cell_z', 'cell_s', 'core_z', 'core_s']
P('  s-layer strata:'); P(T1[T1.layer == 's-layer'].sort_values('d_core', ascending=False)[cols].round(3).to_string(index=False))
P('  z-layer strata (top 15):'); P(T1[T1.layer == 'z-layer'].sort_values('d_core')[cols].head(15).round(3).to_string(index=False))
for f in ['cell_type', 'TNiche']:
    g = T1.groupby(f).agg(n=('d_core', 'size'), d_core=('d_core', 'median'), d_cell=('d_cell', 'median')).query('n >= 3').sort_values('d_core')
    P(f'  median s-z by {f} (>= 3 strata):'); P(g.round(3).to_string())

# ---------------- PART 2
sub = np.sort(rng.choice(len(F), 300000, replace=False))
def km(X, k, seed=0, fit_idx=None):
    fi = fit_idx if fit_idx is not None else np.arange(len(X))
    if len(fi) > 100000: fi = np.sort(np.random.default_rng(seed).choice(fi, 100000, replace=False))
    m = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=8192).fit(X[fi]); return m.predict(X)
def lodo(Xc, y, dn):
    sc = np.zeros(len(y))
    for d in np.unique(dn):
        te = dn == d
        if len(np.unique(y[~te])) < 2: continue
        ss = StandardScaler().fit(Xc[~te]); sc[te] = LogisticRegression(C=0.1, max_iter=2000).fit(ss.transform(Xc[~te]), y[~te]).decision_function(ss.transform(Xc[te]))
    return sc
def clr(Pm):
    x = np.log(Pm + 1e-3); return x - x.mean(1, keepdims=True)
genes = np.asarray(a.var_names).astype(str)
Ssh = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r')
he = pd.read_csv('/project/zhihuanglab/yancui/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type'])
he = he[~he.obs.duplicated(keep=False)].set_index('obs').annotation_type
heF = he.reindex(o.index[F]).to_numpy()
desc_rows = []
for outer, Mo in [('z', Z), ('s', S)]:
    dom = km(Mo, 12, fit_idx=sub)
    rows = []
    for D in range(12):
        idx = np.flatnonzero(dom == D)
        share = pd.Series(idx).groupby(core[idx]).size().reindex(C.index).fillna(0) / pd.Series(core).value_counts().reindex(C.index)
        sc0 = lodo(share.to_numpy()[:, None], C.y.to_numpy(), C.donor.to_numpy())
        rec = dict(outer=outer, domain=D, n=len(idx), top_cells=' / '.join(f'{k} {v:.0%}' for k, v in pd.Series(ct[idx]).value_counts(normalize=True).head(3).items()),
                   share_auc=roc_auc_score(C.y, sc0), share_rho=spearmanr(share, C.path).statistic)
        labs = {}
        for inner, Mi in [('z', Z), ('s', S)]:
            lab = km(Mi[idx], 4, seed=1); labs[inner] = lab
            Pm = pd.crosstab(core[idx], lab, normalize='index').reindex(C.index).fillna(0.25).to_numpy()
            sc = lodo(clr(Pm), C.y.to_numpy(), C.donor.to_numpy())
            rec[f'split_{inner}_auc'] = roc_auc_score(C.y, sc); rec[f'split_{inner}_rho'] = spearmanr(sc, C.path).statistic
        rows.append(rec)
        if outer == 'z':   # describe the s-subdomains of every z-domain
            lab = labs['s']; Xr = None
            gi = F[idx]; pick = np.sort(rng.choice(len(idx), min(len(idx), 20000), replace=False))
            Xr = a[gi[pick]].layers['counts']; Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
            Xn = np.log1p(Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e4)
            ctp = ct[idx][pick]
            for c_ in np.unique(ctp): k_ = ctp == c_; Xn[k_] -= Xn[k_].mean(0)
            shp = np.asarray(Ssh[gi[pick]])
            for j in range(4):
                mj = lab == j; pj = lab[pick] == j
                Pm = pd.crosstab(core[idx], lab, normalize='index').reindex(C.index).fillna(0)[j]
                hj = pd.Series(heF[idx][mj]).dropna().value_counts(normalize=True); hall = pd.Series(heF[idx]).dropna().value_counts(normalize=True)
                enr = (hj / hall.reindex(hj.index)).loc[hj.index[hj > 0.02]].sort_values(ascending=False).head(3) if len(hj) else pd.Series(dtype=float)
                dg = Xn[pj].mean(0) - Xn[~pj].mean(0) if pj.sum() > 30 else np.zeros(len(genes))
                dsh = shp[pj].mean(0) - shp[~pj].mean(0) if pj.sum() > 30 else np.zeros(len(types))
                desc_rows.append(dict(z_domain=D, s_sub=j, frac_of_domain=mj.mean(), rho_share_path=spearmanr(Pm, C.path).statistic,
                                      more_minus_less_share=Pm[C.y].mean() - Pm[~C.y].mean(),
                                      cells=' / '.join(f'{k} {v:.0%}' for k, v in pd.Series(ct[idx][mj]).value_counts(normalize=True).head(3).items()),
                                      he_enriched=' / '.join(f'{k} x{v:.1f}' for k, v in enr.items()),
                                      genes_up=','.join(genes[np.argsort(-dg)[:8]]), genes_down=','.join(genes[np.argsort(dg)[:6]]),
                                      shift_up=','.join(f'{types[i]} {dsh[i]:+.3f}' for i in np.argsort(-dsh)[:3]),
                                      shift_down=','.join(f'{types[i]} {dsh[i]:+.3f}' for i in np.argsort(dsh)[:2])))
    T2 = pd.DataFrame(rows); T2.to_csv(OUT / f'part2_{outer}domains.csv', index=False)
    P(f'\n== PART 2: {outer}-domains (k=12), each split 4 ways by z or by s; core-level LODO AUROC Less vs More ==')
    P(T2[['domain', 'n', 'top_cells', 'share_auc', 'share_rho', 'split_z_auc', 'split_s_auc', 'split_z_rho', 'split_s_rho']].round(3).to_string(index=False))
    P(f'  split by s better than split by z in {(T2.split_s_auc > T2.split_z_auc).sum()}/12 domains (median AUROC z {T2.split_z_auc.median():.3f}, s {T2.split_s_auc.median():.3f};'
      f' Wilcoxon p {wilcoxon(T2.split_s_auc, T2.split_z_auc).pvalue:.2g}); |rho| with pathology median z {T2.split_z_rho.abs().median():.2f} s {T2.split_s_rho.abs().median():.2f}')
DS = pd.DataFrame(desc_rows); DS.to_csv(OUT / 'part2_s_subdomains_described.csv', index=False)
T2z = pd.read_csv(OUT / 'part2_zdomains.csv')
win = T2z[(T2z.split_s_auc - T2z.split_z_auc >= 0.1) & (T2z.split_s_auc >= 0.7)].domain.tolist()
P(f'\n== z-domains where s splits stage and z does not (s AUROC >= 0.7, margin >= 0.1): {win} ==')
for D in win:
    r = T2z[T2z.domain == D].iloc[0]
    P(f'\n z-domain {D} ({r.top_cells}; n {r.n}) | domain share AUROC {r.share_auc:.2f} | split by z {r.split_z_auc:.2f}  by s {r.split_s_auc:.2f}')
    P(DS[DS.z_domain == D].drop(columns='z_domain').round(3).to_string(index=False))
P('done')
