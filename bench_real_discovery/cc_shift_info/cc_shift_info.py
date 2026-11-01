#!/usr/bin/env python3
"""What do Haruka's pseudo-proportion shift and its z-CC / s-CC embeddings carry?

usage: cc_shift_info.py starmap | codex

A. pseudo-proportion shift  dp = p([z,s]) - p([z,0])   (condition cells; 0 for reference cells)
   A1 mean dp per neighbour type vs the observed change in neighbourhood composition
      (condition - reference) and vs the residual the reference head leaves (obs - p([z,0]))
   A2 per cell: corr(dp, obs - p([z,0])) per column; fit of obs by p([z,0]) vs p([z,s])
   A3 plaque gradient per column, computed within anatomy: dp vs observed composition
   A4 dimensionality of dp (PCA)
   A5 neighbour-mean shift: size per neighbour-type block, plaque-adjacent vs far
B. linear probes on z-CC, s-CC, z-CC+s-CC (condition cells; held out by slice / donor):
   own cell type, anatomy, observed composition, composition residual, dp, plaque, batch
Observed composition = k=20 spatial neighbours within sample (the model's own target).
"""
import os, sys, json, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
H = '/home/yancui/haruka_v2'; DS = sys.argv[1]; OUT = f'{H}/bench_real_discovery/cc_shift_info'
K = 20

if DS == 'starmap':
    R = f'{H}/results_real_compgated_lambda10_samplebatch_sepheads/ad/'
    MD = R + 'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
    CC = R + 'haruka_z_s_cellcharter_cc3_condition_only'
    a = ad.read_h5ad(f'{H}/bench_real_discovery/data/ad.h5ad', backed='r'); o = a.obs.copy(); xy = np.asarray(a.obsm['spatial'])
    ct = o.cell_type.astype(str).to_numpy(); samp = o.slice_id.astype(str).to_numpy()
    anat = o.tissue_domain.astype(str).to_numpy(); dis = (o.condition.astype(str) == 'disease').to_numpy()
    Dt = pd.read_csv(f'{H}/final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
    dist = pd.Series(a.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float)
    near_thr = 2 * 31.4; unit = 'cell spacings'; dist_unit = 31.4
else:
    V = 'all7_no_oligo'; R = f'{H}/results_codex_wholeslice/codex_wholeslice_{V}/'
    MD = R + 'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
    CC = R + 'haruka_z_s_cellcharter_cc3_condition_only'
    a = ad.read_h5ad(f'{H}/bench_codex_cns_2026/wholeslice/codex_wholeslice_{V}.h5ad', backed='r'); o = a.obs.copy()
    xy = np.asarray(a.obsm['spatial_um'])
    ct = o.cell_type.astype(str).to_numpy(); samp = o.donor_id.astype(str).to_numpy()
    anat = o.region.astype(str).to_numpy(); dis = (o.condition.astype(str) == 'disease').to_numpy()
    dist = np.load(f'{H}/bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy').astype(float)
    near_thr = 30.0; unit = 'um'; dist_unit = 10.0
meta = json.load(open(MD + '/meta.json')); cats = meta['cell_type_categories']
names = np.load(MD + '/obs_names.npy', allow_pickle=True).astype(str)
assert (names == np.asarray(a.obs_names).astype(str)).all(), 'fit rows not aligned with h5ad'
log = open(f'{OUT}/cc_shift_info_{DS}.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

# observed neighbourhood composition (the model's composition target)
code = pd.Categorical(ct, categories=cats).codes
obs = np.zeros((len(ct), len(cats)), np.float32)
for sm in np.unique(samp):
    i = np.where(samp == sm)[0]
    _, nb = cKDTree(xy[i]).query(xy[i], k=K + 1)
    for c in range(1, K + 1):
        np.add.at(obs, (i, code[i[nb[:, c]]]), 1.0)
obs /= K
pbg = np.load(MD + '/background_composition.npy'); pc = np.load(MD + '/condition_composition.npy')
dp = np.load(MD + '/salient_composition_shift.npy'); dnm = np.load(MD + '/salient_neighbor_mean_shift.npy')
P(f'== {DS}: {len(ct)} cells, {dis.sum()} condition; types {cats}')
P('   check dp == p([z,s]) - p([z,0]) on condition cells:', np.abs(dp[dis] - (pc - pbg)[dis]).max().round(6),
  '| dp on reference cells max |.|', np.abs(dp[~dis]).max())

# ---------------- A1 ----------------
D = dis; resid = obs - pbg
t = pd.DataFrame({'obs_ref': obs[~D].mean(0), 'obs_cond': obs[D].mean(0), 'obs_change': obs[D].mean(0) - obs[~D].mean(0),
                  'p_bg_cond': pbg[D].mean(0), 'p_full_cond': pc[D].mean(0), 'resid_cond': resid[D].mean(0),
                  'dp_mean': dp[D].mean(0), 'dp_sd': dp[D].std(0)}, index=cats)
P('\n-- A1 mean over cells (fractions of the 20-cell neighbourhood) --'); P(t.round(4).to_string())
P('   corr across types: dp_mean vs obs_change r=%.2f | dp_mean vs resid_cond r=%.2f' %
  (pearsonr(t.dp_mean, t.obs_change)[0], pearsonr(t.dp_mean, t.resid_cond)[0]))

# ---------------- A2 ----------------
def r2(y, f): return 1 - ((y - f) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
P('\n-- A2 per cell (condition cells; in-sample: the heads were trained on these cells) --')
P('   R2 of observed composition:  p([z,0]) %.3f   p([z,s]) %.3f' % (r2(obs[D], pbg[D]), r2(obs[D], pc[D])))
rows = []
for j, c in enumerate(cats):
    if dp[D, j].std() < 1e-6 or resid[D, j].std() < 1e-6: continue
    rows.append(dict(type=c, r_dp_resid=pearsonr(dp[D, j], resid[D, j])[0]))
A2 = pd.DataFrame(rows).set_index('type'); P('   corr(dp, obs - p([z,0])) per column:'); P(A2.round(3).T.to_string())

# ---------------- A3 ----------------
P(f'\n-- A3 plaque gradient per column: Spearman with log distance, within anatomy, condition cells (negative = higher near plaque) --')
ld = np.log1p(dist / dist_unit); ok = D & np.isfinite(ld)
def within(v):
    num, den = 0., 0
    for g in np.unique(anat[ok]):
        m = ok & (anat == g)
        if m.sum() < 200 or np.std(v[m]) < 1e-9: continue
        num += spearmanr(v[m], ld[m])[0] * m.sum(); den += m.sum()
    return num / den if den else np.nan
near = ok & (dist <= near_thr); far = ok & (dist > 4 * near_thr)
rows = []
for j, c in enumerate(cats):
    rows.append(dict(type=c, rho_dp=within(dp[:, j]), rho_obs=within(obs[:, j]), rho_resid=within(resid[:, j]), rho_pbg=within(pbg[:, j]),
                     dp_near=dp[near, j].mean(), dp_far=dp[far, j].mean(), obs_near=obs[near, j].mean(), obs_far=obs[far, j].mean()))
A3 = pd.DataFrame(rows).set_index('type'); P(A3.round(4).to_string())
P('   corr across types of the gradients: rho_dp vs rho_obs r=%.2f | rho_dp vs rho_resid r=%.2f' %
  (pearsonr(A3.rho_dp, A3.rho_obs)[0], pearsonr(A3.rho_dp, A3.rho_resid)[0]))
P(f'   near = <= {near_thr} {unit} ({near.sum()} cells), far = > {4 * near_thr} {unit} ({far.sum()} cells)')
# what part of the observed gradient is explained by z's head (p_bg) vs left for s (resid)
A3.to_csv(f'{OUT}/A3_gradient_{DS}.csv')

# ---------------- A4 ----------------
pca = PCA(min(6, len(cats))).fit(dp[D]); ev = pca.explained_variance_ratio_
P('\n-- A4 PCA of dp (condition cells) -- variance explained', np.round(ev, 3))
for k in range(2):
    ld_ = pd.Series(pca.components_[k], index=cats).sort_values()
    sc = pca.transform(dp[D])[:, k]; rho = spearmanr(sc[np.isfinite(ld[D])], ld[D][np.isfinite(ld[D])])[0]
    P(f'   PC{k + 1}: rho with log distance {rho:+.2f} | loadings', ', '.join(f'{i} {v:+.2f}' for i, v in ld_.items()))

# ---------------- A5 ----------------
E = meta['neighbor_embedding_dim']; blk = np.linalg.norm(dnm.reshape(len(dnm), len(cats), E), axis=2)
P('\n-- A5 neighbour-mean shift ||D_nm([z,s]) - D_nm([z,0])|| per neighbour-type block, condition cells --')
P(pd.DataFrame({'near': blk[near].mean(0), 'far': blk[far].mean(0), 'ratio': blk[near].mean(0) / (blk[far].mean(0) + 1e-9),
                'rho_logdist': [within(blk[:, j]) for j in range(len(cats))]}, index=cats).round(3).to_string())

# ---------------- B ----------------
src = np.load(CC + '/source_rows.npy'); zcc = np.load(CC + '/spatial.npy'); scc = np.load(CC + '/salient_spatial.npy')
assert dis[src].all()
g = samp[src]; ug = np.unique(g)
P(f'\n-- B linear probes, condition cells n={len(src)}, held out by {"slice" if DS == "starmap" else "donor"} ({len(ug)} groups) --')
X = {'z-CC': zcc, 's-CC': scc, 'z-CC+s-CC': np.hstack([zcc, scc])}
P('   redundancy: R2 of s-CC from z-CC %.3f, z-CC from s-CC %.3f (5-fold)' % tuple(
    np.mean([r2(B[te], Ridge(1.0).fit(A[tr], B[tr]).predict(A[te]))
             for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, g)]) for A, B in [(zcc, scc), (scc, zcc)]))

def heldout(Xf, y, kind, grp=g, strata=None):
    """kind: cls -> balanced acc; reg -> R2 (pooled); auc -> AUROC (within strata if given)"""
    pred = np.zeros(len(y)) if kind != 'cls' else np.empty(len(y), dtype=object)
    if kind == 'reg': pred = np.zeros_like(y, dtype=float)
    for h in np.unique(grp):
        tr, te = grp != h, grp == h
        sc = StandardScaler().fit(Xf[tr]); A, B = sc.transform(Xf[tr]), sc.transform(Xf[te])
        if kind == 'reg':
            ytr = y[tr]; mu = ytr.mean(0); pred[te] = Ridge(10.0).fit(A, ytr - mu).predict(B) + mu
        elif kind == 'cls':
            pred[te] = LogisticRegression(max_iter=300, class_weight='balanced').fit(A, y[tr]).predict(B)
        else:
            pred[te] = LogisticRegression(max_iter=300, class_weight='balanced').fit(A, y[tr]).decision_function(B)
    if kind == 'reg': return r2(y, pred)
    if kind == 'cls': return balanced_accuracy_score(y.astype(str), pred.astype(str))
    if strata is None: return roc_auc_score(y, pred)
    num = den = 0
    for s_ in np.unique(strata):                      # per held-out group x anatomy, n-weighted
        for h in np.unique(grp):
            m = (strata == s_) & (grp == h)
            if y[m].sum() >= 20 and (~y[m]).sum() >= 20:
                num += roc_auc_score(y[m], pred[m]) * m.sum(); den += m.sum()
    return num / den

yy = {'own cell type (bal. acc)': (ct[src], 'cls', None),
      'anatomy (bal. acc)': (anat[src], 'cls', None),
      'observed composition (R2)': (obs[src], 'reg', None),
      'composition residual obs-p([z,0]) (R2)': (resid[src], 'reg', None),
      'pseudo-proportion shift dp (R2)': (dp[src], 'reg', None),
      'plaque-adjacent (AUROC)': (np.nan_to_num(dist[src], nan=1e9) <= near_thr, 'auc', None),
      'plaque-adjacent within anatomy (AUROC)': (np.nan_to_num(dist[src], nan=1e9) <= near_thr, 'auc', anat[src])}
res = []
for name, (y, kind, strata) in yy.items():
    vals = {k: heldout(Xf, y, kind, strata=strata) for k, Xf in X.items()}
    res.append(dict(target=name, **vals)); P(f'   {name:42s}', '  '.join(f'{k} {v:.3f}' for k, v in vals.items()))
# batch: within-condition sample identity, 5-fold CV (cannot hold out the label itself)
for k, Xf in X.items():
    pr = np.empty(len(g), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, g):
        sc = StandardScaler().fit(Xf[tr]); pr[te] = LogisticRegression(max_iter=300, class_weight='balanced').fit(sc.transform(Xf[tr]), g[tr]).predict(sc.transform(Xf[te]))
    res.append(dict(target=f'batch: {"slice" if DS == "starmap" else "donor"} (bal. acc, 5-fold; chance {1 / len(ug):.2f})', **{k: balanced_accuracy_score(g, pr)}))
B = pd.DataFrame(res).groupby('target', sort=False).first()
P('   ' + B.iloc[-1:].round(3).to_string().replace('\n', '\n   '))
B.to_csv(f'{OUT}/B_probes_{DS}.csv')
P('\n   per-column R2 of dp from each embedding:')
for k, Xf in X.items():
    cols = [heldout(Xf, dp[src][:, j], 'reg') if dp[src][:, j].std() > 1e-6 else np.nan for j in range(len(cats))]
    P(f'   {k:10s}', ', '.join(f'{c} {v:+.2f}' for c, v in zip(cats, cols)))
P('done')
