#!/usr/bin/env python3
"""Sanity check of Haruka's counterfactual cell proportions for condition cells.

usage: check_cf_proportion.py starmap | codex

For every condition cell the composition head gives two neighbourhood proportions:
  background  p_bg   = D_comp([z,0])   "what this niche would look like without the condition"
  salient     p_full = D_comp([z,s])   the condition niche;  dp = p_full - p_bg
C1 bookkeeping: simplex, dp == p_full - p_bg, dp == 0 on reference cells
C2 fit to the observed k=20 neighbourhood (in-sample: the head was trained on these cells)
C3 is the background reference-like?  matched on anatomy x own cell type, compare
   p_bg(condition) and p_full(condition) with the observed reference and condition niches
C4 plaque: per column, Spearman with log distance within anatomy (observed / p_bg / p_full),
   and the microglia column by distance bin
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import os, sys, json, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
H = f'{_HR}'; DS = sys.argv[1]; OUT = f'{H}/bench_real_discovery/cc_shift_info'; K = 20

if DS == 'starmap':
    MD = f'{H}/results_real_compgated_lambda10_samplebatch_sepheads/ad/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
    a = ad.read_h5ad(f'{H}/bench_real_discovery/data/ad.h5ad', backed='r'); o = a.obs.copy(); xy = np.asarray(a.obsm['spatial'])
    ct = o.cell_type.astype(str).to_numpy(); samp = o.slice_id.astype(str).to_numpy(); anat = o.tissue_domain.astype(str).to_numpy()
    Dt = pd.read_csv(f'{H}/final_results_nb/starmap_ad/biology/plaque_distance/starmap_plaque_distance_cells.csv.gz').set_index('cell_id')
    dist = pd.Series(a.obs_names).map(Dt.distance_to_abeta_associated_cell_spatial_units).to_numpy(float) / 31.4   # cell spacings
    bins = [-1, 1e-9, 1, 2, 4, 8, np.inf]; unit = 'cell spacings'; MIC = 'Micro'
else:
    V = 'all7_no_oligo'
    MD = f'{H}/results_codex_wholeslice/codex_wholeslice_{V}/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
    a = ad.read_h5ad(f'{H}/bench_codex_cns_2026/wholeslice/codex_wholeslice_{V}.h5ad', backed='r'); o = a.obs.copy(); xy = np.asarray(a.obsm['spatial_um'])
    ct = o.cell_type.astype(str).to_numpy(); samp = o.donor_id.astype(str).to_numpy(); anat = o.region.astype(str).to_numpy()
    dist = np.load(f'{H}/bench_codex_cns_2026/wholeslice/plaque_states/cell_dist_any_abeta_um.npy').astype(float)
    bins = [0, 10, 20, 40, 60, 100, np.inf]; unit = 'um'; MIC = 'Microglia'
dis = (o.condition.astype(str) == 'disease').to_numpy()
meta = json.load(open(MD + '/meta.json')); cats = meta['cell_type_categories']
assert (np.load(MD + '/obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
log = open(f'{OUT}/check_cf_proportion_{DS}.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

code = pd.Categorical(ct, categories=cats).codes
obs = np.zeros((len(ct), len(cats)), np.float32)
for sm in np.unique(samp):
    i = np.where(samp == sm)[0]; _, nb = cKDTree(xy[i]).query(xy[i], k=K + 1)
    for c in range(1, K + 1): np.add.at(obs, (i, code[i[nb[:, c]]]), 1.0)
obs /= K
pbg = np.load(MD + '/background_composition.npy'); pf = np.load(MD + '/condition_composition.npy'); dp = np.load(MD + '/salient_composition_shift.npy')
P(f'== {DS}: {len(ct)} cells ({dis.sum()} condition, {(~dis).sum()} reference); samples {sorted(set(samp))}')

# C1
P('\n-- C1 bookkeeping --')
P('   rows sum to 1: p_bg max |1-sum| %.1e, p_full %.1e; min entry p_bg %.1e p_full %.1e' %
  (np.abs(pbg.sum(1) - 1).max(), np.abs(pf.sum(1) - 1).max(), pbg.min(), pf.min()))
P('   dp == p_full - p_bg on condition cells: max err %.1e;  dp on reference cells: max |.| %.1e' %
  (np.abs(dp[dis] - (pf - pbg)[dis]).max(), np.abs(dp[~dis]).max()))
P('   NOTE p_full on reference cells = D([z,s]) with their own s; Haruka sets dp=0 there, so only p_bg is used for reference cells')

# C2
def r2(y, f): return 1 - ((y - f) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
def tv(p, q): return 0.5 * np.abs(p - q).sum(-1)
P('\n-- C2 fit to the observed neighbourhood (R2 over all columns; mean total-variation distance) --')
P('   reference cells : p_bg   R2 %.3f  TV %.3f' % (r2(obs[~dis], pbg[~dis]), tv(obs[~dis], pbg[~dis]).mean()))
P('   condition cells : p_bg   R2 %.3f  TV %.3f' % (r2(obs[dis], pbg[dis]), tv(obs[dis], pbg[dis]).mean()))
P('   condition cells : p_full R2 %.3f  TV %.3f' % (r2(obs[dis], pf[dis]), tv(obs[dis], pf[dis]).mean()))

# C3 matched strata
P('\n-- C3 is the background reference-like?  strata = anatomy x own cell type, >=30 cells per arm --')
rows = []
for (g, c), idx in pd.DataFrame({'g': anat, 'c': ct}).groupby(['g', 'c']).groups.items():
    idx = np.asarray(idx); r, d = idx[~dis[idx]], idx[dis[idx]]
    if len(r) < 30 or len(d) < 30: continue
    rows.append(dict(anat=g, own=c, n_ref=len(r), n_cond=len(d), obs_ref=obs[r].mean(0), obs_cond=obs[d].mean(0),
                     pbg_cond=pbg[d].mean(0), pfull_cond=pf[d].mean(0), pbg_ref=pbg[r].mean(0)))
S = pd.DataFrame(rows); w = S.n_cond.to_numpy(float); w /= w.sum()
P(f'   {len(S)} strata, covering {S.n_cond.sum()} of {dis.sum()} condition cells')
tvs = {k: np.array([tv(S[a_][i], S[b_][i]) for i in range(len(S))]) for k, (a_, b_) in {
    'TV(obs cond, obs ref)   [real condition effect]': ('obs_cond', 'obs_ref'),
    'TV(p_bg cond, obs ref)  [background vs reference]': ('pbg_cond', 'obs_ref'),
    'TV(p_bg cond, obs cond) [background vs condition]': ('pbg_cond', 'obs_cond'),
    'TV(p_full cond, obs cond)[salient vs condition]': ('pfull_cond', 'obs_cond'),
    'TV(p_bg ref, obs ref)   [head calibration on reference]': ('pbg_ref', 'obs_ref')}.items()}
for k, v in tvs.items(): P(f'   {k:58s} {np.dot(w, v):.4f}')
M = lambda col: np.einsum('i,ij->j', w, np.stack(S[col]))
T = pd.DataFrame({c: M(c) for c in ['obs_ref', 'obs_cond', 'pbg_cond', 'pfull_cond']}, index=cats)
T['cond_effect'] = T.obs_cond - T.obs_ref
T['bg_share'] = (T.pbg_cond - T.obs_ref) / T.cond_effect          # 0 = background is reference-like
T['full_share'] = (T.pfull_cond - T.obs_ref) / T.cond_effect      # 1 = salient reproduces the condition
T.loc[T.cond_effect.abs() < 0.003, ['bg_share', 'full_share']] = np.nan
P('   per neighbour type, matched means (share = fraction of the condition effect obs_cond - obs_ref):')
P(T.round(4).to_string())
T.to_csv(f'{OUT}/check_cf_C3_{DS}.csv')

# C4 plaque
P(f'\n-- C4 plaque, condition cells: Spearman with log distance within anatomy (negative = higher near plaque) --')
ld = np.log1p(np.where(np.isfinite(dist), dist, np.nan)); ok = dis & np.isfinite(ld)
def within(v):
    num = den = 0.
    for g in np.unique(anat[ok]):
        m = ok & (anat == g)
        if m.sum() < 200 or np.std(v[m]) < 1e-9: continue
        num += spearmanr(v[m], ld[m])[0] * m.sum(); den += m.sum()
    return num / den if den else np.nan
C4 = pd.DataFrame({'observed': [within(obs[:, j]) for j in range(len(cats))], 'p_bg': [within(pbg[:, j]) for j in range(len(cats))],
                   'p_full': [within(pf[:, j]) for j in range(len(cats))], 'dp': [within(dp[:, j]) for j in range(len(cats))]}, index=cats)
P(C4.round(3).to_string()); C4.to_csv(f'{OUT}/check_cf_C4_{DS}.csv')
j = cats.index(MIC); sub = ok if DS == 'starmap' else ok & (anat == 'GM')
P(f'\n   {MIC} neighbour fraction by distance bin ({unit}{"" if DS == "starmap" else ", grey matter"}); reference-arm observed mean '
  f'{obs[~dis & ((anat == "GM") if DS == "codex" else True), j].mean():.4f}')
tb = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = sub & (dist > lo) & (dist <= hi)
    tb.append(dict(bin=f'({lo:g},{hi:g}]', n=int(m.sum()), observed=obs[m, j].mean(), p_bg=pbg[m, j].mean(), p_full=pf[m, j].mean(), dp=dp[m, j].mean()))
P(pd.DataFrame(tb).round(4).to_string(index=False)); pd.DataFrame(tb).to_csv(f'{OUT}/check_cf_C4_microglia_bins_{DS}.csv', index=False)
tb = []                                                     # same, per condition sample (slice / donor)
for sm in np.unique(samp[dis]):
    ref = obs[~dis & ((anat == 'GM') if DS == 'codex' else True), j].mean()
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = sub & (samp == sm) & (dist > lo) & (dist <= hi)
        if m.sum() >= 10:
            tb.append(dict(sample=sm, bin=f'({lo:g},{hi:g}]', n=int(m.sum()), observed=obs[m, j].mean(), p_bg=pbg[m, j].mean(),
                           p_full=pf[m, j].mean(), dp=dp[m, j].mean(), reference_observed=ref))
pd.DataFrame(tb).to_csv(f'{OUT}/check_cf_C4_microglia_bins_by_sample_{DS}.csv', index=False)
# C3 matched decomposition per condition sample (strata matched to the pooled reference arm)
rows = []
for sm in np.unique(samp[dis]):
    for (g, c), idx in pd.DataFrame({'g': anat, 'c': ct}).groupby(['g', 'c']).groups.items():
        idx = np.asarray(idx); r, d = idx[~dis[idx]], idx[dis[idx] & (samp[idx] == sm)]
        if len(r) < 30 or len(d) < 30: continue
        for jj, nm in enumerate(cats):
            rows.append(dict(sample=sm, anat=g, own=c, type=nm, n=len(d), obs_ref=obs[r, jj].mean(), obs_cond=obs[d, jj].mean(),
                             pbg_cond=pbg[d, jj].mean(), pfull_cond=pf[d, jj].mean()))
Q = pd.DataFrame(rows)
Q = Q.groupby(['sample', 'type']).apply(lambda x: pd.Series({k: np.average(x[k], weights=x.n) for k in ['obs_ref', 'obs_cond', 'pbg_cond', 'pfull_cond']}))
Q.to_csv(f'{OUT}/check_cf_C3_by_sample_{DS}.csv')
P('done')
