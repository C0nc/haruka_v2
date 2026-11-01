#!/usr/bin/env python3
"""OSKM: split the published Spatial Aging Clock shift into Haruka's z part and s part.

Fit: results_oskm_cf_expression (old control = reference, old OSKM = condition, decoder batch = pair),
which saved decoded expression at library size 250 (the clock's target sum):
  bg  = D([z,0]) for every cell;  sal = D([z,s]) (used for OSKM cells; control cells keep bg)
The frozen clock (Sun et al., lasso_cv5_nalphas20_spatialsmooth_alpha08_neigh20) is run exactly as
in final_results_agent_oskm/.../run_official_clock_parallel.py (normalise 250, log1p, spatial smoothing
within mouse x cell type, frozen StandardScaler+Lasso) on:  observed counts (cached), bg, and bg-with-sal-for-OSKM.
Per pair x cell type x subregion stratum (>= 20 cells per arm), medians:
  observed shift = clock_obs(OSKM) - clock_obs(control)
  z part         = clock_bg(OSKM)  - clock_bg(control)        (what the shared space explains)
  s part         = clock_sal(OSKM) - clock_bg(OSKM)           (what s adds)
Sanity: clock_bg(control) vs clock_obs(control).  Gene level: effective lasso weight x (sal - bg) per cell type.
"""
import os, sys, json, pickle, warnings, multiprocessing as mp
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'bench_oskm_reprogramming/zs_rejuvenation/cf_clock'; OUT.mkdir(exist_ok=True)
MD = H / 'results_oskm_cf_expression/oskm_old_full_684253/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
EXT = H / 'final_results_agent_oskm/spatial_aging_clock_extension'
sys.path.insert(0, str(EXT / 'scripts')); import run_official_clock_parallel as clk
REPO = H / 'bench_oskm_reprogramming/external/SpatialAgingClock'
log = open(OUT / 'cf_clock.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

a = ad.read_h5ad(H / 'bench_oskm_reprogramming/data/oskm_old_full_684253.h5ad', backed='r'); o = a.obs
names = np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str); assert (names == np.asarray(a.obs_names).astype(str)).all()
bg = np.load(MD / 'cf_expression_background.npy'); sal = np.load(MD / 'cf_expression_salient.npy')
cond = np.load(MD / 'cf_expression_condition_mask.npy')
P(f'decoded: {bg.shape}, row sums bg median {np.median(bg.sum(1)):.1f}; OSKM cells {cond.sum()}')
salfull = bg.copy(); salfull[cond] = sal[cond]
ct = o.celltype.astype(str).to_numpy(); mouse = o.mouse_id.astype(str).to_numpy()
spatial = np.asarray(a.obsm['spatial']); varn = np.asarray(a.var_names).astype(str)

def run_clock(X, tag):
    f = OUT / f'clock_{tag}.npy'
    if f.exists(): return np.load(f)
    clk.G_COUNTS = sparse.csr_matrix(X); clk.G_SPATIAL = spatial; clk.G_VARS = varn; clk.G_CLOCK_REPO = str(REPO)
    clk.G_PARTIALS = str(OUT / f'partials_{tag}'); Path(clk.G_PARTIALS).mkdir(exist_ok=True)
    tasks = [(c, mm, np.flatnonzero((ct == c) & (mouse == mm)).astype(np.int64)) for c in sorted(set(ct)) for mm in sorted(set(mouse[ct == c]))]
    pred = np.full(len(ct), np.nan)
    with mp.get_context('fork').Pool(16) as pool:
        for r in pool.imap_unordered(clk._worker, tasks):
            pred[r['rows']] = r['pred']
    np.save(f, pred); return pred

obs_cache = pd.read_csv(EXT / 'clock_full/spatial_clock_predictions.csv.gz').set_index('cell_id').reindex(names)
c_obs = obs_cache.predicted_age.to_numpy()
c_bg = run_clock(bg, 'background'); P('clock on background done')
c_sal = run_clock(salfull, 'salient'); P('clock on salient done')
oskm = (o.condition.astype(str) == 'oskm').to_numpy(); assert (oskm == cond).all()
pair = o.paired_batch.astype(str).to_numpy(); sub = o.subregion.astype(str).to_numpy()

P('\n-- sanity: clock on decoded [z,0] vs clock on observed counts --')
for arm, m in [('control', ~oskm), ('OSKM', oskm)]:
    P(f'   {arm:8s} Pearson {pearsonr(c_bg[m], c_obs[m])[0]:.3f}  Spearman {spearmanr(c_bg[m], c_obs[m]).statistic:.3f}  '
      f'median obs {np.median(c_obs[m]):.2f}  median bg {np.median(c_bg[m]):.2f} months')
per_ct = pd.DataFrame([dict(cell_type=c, control_r=pearsonr(c_bg[(~oskm) & (ct == c)], c_obs[(~oskm) & (ct == c)])[0]) for c in sorted(set(ct))])
P('   per cell type (control cells) Pearson: ' + ', '.join(f'{r.cell_type} {r.control_r:.2f}' for r in per_ct.itertuples()))

rows = []
for (p, c, r), idx in pd.DataFrame({'p': pair, 'c': ct, 'r': sub}).groupby(['p', 'c', 'r']).indices.items():
    T, C = idx[oskm[idx]], idx[~oskm[idx]]
    if len(T) < 20 or len(C) < 20: continue
    rows.append(dict(pair=p, cell_type=c, subregion=r, n_oskm=len(T), n_ctrl=len(C),
                     observed=np.median(c_obs[T]) - np.median(c_obs[C]),
                     z_part=np.median(c_bg[T]) - np.median(c_bg[C]),
                     s_part=np.median(c_sal[T] - c_bg[T]),
                     decoded_total=np.median(c_sal[T]) - np.median(c_bg[C])))
S = pd.DataFrame(rows); S.to_csv(OUT / 'strata_shift.csv', index=False)
P(f'\n-- clock shift split, {len(S)} pair x cell type x subregion strata (months; + = OSKM older) --')
P('   across strata: corr(observed, z part) r %.2f | corr(observed, s part) r %.2f | corr(observed, decoded total) r %.2f' %
  (pearsonr(S.observed, S.z_part)[0], pearsonr(S.observed, S.s_part)[0], pearsonr(S.observed, S.decoded_total)[0]))
P('   sign agreement with observed: z part %.0f%% | s part %.0f%% | decoded total %.0f%%' %
  (100 * np.mean(np.sign(S.z_part) == np.sign(S.observed)), 100 * np.mean(np.sign(S.s_part) == np.sign(S.observed)),
   100 * np.mean(np.sign(S.decoded_total) == np.sign(S.observed))))
w = S.n_oskm
P('   n-weighted mean |part| / |observed|: z %.2f  s %.2f' % (np.average(np.abs(S.z_part), weights=w) / np.average(np.abs(S.observed), weights=w),
                                                           np.average(np.abs(S.s_part), weights=w) / np.average(np.abs(S.observed), weights=w)))
P('\n   by pair (median over strata):'); P('   ' + S.groupby('pair')[['observed', 'z_part', 's_part', 'decoded_total']].median().round(2).to_string().replace('\n', '\n   '))
# within each cell type: across its subregion x pair strata, which part tracks the observed shift?
tab = []
for c, g in S.groupby('cell_type'):
    if len(g) < 6: continue
    tab.append(dict(cell_type=c, strata=len(g), observed=g.observed.median(), z_part=g.z_part.median(), s_part=g.s_part.median(),
                    r_obs_z=pearsonr(g.observed, g.z_part)[0], r_obs_s=pearsonr(g.observed, g.s_part)[0]))
T = pd.DataFrame(tab).sort_values('strata', ascending=False); T.to_csv(OUT / 'by_celltype.csv', index=False)
P('\n   within each cell type (across its pair x subregion strata): medians and corr with observed shift')
P('   ' + T.round(2).to_string(index=False).replace('\n', '\n   '))
P(f'   s tracks observed better than z in {int((T.r_obs_s > T.r_obs_z).sum())}/{len(T)} cell types')

# gene level: effective lasso weight (coef / scaler scale) x mean (log1p sal - log1p bg) per cell type, OSKM cells
P('\n-- clock genes moved by s (OSKM cells; weight x mean log-expression change; + = pushes older) --')
grows = []
for c in sorted(set(ct)):
    stem = f'{clk.CLOCK_NAME}_{c}'; cd_ = REPO / 'spatialclock/clocks'
    if not (cd_ / f'{stem}.pkl').exists(): continue
    with open(cd_ / f'{stem}.pkl', 'rb') as fh: pipe = pickle.load(fh)
    genes = pd.read_csv(cd_ / 'mean_impute_files' / f'{stem}.csv').columns.astype(str)
    sc, lasso = pipe.steps[0][1], pipe.steps[-1][1]; wv = lasso.coef_ / sc.scale_
    pos = pd.Index(varn).get_indexer(genes); ok = pos >= 0; m = oskm & (ct == c)
    d = np.log1p(sal[np.flatnonzero(m)[:, None], pos[ok]] if False else salfull[m][:, pos[ok]]).mean(0) - np.log1p(bg[m][:, pos[ok]]).mean(0)
    contrib = wv[ok] * d
    for gname, wg, dg, cg in zip(genes[ok], wv[ok], d, contrib):
        grows.append(dict(cell_type=c, gene=gname, weight=wg, delta_log=dg, contribution=cg))
G = pd.DataFrame(grows); G.to_csv(OUT / 'gene_contributions.csv', index=False)
for c, g in G.groupby('cell_type'):
    g = g.reindex(g.contribution.abs().sort_values(ascending=False).index).head(6)
    P(f'   {c:18s} sum {G[G.cell_type == c].contribution.sum():+.2f} | ' + ', '.join(f'{r.gene} {r.contribution:+.2f}' for r in g.itertuples()))
P('done')
