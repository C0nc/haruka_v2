#!/usr/bin/env python3
"""OSKM vs old control: does per-cell z or s better capture rejuvenation?  (no z-CC / s-CC)

Fit: results_real_compgated_nb_full_oskm/oskm_old_full_684253 (old control = reference, old OSKM = condition).
Endpoints
  youngward      per-cell projection of log expression from the matched old-control centroid to the
                 young centroid within cell type x subregion (score_unscaled; unsmoothed) and its k=15
                 spatial mean (local_score_unscaled), OSKM cells
  clock age      published frozen Spatial Aging Clock predicted age, both arms
Tests (leave-one-pair-out ridge, alpha 100, features standardised on training pairs)
  A  within-stratum heterogeneity, OSKM cells: X and y centred within pair x cell type x subregion
     -> which latent ranks cells by how youngward they are (Spearman per held-out pair)
  B  arm effect on the clock, both arms: y = clock age centred within pair x cell type x subregion
     (pooled arms), X centred the same way -> which latent carries the OSKM-vs-control age shift;
     reported overall and as the held-out OSKM-minus-control predicted difference vs the observed one
"""
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = '/home/yancui/haruka_v2'; OUT = f'{H}/bench_oskm_reprogramming/zs_rejuvenation'
MD = f'{H}/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
A_ = f'{H}/final_results_agent_oskm'
log = open(f'{OUT}/zs_rejuvenation.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

names = pd.Index(np.load(f'{MD}/obs_names.npy', allow_pickle=True).astype(str))
Z = np.load(f'{MD}/background.npy').astype(np.float64); S = np.load(f'{MD}/salient.npy').astype(np.float64)
C = pd.read_csv(f'{A_}/spatial_aging_clock_extension/clock_full/spatial_clock_predictions.csv.gz').set_index('cell_id')
R = pd.read_csv(f'{A_}/tables/oskm_cell_scores_and_domains.csv.gz').set_index('cell_id')
P(f'model rows {len(names)} | clock rows {len(C)} (matched {names.isin(C.index).mean():.3f}) | score rows {len(R)} (matched {R.index.isin(names).mean():.3f})')

def centre(X, g):
    X = np.array(X, dtype=np.float64, copy=True)
    df = pd.DataFrame(X); X -= df.groupby(g).transform('mean').to_numpy(); return X

def lopo(Xs, y, pair):
    """returns dict rep -> (pooled Spearman, per-pair Spearman list, held-out predictions)"""
    out = {}
    for k, X in Xs.items():
        pred = np.zeros(len(y)); per = []
        for p in np.unique(pair):
            tr, te = pair != p, pair == p
            sc = StandardScaler().fit(X[tr])
            pred[te] = Ridge(alpha=100.0, solver='lsqr', tol=1e-4).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
            per.append(spearmanr(pred[te], y[te]).statistic)
        out[k] = (spearmanr(pred, y).statistic, per, pred)
    return out

# ---------------- A: youngward heterogeneity within strata, OSKM cells ----------------
rows = names.get_indexer(R.index); ok = (rows >= 0) & np.isfinite(R.score_unscaled.to_numpy()) & np.isfinite(R.local_score_unscaled.to_numpy())
P(f'OSKM cells with finite scores: {ok.sum()} of {len(ok)}'); R = R[ok]; rows = rows[ok]
g = (R.paired_batch.astype(str) + '|' + R.cell_type.astype(str) + '|' + R.subregion.astype(str)).to_numpy()
pair = R.paired_batch.astype(str).to_numpy()
Xa = {'z': centre(Z[rows], g), 's': centre(S[rows], g)}; Xa['z+s'] = np.hstack([Xa['z'], Xa['s']])
res = []
P('\n-- A. within-stratum youngward heterogeneity (OSKM cells, leave-one-pair-out) --')
for ep in ['score_unscaled', 'local_score_unscaled']:
    y = centre(R[ep].to_numpy()[:, None], g).ravel()
    o = lopo(Xa, y, pair)
    for k, (r, per, _) in o.items():
        res.append(dict(test='A', endpoint=ep, rep=k, spearman=r, **{f'pair_{p}': v for p, v in zip(np.unique(pair), per)}))
        P(f'   {ep:22s} {k:4s} Spearman {r:+.3f} | held-out pairs ' + ', '.join(f'{p} {v:+.3f}' for p, v in zip(np.unique(pair), per)))
# per cell type (s - z), unsmoothed endpoint
y = centre(R['score_unscaled'].to_numpy()[:, None], g).ravel(); o = lopo({'z': Xa['z'], 's': Xa['s']}, y, pair)
ct = R.cell_type.astype(str).to_numpy(); tab = []
for c in pd.Series(ct).value_counts().index:
    m = ct == c
    if m.sum() < 500: continue
    rz = [spearmanr(o['z'][2][m & (pair == p)], y[m & (pair == p)]).statistic for p in np.unique(pair) if (m & (pair == p)).sum() >= 50]
    rs = [spearmanr(o['s'][2][m & (pair == p)], y[m & (pair == p)]).statistic for p in np.unique(pair) if (m & (pair == p)).sum() >= 50]
    tab.append(dict(cell_type=c, n=int(m.sum()), z=np.mean(rz), s=np.mean(rs), s_minus_z=np.mean(rs) - np.mean(rz),
                    s_wins_pairs=f'{int(np.sum(np.array(rs) > np.array(rz)))}/{len(rs)}'))
T = pd.DataFrame(tab); P('\n   per cell type, unsmoothed youngward, mean held-out-pair Spearman:'); P('   ' + T.round(3).to_string(index=False).replace('\n', '\n   '))
T.to_csv(f'{OUT}/A_by_celltype.csv', index=False)

# ---------------- B: clock age, both arms ----------------
rows = names.get_indexer(C.index); ok = (rows >= 0) & np.isfinite(C.predicted_age.to_numpy()); C = C[ok]; rows = rows[ok]
gB = (C.paired_batch.astype(str) + '|' + C.cell_type.astype(str) + '|' + C.subregion.astype(str)).to_numpy()
pairB = C.paired_batch.astype(str).to_numpy(); arm = (C.condition.astype(str) == 'oskm').to_numpy() if 'oskm' in set(C.condition.astype(str)) else (C.cohort.astype(str) == 'old_treatment').to_numpy()
y = centre(C.predicted_age.to_numpy()[:, None], gB).ravel()
Xb = {'z': centre(Z[rows], gB), 's': centre(S[rows], gB)}; Xb['z+s'] = np.hstack([Xb['z'], Xb['s']])
P(f'\n-- B. clock age centred within pair x cell type x subregion, both arms (n = {len(y)}, OSKM {arm.mean():.1%}) --')
o = lopo(Xb, y, pairB)
for k, (r, per, pred) in o.items():
    d_obs = [y[(pairB == p) & arm].mean() - y[(pairB == p) & ~arm].mean() for p in np.unique(pairB)]
    d_pred = [pred[(pairB == p) & arm].mean() - pred[(pairB == p) & ~arm].mean() for p in np.unique(pairB)]
    res.append(dict(test='B', endpoint='clock_age_centred', rep=k, spearman=r, **{f'pair_{p}': v for p, v in zip(np.unique(pairB), per)}))
    P(f'   {k:4s} Spearman {r:+.3f} | per pair ' + ', '.join(f'{p} {v:+.3f}' for p, v in zip(np.unique(pairB), per)))
    P(f'        OSKM - control age (months, within strata): observed ' + ', '.join(f'{p} {v:+.2f}' for p, v in zip(np.unique(pairB), d_obs))
      + ' | predicted ' + ', '.join(f'{v:+.2f}' for v in d_pred))
# s on reference cells: how much s varies in controls vs OSKM (s should be quiet in the reference arm)
P('\n   ||s|| mean: control %.3f  OSKM %.3f | ||z|| mean: control %.3f  OSKM %.3f' % (
    np.linalg.norm(S[rows][~arm], axis=1).mean(), np.linalg.norm(S[rows][arm], axis=1).mean(),
    np.linalg.norm(Z[rows][~arm], axis=1).mean(), np.linalg.norm(Z[rows][arm], axis=1).mean()))
pd.DataFrame(res).to_csv(f'{OUT}/zs_rejuvenation_summary.csv', index=False)
P('done')
