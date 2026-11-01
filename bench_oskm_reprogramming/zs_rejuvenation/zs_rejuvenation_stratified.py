#!/usr/bin/env python3
"""OSKM: per-cell z vs s, analysed INSIDE each cell type x subregion stratum (no pooled model).

For every stratum with >= MIN cells in each of the 4 pairs:
  X (z or s) and y centred within pair (removes pair offsets inside the stratum),
  ridge(alpha 100) trained on 3 pairs, predicts the held-out pair; Spearman per held-out pair,
  stratum score = mean over the 4 held-out pairs.
Endpoints: youngward score (per cell, unsmoothed; OSKM cells) and clock age (both arms).
Summary across strata: median s - z, fraction of strata where s > z, Wilcoxon signed-rank over strata.
"""
import os, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8'); os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = '/home/yancui/haruka_v2'; OUT = f'{H}/bench_oskm_reprogramming/zs_rejuvenation'; MIN = 100
MD = f'{H}/results_real_compgated_nb_full_oskm/oskm_old_full_684253/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
A_ = f'{H}/final_results_agent_oskm'
log = open(f'{OUT}/zs_rejuvenation_stratified.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
names = pd.Index(np.load(f'{MD}/obs_names.npy', allow_pickle=True).astype(str))
Z = np.load(f'{MD}/background.npy').astype(np.float64); S = np.load(f'{MD}/salient.npy').astype(np.float64)

def pair_centre(X, pair):
    X = X.copy()
    for p in np.unique(pair):
        m = pair == p; X[m] -= X[m].mean(0)
    return X

def run(frame, ycol, tag):
    rows = names.get_indexer(frame.index)
    ok = (rows >= 0) & np.isfinite(frame[ycol].to_numpy()); frame = frame[ok]; rows = rows[ok]
    out = []
    for (c, r), idx in frame.groupby(['cell_type', 'subregion']).indices.items():
        pair = frame.paired_batch.astype(str).to_numpy()[idx]
        cnt = pd.Series(pair).value_counts()
        if len(cnt) < 4 or cnt.min() < MIN: continue
        y = pair_centre(frame[ycol].to_numpy()[idx][:, None], pair).ravel()
        rec = dict(cell_type=c, subregion=r, n=len(idx))
        for k, M in [('z', Z), ('s', S)]:
            X = pair_centre(M[rows[idx]], pair); per = []
            for p in np.unique(pair):
                tr, te = pair != p, pair == p
                sc = StandardScaler().fit(X[tr])
                pr = Ridge(alpha=100.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
                per.append(spearmanr(pr, y[te]).statistic)
            rec[k] = np.mean(per); rec[f'{k}_pairs'] = per
        rec['s_minus_z'] = rec['s'] - rec['z']; rec['s_wins_pairs'] = int(np.sum(np.array(rec['s_pairs']) > np.array(rec['z_pairs'])))
        out.append(rec)
    T = pd.DataFrame(out).sort_values('s_minus_z', ascending=False)
    T.drop(columns=['z_pairs', 's_pairs']).to_csv(f'{OUT}/stratified_{tag}.csv', index=False)
    w = wilcoxon(T.s, T.z)
    P(f'\n== {tag}: {len(T)} strata (cell type x subregion, >= {MIN} cells in each pair), {T.n.sum()} cells')
    P(f'   median Spearman z {T.z.median():.3f}  s {T.s.median():.3f} | median s - z {T.s_minus_z.median():+.3f} | '
      f's > z in {np.mean(T.s > T.z):.0%} of strata | Wilcoxon p = {w.pvalue:.2g} | n-weighted s - z {np.average(T.s_minus_z, weights=T.n):+.3f}')
    by = T.groupby('cell_type').apply(lambda x: pd.Series({'strata': len(x), 'z': x.z.median(), 's': x.s.median(),
                                                          's>z strata': f'{int((x.s > x.z).sum())}/{len(x)}'})).sort_values('strata', ascending=False)
    P('   by cell type (median over its subregions):'); P('   ' + by.round(3).to_string().replace('\n', '\n   '))
    by2 = T.groupby('subregion').apply(lambda x: pd.Series({'strata': len(x), 'z': x.z.median(), 's': x.s.median(),
                                                           's>z strata': f'{int((x.s > x.z).sum())}/{len(x)}'}))
    P('   by subregion:'); P('   ' + by2.round(3).to_string().replace('\n', '\n   '))
    P('   top strata where s > z:'); P('   ' + T.head(6)[['cell_type', 'subregion', 'n', 'z', 's', 's_wins_pairs']].round(3).to_string(index=False).replace('\n', '\n   '))
    P('   top strata where z > s:'); P('   ' + T.tail(6)[['cell_type', 'subregion', 'n', 'z', 's', 's_wins_pairs']].round(3).to_string(index=False).replace('\n', '\n   '))

R = pd.read_csv(f'{A_}/tables/oskm_cell_scores_and_domains.csv.gz').set_index('cell_id')
run(R, 'score_unscaled', 'youngward_oskm')
C = pd.read_csv(f'{A_}/spatial_aging_clock_extension/clock_full/spatial_clock_predictions.csv.gz').set_index('cell_id')
run(C[C.condition.astype(str) == C.condition.astype(str).iloc[-1]] if False else C, 'predicted_age', 'clock_age_both_arms')
oskm = C.cohort.astype(str) == 'old_treatment'
run(C[oskm], 'predicted_age', 'clock_age_oskm')
P('done')
