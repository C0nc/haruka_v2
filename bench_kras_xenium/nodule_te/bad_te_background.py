#!/usr/bin/env python3
"""KRAS: which BACKGROUND feature of a treated nodule goes with a 'bad' treatment effect (TE)?
Bad TE = the nodule's predicted Itgb6+ conversion: mean over its tumour cells of the Tumor_epithelial_Itgb6 component of
D([z,s]) - D([z,0]) (PC1 of the nodule TE is 94% this swap).  Fit: results_kras_te_null (saved model + 3 shuffled-s nulls).
Nodules as nodule_te.py (tumour cells, 25 um radius graph per section, >= 50 cells); 132 treated nodules, 4 sections.
Candidate background features per nodule:
  pbg:<type>   predicted untreated neighbourhood D([z,0]) share, non-tumour neighbour types
  obs:<type>   observed k=20 neighbour share, non-tumour types (post-treatment context, read with care)
  geo:*        log size, tumour density (tumour share of the k=20 neighbourhood), distance to nearest airway (Ciliated)
               and nearest vessel (Arterial_EC / Vein_EC) from the nodule's cells (median um)
  zPC1-5       nodule mean z, section-centred PCA
Three checks per feature, all Spearman within section then median over the 4 sections (and how many share the sign):
  real   vs the real Itgb6+ TE                    (partial: on log size + density, pooled within section)
  null   vs the same TE decoded from shuffled s (mean |rho| over 3 nulls)  -> decoder mechanics if comparable
  resp   vs the nodule responder rate (MAPK MPAS, never seen by Haruka)   -> independent biology
"""
import os as _os, pathlib as _pl
_HR = _os.environ.get('HARUKA_ROOT') or str(next(p for p in _pl.Path(__file__).resolve().parents if (p / 'notebooks' / 'repro.py').exists()))
import os, json, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
H = Path(f'{_HR}'); OUT = H / 'bench_kras_xenium/nodule_te'
MD = H / 'results_kras_te_null/mouselung_kras_xenium_clean_noC3/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
a = ad.read_h5ad(f'{_HR}/data/mouselung_kras_xenium_clean_noC3.h5ad', backed='r'); o = a.obs
assert (np.load(MD / 'obs_names.npy', allow_pickle=True).astype(str) == np.asarray(a.obs_names).astype(str)).all()
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
TUMOR = ['Tumor_EMT', 'Tumor_epithelial', 'Tumor_epithelial_Itgb6']
ct = o.celltype.astype(str).to_numpy(); sl = o.slice_id.astype(str).to_numpy(); treated = (o.condition_label.astype(str) == 'KRASi_treated').to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]; tum = np.isin(ct, TUMOR)
jI = types.index('Tumor_epithelial_Itgb6'); nontum = [i for i, t in enumerate(types) if t not in TUMOR]
dp = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r'); pbg = np.load(MD / 'background_composition.npy', mmap_mode='r')
nulls = [np.load(MD / f'null_shuffled_s_composition_shift_{k}.npy', mmap_mode='r') for k in range(3)]
Z = np.load(MD / 'background.npy', mmap_mode='r')
sig = pd.read_csv(H / 'bench_kras_xenium/biological_responder/all_tumor_signature_scores.csv.gz')
mpas = np.full(len(ct), np.nan); mpas[tum] = sig.MAPK_MPAS.to_numpy()
p25 = {t: np.nanpercentile(mpas[tum & ~treated & (ct == t)], 25) for t in TUMOR}
resp = tum & (np.nan_to_num(mpas, nan=np.inf) < np.array([p25.get(c, -np.inf) for c in ct]))
code = pd.Categorical(ct, categories=types).codes
log = open(OUT / 'bad_te_background.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
rows = []
for s_ in np.unique(sl[treated]):
    m = np.flatnonzero(sl == s_); q = m[tum[m]]
    _, nb = NearestNeighbors(n_neighbors=21).fit(xy[m]).kneighbors(xy[q]); nbc = code[m][nb[:, 1:]]
    obs = np.stack([(nbc == j).mean(1) for j in range(len(types))], 1)
    _, lab = connected_components(radius_neighbors_graph(xy[q], 25.0, mode='connectivity'), directed=False)
    air = cKDTree(xy[m][ct[m] == 'Ciliated']); ves = cKDTree(xy[m][np.isin(ct[m], ['Arterial_EC', 'Vein_EC'])])
    for l in np.flatnonzero(np.bincount(lab) >= 50):
        k = lab == l; idx = q[k]
        rec = dict(section=s_, n=len(idx), log_size=np.log(len(idx)), density=obs[k][:, [types.index(t) for t in TUMOR]].sum(1).mean(),
                   d_airway=np.median(air.query(xy[idx])[0]), d_vessel=np.median(ves.query(xy[idx])[0]),
                   te=np.asarray(dp[idx])[:, jI].mean(), resp=resp[idx].mean(), mpas=np.nanmean(mpas[idx]),
                   **{f'null{j}': np.asarray(nu[idx])[:, jI].mean() for j, nu in enumerate(nulls)})
        PB = np.asarray(pbg[idx]).mean(0); rec.update({f'pbg:{types[i]}': PB[i] for i in nontum})
        rec.update({f'obs:{types[i]}': obs[k][:, i].mean() for i in nontum})
        rec['_z'] = np.asarray(Z[idx]).mean(0)
        rows.append(rec)
N = pd.DataFrame(rows); P(f'treated nodules {len(N)}:', N.section.value_counts().to_dict())
Zn = np.stack(N.pop('_z').to_numpy()); sec = N.section.to_numpy()
Zc = Zn.copy()
for u in np.unique(sec): Zc[sec == u] -= Zc[sec == u].mean(0)
zp = PCA(5, random_state=0).fit_transform(Zc)
for j in range(5): N[f'zPC{j+1}'] = zp[:, j]
N.to_csv(OUT / 'bad_te_nodules.csv', index=False)
feats = [c for c in N.columns if c.startswith(('pbg:', 'obs:', 'zPC')) or c in ('log_size', 'density', 'd_airway', 'd_vessel')]
feats = [f for f in feats if N[f].std() > 0 and N[f].nunique() > 5]
def within(f, y):
    r = []
    for u in np.unique(sec):
        m = sec == u
        if m.sum() >= 6 and N.loc[m, f].nunique() > 2: r.append(spearmanr(N.loc[m, f], N.loc[m, y]).statistic)
    return np.array(r)
def partial(f, y):
    # rank-residualise feature and outcome on log size + density within section, then correlate
    def res(v):
        out = np.zeros(len(N))
        for u in np.unique(sec):
            m = sec == u; X = np.c_[np.ones(m.sum()), rankdata(N.log_size[m]), rankdata(N.density[m])]; yv = rankdata(v[m])
            out[m] = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
        return out
    return np.corrcoef(res(N[f].to_numpy()), res(N[y].to_numpy()))[0, 1]
R = []
for f in feats:
    rr = within(f, 'te'); rn = [np.median(within(f, f'null{j}')) for j in range(3)]; rp = within(f, 'resp')
    R.append(dict(feature=f, real=np.median(rr), same_sign=int((np.sign(rr) == np.sign(np.median(rr))).sum()), partial=partial(f, 'te') if f not in ('log_size', 'density') else np.nan,
                  null_abs=np.mean(np.abs(rn)), resp=np.median(rp), resp_same_sign=int((np.sign(rp) == np.sign(np.median(rp))).sum())))
R = pd.DataFrame(R); R['real_minus_null'] = R.real.abs() - R.null_abs
R = R.sort_values('real', key=np.abs, ascending=False); R.to_csv(OUT / 'bad_te_background.csv', index=False)
pd.set_option('display.width', 250)
P('\nreal = within-section Spearman with the Itgb6+ TE (median of 4 sections) | same_sign of 4 | partial on size+density')
P('null_abs = mean |rho| with the shuffled-s TE | resp = Spearman with the independent responder rate\n')
P(R.head(25).round(3).to_string(index=False))
keep = R[(R.real.abs() >= 0.3) & (R.same_sign >= 3) & (R.real_minus_null >= 0.15) & (R.partial.abs().fillna(1) >= 0.2)]
P('\n== candidates: |real| >= 0.3, >= 3/4 sections agree, beats the null by >= 0.15, survives size/density (|partial| >= 0.2) ==')
P(keep.round(3).to_string(index=False) if len(keep) else '  none')
P(f'\nsanity: TE vs responder rate within section median {np.median(within("te", "resp")):+.2f}; real vs null TE (cell-mean) Spearman pooled {spearmanr(N.te, N[["null0","null1","null2"]].mean(1)).statistic:+.2f}')
P('done')
