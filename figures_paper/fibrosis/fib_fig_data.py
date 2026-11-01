#!/usr/bin/env python3
"""Data for the lung-fibrosis lesion / gradient figure (make_fibrosis_figure.py).
Recomputes, exactly as bench_lung_fibrosis_gse250346/zs_stage/lesion_cases.py (same cells, same 15k subsample rule,
same centring and PCA, axis oriented + = towards the lesion), per case:
  per-cell axis values (z, s), distance, core  -> cases/{tag}.npz
  per-core means in distance bins               -> cases/{tag}_profile.csv
  Spearman of each decoded neighbour-type shift with the s axis -> cases/{tag}_shift.csv
Example core for the tissue maps = EXAMPLE_CORE (figure: TILD117MA1, the core whose alveolar-FB s and z r are both closest
to the pooled values; default without it: the focus-rich core with the strongest s r, which overstates the effect); all its cells saved with H&E region, lineage, distance to focus and the
alveolar-FB axis values -> example_core.npz.
"""
import os, json, warnings
for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(v, '8')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); OUT = H / 'figures_paper/fibrosis/data'; (OUT / 'cases').mkdir(parents=True, exist_ok=True)
MD = H / 'results_real_compgated_nb_final/gse250346_unaffected_vs_all_fibrosis/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
o = ad.read_h5ad(H / 'bench_lung_fibrosis_gse250346/data/gse250346_unaffected_vs_all_fibrosis.h5ad', backed='r').obs
types = json.load(open(MD / 'meta.json'))['cell_type_categories']
stage = o.sample_affect.astype(str).to_numpy(); core = o['sample'].astype(str).to_numpy(); ctA = o.cell_type.astype(str).to_numpy()
xy = np.c_[o.x_centroid.to_numpy(float), o.y_centroid.to_numpy(float)]
pos = pd.Series(np.arange(len(o)), index=o.index)
he = pd.read_csv('/project/zhihuanglab/yancui/haruka_fibrosis_he/he_annot_matched.csv.gz', usecols=['obs', 'annotation_type']); he['i'] = pos.reindex(he.obs).to_numpy()
Zm = np.load(MD / 'background.npy', mmap_mode='r'); Sm = np.load(MD / 'salient.npy', mmap_mode='r'); SH = np.load(MD / 'salient_composition_shift.npy', mmap_mode='r')
def centre(M, g):
    M = np.asarray(M, np.float64).copy()
    for k in pd.Series(np.arange(len(g))).groupby(g).indices.values(): M[k] -= M[k].mean(0)
    return M
def per_core_r(ax, d, cc):
    return {q: (np.corrcoef(ax[cc == q], d[cc == q])[0, 1], (cc == q).sum()) for q in np.unique(cc) if (cc == q).sum() >= 50 and ax[cc == q].std() > 0}
smean = lambda rr: np.average([v[0] for v in rr.values()], weights=[v[1] for v in rr.values()])
CASES = [('TLS', 'CD4+ T-cells', 0), ('fibroblastic_focus', 'Alveolar FBs', 0), ('microscopic_honeycombing', 'Alveolar FBs', 0),
         ('microscopic_honeycombing', 'Interstitial Macrophages', 0), ('fibroblastic_focus', 'Capillary', 1)]
fibcores = set(core[stage != 'Unaffected']); bins = [0, 25, 50, 100, 200, 400]; keep_fit = {}
for L, T, pcj in CASES:
    tag = f'{T.split()[0].replace("+", "")}_{L}'
    li = he.i[he.annotation_type == L].to_numpy(); ii, dd = [], []
    for c in sorted(fibcores):
        lc = li[core[li] == c]
        if len(lc) < 30: continue
        ci = np.flatnonzero(core == c); ci = ci[~np.isin(ci, lc)]
        d, _ = cKDTree(xy[lc]).query(xy[ci]); k = (d <= 400) & (ctA[ci] == T); ii.append(ci[k]); dd.append(d[k])
    ii = np.concatenate(ii); dd = np.concatenate(dd)
    if len(ii) > 15000:
        m = np.sort(np.random.default_rng(0).choice(len(ii), 15000, replace=False)); ii, dd = ii[m], dd[m]
    o_ = np.argsort(ii); ii, dd = ii[o_], dd[o_]; cc = core[ii]; ax = {}
    for k, M in [('z', Zm), ('s', Sm)]:
        X = centre(StandardScaler().fit_transform(np.asarray(M[ii])), cc); pca = PCA(2, random_state=0).fit(X); a_ = pca.transform(X)[:, pcj]
        sgn = -1.0 if smean(per_core_r(a_, dd, cc)) > 0 else 1.0; ax[k] = sgn * a_
        keep_fit[(tag, k)] = (pca, pcj, sgn)
    np.savez(OUT / 'cases' / f'{tag}.npz', ii=ii, dist=dd, core=cc, z=ax['z'], s=ax['s'], lesion=L, cell_type=T, axis=f'PC{pcj+1}')
    lab = np.digitize(dd, bins[1:-1])
    pd.concat({k: pd.DataFrame({'b': lab, 'c': cc, 'v': v}).groupby(['c', 'b']).v.mean().unstack() for k, v in ax.items()}).to_csv(OUT / 'cases' / f'{tag}_profile.csv')
    sh = np.asarray(SH[ii]); pd.DataFrame({'neighbour': types, 'rho': np.nan_to_num([spearmanr(sh[:, j], ax['s']).statistic for j in range(len(types))])}).to_csv(OUT / 'cases' / f'{tag}_shift.csv', index=False)
    print(tag, len(ii), 'cells', len(np.unique(cc)), 'cores', flush=True)

# example core for the maps: alveolar FB -> fibroblastic focus
z = np.load(OUT / 'cases' / 'Alveolar_fibroblastic_focus.npz', allow_pickle=True)
rr = per_core_r(z['s'], z['dist'], z['core'])
foc = he.i[he.annotation_type == 'fibroblastic_focus'].to_numpy(); nfoc = pd.Series(core[foc]).value_counts()
good = [q for q, (r, n) in rr.items() if r < 0]
ex = os.environ.get('EXAMPLE_CORE') or max(good, key=lambda q: (nfoc.get(q, 0) >= 60, rr[q][1] >= 600, -rr[q][0]))
# figure uses EXAMPLE_CORE=TILD117MA1: per-core alveolar-FB r closest to the pooled values for both s (-0.125 vs -0.128) and z (-0.014 vs -0.034)
print('example core', ex, 'focus cells', nfoc.get(ex), 'AFB r', round(rr[ex][0], 3), 'n', rr[ex][1])
ci = np.flatnonzero(core == ex)
reg = pd.Series('none', index=np.arange(len(o)))
pri = ['fibroblastic_focus', 'TLS', 'mixed_inflammation', 'remodeled_epithelium', 'microscopic_honeycombing', 'severe_fibrosis', 'fibrosis',
       'advanced_remodeling', 'minimally_remodeled_alveoli', 'remnant_alveoli']
hh = he[he.i.isin(ci)]
for r in reversed(pri): reg.loc[hh.i[hh.annotation_type == r].to_numpy()] = r   # highest priority written last
lc = foc[core[foc] == ex]; dfoc, _ = cKDTree(xy[lc]).query(xy[ci])
# alveolar-FB axis values for all alveolar FBs of this core within 400 um (same transform as the case fit)
afb = ci[(ctA[ci] == 'Alveolar FBs') & ~np.isin(ci, lc) & (dfoc <= 400)]
case = dict(zip(z['ii'].tolist(), range(len(z['ii']))))
vals = {}
for k in ['z', 's']:
    vals[k] = np.array([z[k][case[i]] if i in case else np.nan for i in afb])
np.savez(OUT / 'example_core.npz', core=ex, xy=xy[ci], cell_type=ctA[ci], lineage=o.lineage.astype(str).to_numpy()[ci], region=reg.iloc[ci].to_numpy(),
         dist_focus=dfoc, focus=np.isin(ci, lc), afb_xy=xy[afb], afb_z=vals['z'], afb_s=vals['s'], afb_dist=dfoc[np.isin(ci, afb)])
print('saved; alveolar FBs with axis values', np.isfinite(vals['s']).sum(), 'of', len(afb))
