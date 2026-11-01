#!/usr/bin/env python3
"""SPAC-seq lung M003, lesion-level deep dive: is Haruka's counterfactual composition shift aligned
with known biology?

Unit = lesion: connected component (<= 4 bins) of same-target eligible anchors, >= 5 anchors.
NT lesions are built the same way.  Per lesion (its context bins, from the best-quality fit):
  observed k=20 full-tissue neighbour composition, background p([z,0]), salient shift dp
Test per target vs NT lesions: Mann-Whitney on lesion means (n = lesions, not bins).
Biology readout (raw counts, all genes, context bins): immune-recruitment / activation genes
  T / CD8 effector, IFN-gamma response, DC, antigen presentation, and the target gene itself.
Prior expectations: Gdf15 loss -> more T-cell (CD8) infiltration (GDF-15 blocks T-cell recruitment);
Fzd5 loss -> Wnt/beta-catenin off -> more DC/T infiltration; Lat has no tumour-cell function.
"""
import os, sys, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')
H = Path('/home/yancui/haruka_v2'); B = H / 'bench_spacseq'; sys.path.insert(0, str(B))
import prepare_multislice_all_guides as base
SID = 'M003'; OUT = H / 'results_spac_best_quality/m003_deep'; OUT.mkdir(parents=True, exist_ok=True)
MD = H / 'results_spac_best_quality/spac_lung_best_quality_5targets_halo/haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
log = open(OUT / 'm003_deep.log', 'w')
def P(*x):
    s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()

a = ad.read_h5ad(B / 'data/spac_lung_best_quality_5targets_halo.h5ad', backed='r'); o = a.obs
TARGETS = list(a.uns['design']['targets']); cats = ['B', 'DC', 'Macrophage', 'Mast', 'NK', 'Neutrophil', 'Non-immune cell', 'T']
m = (o.slice_id.astype(str) == SID).to_numpy(); pt = o.perturbation.astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy()
xy = np.asarray(a.obsm['spatial'], float); anc = o.is_anchor.to_numpy(bool)
pbg = np.load(MD / 'background_composition.npy'); dp = np.load(MD / 'salient_composition_shift.npy')
mc = list(pd.read_json(MD / 'meta.json', typ='series')['cell_type_categories'])
pbg = pd.DataFrame(pbg, columns=mc)[cats].to_numpy(); dp = pd.DataFrame(dp, columns=mc)[cats].to_numpy()
# observed composition: k=20 within the slice; halo bins make this the real tissue neighbourhood
i = np.where(m)[0]; _, nb = cKDTree(xy[i]).query(xy[i], k=21)
code = pd.Categorical(ct[i], categories=cats).codes
obs = np.zeros((len(o), len(cats))); obs[i] = np.stack([(code[nb[:, 1:]] == j).mean(1) for j in range(len(cats))], 1)

# ---- lesions ----
lesion = np.full(len(o), -1); info = []
nid = 0
for t in ['NT'] + TARGETS:
    A_ = np.where(m & anc & (pt == t))[0]
    if len(A_) < 5: continue
    pr = cKDTree(xy[A_]).query_pairs(4.0, output_type='ndarray')
    adj = sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(A_), len(A_)))
    _, lab = connected_components(adj, directed=False)
    ctx = np.where(m & (pt == t))[0]; own = lab[cKDTree(xy[A_]).query(xy[ctx])[1]]      # context -> its nearest anchor's component
    for c in np.unique(lab):
        if (lab == c).sum() < 5: continue
        rows = ctx[own == c]; lesion[rows] = nid
        info.append(dict(lesion=nid, target=t, n_anchor=int((lab == c).sum()), n_bins=len(rows),
                         x=xy[rows, 0].mean(), y=xy[rows, 1].mean())); nid += 1
L = pd.DataFrame(info).set_index('lesion')
P(f'{SID}: lesions (>= 5 anchors within 4 bins) per target:', L.target.value_counts().to_dict())

# ---- per-lesion composition: observed / background / salient ----
for k, M_ in [('obs', obs), ('bg', pbg), ('sal', dp)]:
    for j, c in enumerate(cats):
        L[f'{k}_{c}'] = [M_[lesion == l, j].mean() for l in L.index]

# ---- biology genes (raw counts, all genes) ----
GENES = {'T': ['Cd3e', 'Cd3g', 'Cd8a', 'Cd8b1', 'Cd4'], 'CD8 effector': ['Gzmb', 'Gzma', 'Prf1', 'Ifng', 'Nkg7'],
         'IFN-gamma response': ['Cxcl9', 'Cxcl10', 'Stat1', 'Irf1', 'Gbp2', 'Ido1'],
         'Antigen presentation': ['B2m', 'H2-K1', 'H2-D1', 'Tap1', 'Psmb8', 'Psmb9', 'Cd74', 'H2-Aa', 'H2-Ab1'],
         'DC': ['Xcr1', 'Clec9a', 'Itgax', 'Ccr7', 'Flt3'], 'Exhaustion': ['Pdcd1', 'Havcr2', 'Lag3', 'Tox'],
         'Wnt target': ['Axin2', 'Lgr5', 'Nkd1', 'Sp5'], 'Myeloid suppressive': ['Arg1', 'Spp1', 'Chil3', 'Mrc1']}
d = base.load_slice(B / 'raw', SID)
rna = ad.read_h5ad(d['root'] / 'filtered_gene_bc_matrix.h5', backed='r')
names = np.array([n.split(':', 1)[1] for n in a.obs_names[m]]); rows_in = np.where(m)[0]
need = sorted({g for v in GENES.values() for g in v} | set(TARGETS))
vn = pd.Index(rna.var_names.astype(str)); have = [g for g in need if g in vn]
X = rna[names].to_memory(); tot = np.asarray(X.X.sum(1)).ravel().clip(1)
X.var_names_make_unique(); Xg = X[:, have].X; Xg = Xg.toarray() if sp.issparse(Xg) else np.asarray(Xg)
cp10k = np.log1p(Xg * 1e4 / tot[:, None]); E = pd.DataFrame(cp10k, columns=have)
P('genes missing from the panel:', sorted(set(need) - set(have)))
lesm = lesion[rows_in]
for prog, gs in GENES.items():
    gs = [g for g in gs if g in E]
    if not gs: continue
    score = E[gs].mean(1).to_numpy()
    L[f'prog_{prog}'] = [score[lesm == l].mean() for l in L.index]
for g in TARGETS:
    if g in E: L[f'gene_{g}'] = [E[g].to_numpy()[lesm == l].mean() for l in L.index]
L.to_csv(OUT / 'lesions_m003.csv')

# ---- tests: target lesions vs NT lesions ----
def test(col):
    nt = L.loc[L.target == 'NT', col]; res = {}
    for t in TARGETS:
        x = L.loc[L.target == t, col]
        if len(x) >= 2 and len(nt) >= 2:
            res[t] = (x.mean() - nt.mean(), mannwhitneyu(x, nt).pvalue, len(x))
    return res
P(f'\nNT lesions: {(L.target == "NT").sum()}')
for block, cols in [('observed neighbour composition', [f'obs_{c}' for c in ['T', 'DC', 'Macrophage', 'NK', 'Neutrophil', 'Non-immune cell']]),
                    ('background p([z,0])', [f'bg_{c}' for c in ['T', 'DC', 'Macrophage', 'Non-immune cell']]),
                    ('Haruka salient shift', [f'sal_{c}' for c in ['T', 'DC', 'Macrophage', 'NK', 'Neutrophil', 'Non-immune cell']]),
                    ('expression programmes (context bins)', [f'prog_{p}' for p in GENES if f'prog_{p}' in L]),
                    ('target gene itself (knock-down?)', [f'gene_{g}' for g in TARGETS if f'gene_{g}' in L])]:
    P(f'\n-- {block}: target - NT lesion mean (Mann-Whitney p, n lesions) --')
    for col in cols:
        r = test(col)
        P(f'   {col:34s}', '  '.join(f'{t} {v[0]:+.4f} (p {v[1]:.2f}, n {v[2]})' for t, v in r.items()))

# ---- within-lesion alignment: does the salient shift track observed immune infiltration / IFN response? ----
P('\n-- across all lesions (n = %d): Spearman of Haruka salient T-shift with --' % len(L))
from scipy.stats import spearmanr
for col in ['obs_T', 'obs_Non-immune cell', 'prog_T', 'prog_CD8 effector', 'prog_IFN-gamma response', 'prog_Antigen presentation', 'prog_DC']:
    if col in L:
        r, p = spearmanr(L['sal_T'], L[col]); r2, p2 = spearmanr(L['bg_T'], L[col])
        P(f'   {col:28s} salient rho {r:+.2f} (p {p:.3f}) | background rho {r2:+.2f} (p {p2:.3f})')
P('done')
