"""Confound checks: (a) library size vs barcode status, (b) per-gRNA spatial clustering (clonal AAV spread),
(c) anatomical position of barcode+ cells, (d) neighbour-of-perturbed contamination by other perturbed cells.
Usage: python -I 05_confounds.py <annot_dir> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd, anndata as ad
from scipy import stats
from scipy.spatial import cKDTree

ind, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
CHIPS = ['C02943C3', 'B03018A2', 'A03599E2']
TYPES = ['ExN', 'InN', 'Astro', 'Oligo', 'OPC', 'Micro', 'Endo', 'ChP']
res = {}
rows = []
for c in CHIPS:
    a = ad.read_h5ad(os.path.join(ind, f'{c}.annot.h5ad'))
    cz = pd.read_csv(os.path.join(ind, f'{c}_cluster_marker_z.csv'), index_col=0)
    lab = cz['label'].where(cz[TYPES].max(1) >= 0.5, 'LowQ'); lab.index = lab.index.astype(str)
    obs = a.obs.copy()
    obs['cell_type'] = obs['leiden'].astype(str).map(lab).values
    xy = a.obsm['spatial'] * 0.5
    pos = obs.perturbation.values != 'none'
    # (a) library size
    lo = obs.umi_endo.values
    u = stats.mannwhitneyu(lo[pos], lo[~pos], alternative='two-sided')
    r = dict(chip=c, median_umi_bcpos=float(np.median(lo[pos])), median_umi_bcneg=float(np.median(lo[~pos])),
             p_umi=float(u.pvalue), frac_LowQ_bcpos=float((obs.cell_type.values[pos] == 'LowQ').mean()),
             frac_LowQ_all=float((obs.cell_type == 'LowQ').mean()))
    # logistic: P(barcode+) ~ log umi
    import statsmodels.api as sm
    X = sm.add_constant(np.log(np.maximum(lo, 1)))
    m = sm.GLM(pos.astype(float), X, family=sm.families.Binomial()).fit()
    r['logit_bcpos_on_log_umi'] = float(m.params[1]); r['p_logit'] = float(m.pvalues[1])
    # (c) spatial clustering of barcode+ vs matched random
    tree = cKDTree(xy)
    d_pos = cKDTree(xy[pos]).query(xy[pos], k=2)[0][:, 1]
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(20):
        s = rng.choice(len(xy), pos.sum(), replace=False)
        nulls.append(np.median(cKDTree(xy[s]).query(xy[s], k=2)[0][:, 1]))
    r['nn_bcpos_um'] = float(np.median(d_pos)); r['nn_random_um_mean'] = float(np.mean(nulls))
    # (d) neighbour contamination
    k = 20
    nn = tree.query(xy[pos], k=k + 1)[1][:, 1:]
    r['frac_perturbed_among_20nn'] = float(pos[nn].mean())
    res[c] = r
    # (b) per-gRNA clustering: NN distance within same KO vs to other KO cells (single-barcode only)
    sb = ~obs.perturbation.isin(['none', 'multi']).values
    P = obs.perturbation.values[sb]; XY = xy[sb]
    for ko in pd.unique(P):
        i = np.where(P == ko)[0]
        if len(i) < 10:
            continue
        d_same = cKDTree(XY[i]).query(XY[i], k=2)[0][:, 1]
        oth = np.where(P != ko)[0]
        d_oth = cKDTree(XY[oth]).query(XY[i], k=1)[0]
        rows.append(dict(chip=c, ko=ko, n=len(i), med_nn_same_ko_um=float(np.median(d_same)),
                         med_nn_other_ko_um=float(np.median(d_oth)),
                         p_same_lt_other=float(stats.wilcoxon(d_same, d_oth, alternative='less').pvalue)))
pd.DataFrame(rows).to_csv(os.path.join(out, 'per_grna_spatial_clustering.csv'), index=False)
json.dump(res, open(os.path.join(out, 'confounds.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
print(pd.DataFrame(rows).round(3).to_string(index=False))
