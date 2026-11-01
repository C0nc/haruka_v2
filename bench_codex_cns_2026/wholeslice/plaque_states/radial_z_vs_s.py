#!/usr/bin/env python3
"""Plaque-centred radial test of z vs s (intrinsic, not CC).

Cells: grey matter, 4 AD donors (+ Ctl-3 as the amyloid-positive control).
Features are centred within donor x cell type, and every test is run within cell type, so
neither donor offsets nor neuron/astro/microglia composition can create a gradient.
  A  eta^2 of distance bin (multivariate, standardised features)
  B  donor-held-out ridge decoding of log distance (cap 150 um): Spearman on the held-out donor
     and, with the AD-trained model, on Ctl-3
  C  dense vs diffuse: cells <= 20 um of a plaque, nearest plaque type, donor-held-out AUROC
  D  the decoded gradient axis of z and of s, correlated with the 34 proteins
"""
import warnings
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
SEED = 0; PX = 0.3774
M = ('/home/yancui/haruka_v2/results_codex_wholeslice/codex_wholeslice_all7_no_oligo/'
     'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0')
SRC = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/external/CODEX-CNS/codex_notebooks/marker_intensities')
FILES = {'AD-1': '11162021_Bahareh-3155_EDTA_abeta.csv', 'AD-2': '10202021_BaharehNP_2997-ADCortex_EDTA_abeta.csv',
         'AD-3': '10122021_BaharehNP_ADCortex_EDTA_abeta.csv', 'AD-4': '12032021-Bahareh_3280_Alz_abeta.csv',
         'Ctl-3': '10122021_BaharehNP_1-CtlCortex1796_EDTA_MKT6_abeta.csv'}
AD = ['AD-1', 'AD-2', 'AD-3', 'AD-4']
BINS = [0, 10, 20, 40, 60, 100, np.inf]; BL = ['0-10', '10-20', '20-40', '40-60', '60-100', '>100']

a = ad.read_h5ad('../codex_wholeslice_all7_no_oligo.h5ad'); o = a.obs
don = o.donor_id.astype(str).to_numpy(); ct = o.cell_type.astype(str).to_numpy(); xy = a.obsm['spatial'].astype(float)
dd = np.full(len(o), np.nan); df = np.full(len(o), np.nan)
for d, f in FILES.items():
    p = pd.read_csv(SRC / f); m = don == d
    for cls, out in [('dense', dd), ('diffuse', df)]:
        q = p.loc[p['class'] == cls, ['x', 'y']].to_numpy(float)
        out[m] = NearestNeighbors(n_neighbors=1).fit(q).kneighbors(xy[m])[0][:, 0] * PX
dist = np.fmin(dd, df); near_dense = (dd <= df).astype(int)
use = (o.region.to_numpy() == 'GM') & np.isin(don, AD + ['Ctl-3'])
X = {'z': np.load(f'{M}/background.npy'), 's': np.load(f'{M}/salient.npy'), 'protein': np.asarray(a.X)}
prot = list(a.var_names)
for k in X:                                        # centre within donor x cell type
    x = X[k].astype(np.float64).copy()
    for d in np.unique(don):
        for t in np.unique(ct):
            m = (don == d) & (ct == t); x[m] -= x[m].mean(0)
    X[k] = x
P = X['protein']
bins = pd.cut(dist, BINS, labels=BL, right=True)
ld = np.log1p(np.minimum(dist, 150))

def eta2(x, lab):
    xs = StandardScaler().fit_transform(x); gm = xs.mean(0)
    b = sum((lab == g).sum() * ((xs[lab == g].mean(0) - gm) ** 2).sum() for g in np.unique(lab))
    return b / ((xs - gm) ** 2).sum()

rows, axes = [], {}
for t in ['Neuron', 'Astrocyte', 'Microglia']:
    m = use & (ct == t); lab = np.asarray(bins)[m].astype(str)
    print(f'\n===== {t} (GM; AD n={np.isin(don[m], AD).sum()}, Ctl-3 n={(don[m] == "Ctl-3").sum()}) =====')
    print('  cells per bin (AD):', pd.Series(lab[np.isin(don[m], AD)]).value_counts().reindex(BL).to_dict())
    for k, x in X.items():
        xa = x[m & np.isin(don, AD)]; la = np.asarray(bins)[m & np.isin(don, AD)].astype(str)
        xc = x[m & (don == 'Ctl-3')]; lc = np.asarray(bins)[m & (don == 'Ctl-3')].astype(str)
        e_ad, e_ct = eta2(xa, la), eta2(xc, lc)
        rho = []
        for d in AD:                                                 # donor-held-out decoding
            tr = m & np.isin(don, AD) & (don != d); te = m & (don == d)
            sc = StandardScaler().fit(x[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[tr]), ld[tr])
            rho.append(spearmanr(r.predict(sc.transform(x[te])), ld[te]).statistic)
        tr = m & np.isin(don, AD); te = m & (don == 'Ctl-3')
        sc = StandardScaler().fit(x[tr]); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(x[tr]), ld[tr])
        rho_ctl = spearmanr(r.predict(sc.transform(x[te])), ld[te]).statistic
        proj = r.predict(sc.transform(x[m & np.isin(don, AD)]))
        axes[(t, k)] = pd.Series([spearmanr(proj, P[m & np.isin(don, AD), j]).statistic for j in range(len(prot))], index=prot)
        # dense vs diffuse among cells <= 20 um
        near = m & (dist <= 20); auc = []
        for d in AD:
            tr2 = near & np.isin(don, AD) & (don != d); te2 = near & (don == d)
            if len(np.unique(near_dense[te2])) < 2: continue
            sc2 = StandardScaler().fit(x[tr2])
            c = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED).fit(sc2.transform(x[tr2]), near_dense[tr2])
            auc.append(roc_auc_score(near_dense[te2], c.predict_proba(sc2.transform(x[te2]))[:, 1]))
        rows.append(dict(cell_type=t, rep=k, eta2_AD=e_ad, eta2_Ctl3=e_ct, rho_heldout_AD=np.mean(rho),
                         rho_AD_min=np.min(rho), rho_on_Ctl3=rho_ctl, dense_vs_diffuse_auroc=np.mean(auc)))
        print(f'  {k:8s} eta2 AD {e_ad:.4f} | Ctl-3 {e_ct:.4f}   distance rho held-out AD {np.mean(rho):+.3f} '
              f'(min {np.min(rho):+.3f}) | AD->Ctl-3 {rho_ctl:+.3f}   dense-vs-diffuse AUROC {np.mean(auc):.3f}')
R = pd.DataFrame(rows); R.to_csv('radial_z_vs_s_summary.csv', index=False)

print('\n===== D. what the distance axis means: top proteins correlated with each decoded axis (AD cells) =====')
print('   (axis predicts log distance, so NEGATIVE rho = higher near the plaque)')
for (t, k), sr in axes.items():
    if k == 'protein': continue
    sr = sr.sort_values()
    print(f'  {t:9s} {k}: near-plaque up ' + ', '.join(f'{g} {v:+.2f}' for g, v in sr[:6].items()) +
          ' | far up ' + ', '.join(f'{g} {v:+.2f}' for g, v in sr[::-1][:3].items()))
pd.DataFrame({f'{t}|{k}': v for (t, k), v in axes.items()}).to_csv('radial_axis_protein_rho.csv')

prof = pd.DataFrame(P[use & np.isin(don, AD)], columns=prot).assign(bin=np.asarray(bins)[use & np.isin(don, AD)].astype(str),
                                                                    ct=ct[use & np.isin(don, AD)])
pm = prof.groupby(['ct', 'bin'])[['ApoE', 'CD68', 'CD11c', 'HLA-DR', 'TMEM119', 'IBA1', 'GFAP', 'Vimentin', 'CD44', 'MAP2', 'Synaptophysin']].mean()
pm = pm.reindex(pd.MultiIndex.from_product([['Neuron', 'Astrocyte', 'Microglia'], BL]))
print('\n===== radial protein profile (AD, GM, centred within donor x type) ====='); print(pm.round(2).to_string())
pm.to_csv('radial_protein_profile.csv')
