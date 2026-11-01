#!/usr/bin/env python3
"""Diagnostic stack for the whole-slice CODEX AD fits (same tests as evaluate_channelnorm_ab.py).

  1. batch     sample NMI, same-donor kNN excess, eta^2 for z, s, z-CC, s-CC
  2. mechanism is the CC donor structure an amplified mean offset? (real / centred / random graph)
  3. cell type do z / s / CC views carry the major cell type? (kNN purity excess)
  4. anatomy   per-donor GM/WM AUROC of z-CC and s-CC PC1
  5. plaque    donor-held-out AUROC for cells <= 10 / 30 um of a dense Abeta centroid,
               all cells and WITHIN grey matter only (removes the GM/WM shortcut),
               plus per-cell-type within-GM AUROC for s vs z
Plaque distances are centroid distances (official dense Abeta objects), so there is no
'inside plaque' (0 um) class here.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/yancui/haruka_v2/bench_codex_cns_2026')
from evaluate_channelnorm_ab import km_nmi, knn_excess, eta2, aggregate, center_within, SEED
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/yancui/haruka_v2/results_codex_wholeslice')
REPS = {'z intrinsic': 'background', 's intrinsic': 'salient', 'z-CC background': 'spatial', 's-CC salient': 'salient_spatial'}


def held_out_auroc(X, y, donor, mask):
    sc = []
    for d in np.unique(donor[mask]):
        tr = mask & (donor != d); te = mask & (donor == d)
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2 or y[te].sum() < 20: continue
        xs = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, random_state=SEED, class_weight='balanced') \
            .fit(xs.transform(X[tr]), y[tr])
        sc.append(roc_auc_score(y[te], clf.predict_proba(xs.transform(X[te]))[:, 1]))
    return (np.mean(sc), len(sc)) if sc else (np.nan, 0)


def run(variant):
    data = Path(__file__).resolve().parent / f'codex_wholeslice_{variant}.h5ad'
    cc = ROOT / f'codex_wholeslice_{variant}' / 'haruka_z_s_cellcharter_cc3_condition_only'
    a = ad.read_h5ad(data); src = np.load(cc / 'source_rows.npy'); obs = a.obs.iloc[src]
    X = {k: np.load(cc / f'{v}.npy').astype(np.float32) for k, v in REPS.items()}
    donor = obs.donor_id.astype(str).to_numpy(); region = obs.region.astype(str).to_numpy()
    ct = obs.cell_type.astype(str).to_numpy(); xy = a.obsm['spatial'][src].astype(np.float64)
    dist = obs.dist_dense_abeta_um.to_numpy(float)
    print(f'\n{"=" * 78}\n{variant}: n={len(donor)} disease cells, donors {sorted(set(donor))}\n'
          f'cell types {pd.Series(ct).value_counts().to_dict()}\n{"=" * 78}')
    print('\n-- 1. batch --')
    print(f'  {"view":18s} {"sample NMI":>11s} {"kNN excess":>11s} {"eta^2":>8s}')
    for k, x in X.items():
        print(f'  {k:18s} {km_nmi(x, donor):11.3f} {knn_excess(x, donor):11.3f} {eta2(x, donor):8.3f}')
    print('\n-- 2. mechanism (z-CC donor NMI by graph) --')
    rng = np.random.default_rng(SEED); z = X['z intrinsic']
    for lbl, xx in [('real spatial graph', aggregate(z, xy, donor)),
                    ('donor-centred, real graph', aggregate(center_within(z, donor), xy, donor)),
                    ('RANDOM within-donor graph', aggregate(z, xy, donor, random=True, rng=rng))]:
        print(f'  {lbl:30s} donor NMI {km_nmi(xx, donor):.3f}')
    print('\n-- 3. cell type (kNN purity excess; 1 = neighbours always same type) --')
    for k, x in X.items():
        print(f'  {k:18s} cell-type kNN excess {knn_excess(x, ct):.3f}  NMI {km_nmi(x, ct):.3f}')
    print('\n-- 4. anatomy (per-donor WM AUROC of PC1; 0.5 = none, far from 0.5 either way = encodes GM/WM) --')
    for k in ['z-CC background', 's-CC salient']:
        pc1 = PCA(n_components=1, random_state=SEED).fit_transform(StandardScaler().fit_transform(X[k]))[:, 0]
        aucs = {d: roc_auc_score((region[donor == d] == 'WM').astype(int), pc1[donor == d])
                for d in np.unique(donor) if len(set(region[donor == d]) & {'GM', 'WM'}) == 2}
        print(f'  {k:18s} ' + ', '.join(f'{d} {v:.3f}' for d, v in aucs.items()))
    print('\n-- 5. plaque (donor-held-out AUROC) --')
    allm = np.isfinite(dist); gm = allm & (region == 'GM')
    for r in (10, 30):
        y = (dist <= r).astype(int)
        print(f'  <= {r} um: positives {y[allm].mean():.3%} of all, {y[gm].mean():.3%} of GM cells')
        for k, x in X.items():
            a1, n1 = held_out_auroc(x, y, donor, allm); a2, n2 = held_out_auroc(x, y, donor, gm)
            print(f'    {k:18s} all {a1:.3f} (n={n1})   GM only {a2:.3f} (n={n2})')
    y = (dist <= 30).astype(int)
    print('  <= 30 um within GM, per cell type (z intrinsic vs s intrinsic):')
    for t in sorted(set(ct)):
        m = gm & (ct == t)
        az, _ = held_out_auroc(X['z intrinsic'], y, donor, m); as_, n = held_out_auroc(X['s intrinsic'], y, donor, m)
        print(f'    {t:16s} n_cells {m.sum():6d}  z {az:.3f}  s {as_:.3f}  (donors {n})')


if __name__ == '__main__':
    for v in sys.argv[1:] or ['all7_no_oligo', 'full3']:
        run(v)
