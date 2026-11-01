"""Re-run the full CODEX AD diagnostic stack on both normalizations.

Reports, for each fit:
  1. batch    : sample NMI (as in the report) + same-donor kNN excess, all four views
  2. mechanism: 4-arm test -- is the CC donor structure just an amplified mean offset?
  3. anatomy  : does s-CC PC1 encode grey/white matter?
  4. biology  : within-donor s-CC PC1 x dense-Abeta rho, pooled and region-stratified
  5. headline : donor-held-out plaque AUROC, s vs z
"""
from __future__ import annotations
import argparse, json, warnings
import numpy as np, pandas as pd, anndata as ad
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as nmi, roc_auc_score
from scipy.stats import spearmanr

SEED = 0
PLQ = ('/home/yancui/haruka_v2/final_results_nb/codex_ad/biology/plaque_distance/'
       'codex_plaque_distance_cells.csv.gz')

def load(data, cc_dir):
    a = ad.read_h5ad(data)
    src = np.load(f'{cc_dir}/source_rows.npy')
    obs = a.obs.iloc[src]
    reps = {'z intrinsic': 'background', 's intrinsic': 'salient',
            'z-CC background': 'spatial', 's-CC salient': 'salient_spatial'}
    X = {k: np.load(f'{cc_dir}/{v}.npy').astype(np.float32) for k, v in reps.items()}
    return a, src, obs, X

def km_nmi(x, lab):
    xs = StandardScaler().fit_transform(x)
    return nmi(lab, MiniBatchKMeans(n_clusters=len(np.unique(lab)), n_init=20,
                                    batch_size=4096, random_state=SEED).fit_predict(xs))

def knn_excess(x, lab):
    xs = StandardScaler().fit_transform(x)
    nb = NearestNeighbors(n_neighbors=31).fit(xs).kneighbors(xs, return_distance=False)[:, 1:]
    e = float((pd.Series(lab).value_counts(normalize=True) ** 2).sum())
    return ((lab[nb] == lab[:, None]).mean() - e) / (1 - e)

def eta2(x, lab):
    xs = StandardScaler().fit_transform(x); gm = xs.mean(0)
    b = sum((lab == g).sum() * ((xs[lab == g].mean(0) - gm) ** 2).sum() for g in np.unique(lab))
    return b / ((xs - gm) ** 2).sum()

def aggregate(x, xy, donor, k=20, hops=3, random=False, rng=None):
    out, cur = [], x.copy()
    for _ in range(hops):
        nxt = np.zeros_like(x)
        for d in np.unique(donor):
            i = np.where(donor == d)[0]
            nb = (rng.integers(0, len(i), size=(len(i), k)) if random else
                  NearestNeighbors(n_neighbors=k+1).fit(xy[i]).kneighbors(xy[i], return_distance=False)[:, 1:])
            nxt[i] = cur[i][nb].mean(1)
        out.append(nxt); cur = nxt
    return np.hstack(out)

def center_within(x, lab):
    o = x.copy()
    for g in np.unique(lab):
        m = lab == g; o[m] -= o[m].mean(0)
    return o

def run(tag, data, cc_dir):
    a, src, obs, X = load(data, cc_dir)
    donor = obs['donor_id'].astype(str).to_numpy()
    region = obs['region'].astype(str).to_numpy()
    xy = a.obsm['spatial'][src].astype(np.float64)
    print(f'\n{"="*78}\n{tag}\n  {data}\n  {cc_dir}\n{"="*78}')
    print(f'  n={len(donor)} condition cells, donors={sorted(set(donor))}')

    print('\n-- 1. batch --')
    print(f'  {"view":18s} {"sample NMI":>11s} {"kNN excess":>11s} {"eta^2":>8s}')
    res = {}
    for k, x in X.items():
        res[k] = (km_nmi(x, donor), knn_excess(x, donor), eta2(x, donor))
        print(f'  {k:18s} {res[k][0]:11.3f} {res[k][1]:11.3f} {res[k][2]:8.3f}')

    print('\n-- 2. mechanism (is z-CC just an amplified mean offset?) --')
    rng = np.random.default_rng(SEED); z = X['z intrinsic']
    for lbl, xx in [('real spatial graph', aggregate(z, xy, donor)),
                    ('donor-centered, real graph', aggregate(center_within(z, donor), xy, donor)),
                    ('RANDOM within-donor graph', aggregate(z, xy, donor, random=True, rng=rng))]:
        print(f'  {lbl:30s} donor NMI {km_nmi(xx, donor):.3f}')

    print('\n-- 3. anatomy (does s-CC PC1 encode GM/WM?) --')
    for k in ['z-CC background', 's-CC salient']:
        pc1 = PCA(n_components=1, random_state=SEED).fit_transform(StandardScaler().fit_transform(X[k]))[:, 0]
        aucs = {d: roc_auc_score((region[donor == d] == 'WM').astype(int), pc1[donor == d])
                for d in np.unique(donor)}
        print(f'  {k:18s} per-donor region AUROC ' +
              ', '.join(f'{d} {v:.3f}' for d, v in aucs.items()))

    print('\n-- 4. biology (s-CC PC1 x dense-Abeta distance) --')
    plq = pd.read_csv(PLQ); plq['cell_id'] = plq['cell_id'].astype(str)
    dist = pd.Series(obs.index.astype(str)).map(
        plq.set_index('cell_id')['distance_to_dense_abeta_centroid_um']).to_numpy()
    pc1 = PCA(n_components=1, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(X['s-CC salient']))[:, 0]
    for d in np.unique(donor):
        m = donor == d
        gm = m & (region == 'GM'); wm = m & (region == 'WM')
        print(f'  {d}: pooled rho {spearmanr(pc1[m], dist[m]).statistic:+.3f} | '
              f'GM {spearmanr(pc1[gm], dist[gm]).statistic:+.3f} | '
              f'WM {spearmanr(pc1[wm], dist[wm]).statistic:+.3f}')

    print('\n-- 5. headline: donor-held-out plaque AUROC --')
    for radius in (0, 30):
        y = obs[f'plaque_within_{radius}um'].to_numpy().astype(int)
        ok = y >= 0
        for k in ['z intrinsic', 's intrinsic', 'z-CC background', 's-CC salient']:
            sc = []
            for d in np.unique(donor):
                tr = ok & (donor != d); te = ok & (donor == d)
                if len(np.unique(y[te])) < 2 or te.sum() < 200: continue
                xs = StandardScaler().fit(X[k][tr])
                clf = LogisticRegression(max_iter=3000, random_state=SEED,
                                         class_weight='balanced').fit(xs.transform(X[k][tr]), y[tr])
                sc.append(roc_auc_score(y[te], clf.predict_proba(xs.transform(X[k][te]))[:, 1]))
            print(f'  plaque<={radius:2d}um  {k:18s} mean donor-held-out AUROC {np.mean(sc):.3f} (n={len(sc)})')
    return res

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/yancui/haruka_v2/results_codex_channelnorm_ab')
    args = p.parse_args()
    P = '/home/yancui/haruka_v2/bench_codex_cns_2026/data/processed'
    M = 'haruka_shared_full_niche_compgated_decoderbatch8_batchmode-decoder_only_seed0'
    CC = 'haruka_z_s_cellcharter_cc3_condition_only'
    for tag, stem in [('A. BASELINE  log1p + global z', 'codex_cns_myeloid_primary_no_aml_no_combat'),
                      ('B. CHANNEL-NORM  p25/p95', 'codex_cns_myeloid_primary_no_aml_channelnorm')]:
        run(tag, f'{P}/{stem}.h5ad', f'{args.root}/{stem}/{CC}')
