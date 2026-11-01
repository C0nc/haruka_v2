#!/usr/bin/env python3
"""Whole-slice CODEX AD dataset: every segmented lineage, harmonised 34-protein panel.

Replaces the myeloid-only input (IBA1 cells, 10 proteins, ~10% of the tissue).

  lineage      from the segmentation mask (GFAP / IBA1 / NeuN / Olig2), checked below
               against canonical markers
  dedup        instances from DIFFERENT masks within 3 px (1.1 um, inside one nucleus)
               are the same cell; keep the instance whose defining marker is stronger
  region       GM / WM from the official QuPath polygons
  plaque       distance to official dense / diffuse Abeta objects (AD donors only)
  A-Beta       kept as a per-cell READOUT, never a model input
  normalise    label-free per-donor channel normalisation (p25 background, p95 range),
               computed over the whole slice, then log1p(x*100) and global z (clip 10)

Two outputs, because Olig2 was segmented in only 3 donors:
  all7_no_oligo   7 donors, oligodendrocyte instances dropped (composition comparable)
  full3           AD-1, AD-4, Ctl-1 with all four lineages
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from matplotlib.path import Path as MplPath
from shapely.geometry import shape
from sklearn.neighbors import NearestNeighbors
from load_lineages import load, SHARED, READOUT, SRC

OUT = Path('/home/yancui/haruka_v2/bench_codex_cns_2026/wholeslice')
PX_UM = 0.3774
DEDUP_PX = 3.0
MASK_MARKER = {'Myeloid': 'IBA1', 'Astrocyte': 'GFAP', 'Neuron': 'NeuN'}   # Olig2 not in shared panel
PLAQUE = {'AD-1': '11162021_Bahareh-3155_EDTA_abeta.csv', 'AD-2': '10202021_BaharehNP_2997-ADCortex_EDTA_abeta.csv',
          'AD-3': '10122021_BaharehNP_ADCortex_EDTA_abeta.csv', 'AD-4': '12032021-Bahareh_3280_Alz_abeta.csv'}
MAJOR = {'Neuron': 'Neuron', 'Astrocyte': 'Astrocyte', 'Oligodendrocyte': 'Oligodendrocyte', 'Myeloid': 'Microglia'}


def dedup(A):
    """Drop the weaker of each cross-mask pair within DEDUP_PX of each other."""
    keep = np.ones(len(A), bool)
    # strength of each instance's own defining marker, z-scored within donor x lineage
    strength = np.zeros(len(A))
    for (don, lin), g in A.groupby(['donor_id', 'lineage_mask']):
        if lin in MASK_MARKER:
            v = np.log1p(g[MASK_MARKER[lin]].to_numpy(float))
            strength[g.index] = (v - v.mean()) / (v.std() + 1e-9)
        else:
            strength[g.index] = 0.0                       # Olig2 instances: neutral strength
    n_pairs = 0
    for don, g in A.groupby('donor_id'):
        idx = g.index.to_numpy(); xy = g[['x', 'y']].to_numpy(float); lin = g.lineage_mask.to_numpy()
        pairs = NearestNeighbors(radius=DEDUP_PX).fit(xy).radius_neighbors(xy, return_distance=False)
        for i, nb in enumerate(pairs):
            for j in nb:
                if j <= i or lin[i] == lin[j] or not (keep[idx[i]] and keep[idx[j]]): continue
                n_pairs += 1
                loser = idx[j] if strength[idx[i]] >= strength[idx[j]] else idx[i]
                keep[loser] = False
    return A[keep].reset_index(drop=True), n_pairs


def inside(geom, xy):
    if geom.geom_type == 'MultiPolygon':
        return np.logical_or.reduce([inside(g, xy) for g in geom.geoms])
    m = MplPath(np.asarray(geom.exterior.coords)).contains_points(xy)
    for ring in geom.interiors: m &= ~MplPath(np.asarray(ring.coords)).contains_points(xy)
    return m


def add_region(A):
    reg = np.full(len(A), 'Unassigned', dtype=object)
    for spec, g in A.groupby('specimen'):
        feats = json.loads((SRC / 'GM_WM_Annotations_QP' / f'{spec}_GM-WM Annotations.geojson').read_text())['features']
        xy = g[['x', 'y']].to_numpy(float); r = np.full(len(g), 'Unassigned', dtype=object)
        for f in feats:
            name = f['properties'].get('classification', {}).get('name', 'Unassigned')
            geom = shape(f['geometry']); geom = geom if geom.is_valid else geom.buffer(0)
            m = inside(geom, xy) & (r == 'Unassigned'); r[m] = name
        reg[g.index] = r
    A['region'] = pd.Series(reg).map({'Grey Matter': 'GM', 'White Matter': 'WM'}).fillna('Unassigned').to_numpy()
    return A


def add_plaque(A):
    A['dist_dense_abeta_um'] = np.nan; A['dist_diffuse_abeta_um'] = np.nan
    for don, f in PLAQUE.items():
        m = (A.donor_id == don).to_numpy()
        p = pd.read_csv(SRC / 'marker_intensities' / f, usecols=['class', 'x', 'y'])
        xy = A.loc[m, ['x', 'y']].to_numpy(float)
        for cls, col in [('dense', 'dist_dense_abeta_um'), ('diffuse', 'dist_diffuse_abeta_um')]:
            P = p.loc[p['class'].eq(cls), ['x', 'y']].to_numpy(float)
            if len(P): A.loc[m, col] = NearestNeighbors(n_neighbors=1).fit(P).kneighbors(xy)[0][:, 0] * PX_UM
    return A


M10 = ['IBA1', 'TMEM119', 'CD74', 'CD14', 'CD163', 'CD68', 'CD11b', 'CD11c', 'HLA-DR', 'CD45']


def add_myeloid_states(A):
    """Myeloid state for every IBA1 cell, transferred in MARKER space.

    The figure-7 myeloid cells come from a different segmentation run: after removing the
    ~0.6 px global offset only 13-39% sit within 2 px of a whole-slice IBA1 instance, so a
    coordinate lookup cannot label the slice.  Instead:
      1. anchors = mutual nearest neighbours <= 2 px (offset-corrected); their 10 myeloid
         marker intensities agree with figure-7 (median r 0.93-0.96), so they are the same cells
      2. a gradient-boosted classifier on per-donor z-scored log1p(10 markers) learns the
         map category from the anchors; leave-one-donor-out accuracy is reported
      3. anchors keep their own label; every other IBA1 cell gets the prediction + probability
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    src = ad.read_h5ad('/home/yancui/haruka_v2/bench_codex_cns_2026/data/processed/'
                       'codex_cns_myeloid_primary_no_aml_no_combat.h5ad', backed='r')
    so = src.obs[['donor_id', 'spatial_X', 'spatial_Y', 'paper_initial_leiden']].copy(); src.file.close()
    so['donor_id'] = so.donor_id.astype(str)
    ann = pd.read_csv('/home/yancui/haruka_v2/bench_codex_cns_2026/celltype_annotation/myeloid_annotation_table.csv',
                      dtype={'cluster': str}).set_index('cluster')
    my = np.where((A.lineage_mask == 'Myeloid').to_numpy())[0]
    don = A.donor_id.to_numpy()[my]
    F = np.log1p(A.iloc[my][M10].clip(lower=0).to_numpy(float))
    for d in np.unique(don):
        m = don == d; F[m] = (F[m] - F[m].mean(0)) / (F[m].std(0) + 1e-9)
    lab = np.full(len(my), '', dtype=object)
    for d in np.unique(don):
        i = np.where(don == d)[0]; s = so[so.donor_id == d]
        if not len(s): continue
        a = A.iloc[my[i]][['x', 'y']].to_numpy(float); b = s[['spatial_X', 'spatial_Y']].to_numpy(float)
        j = NearestNeighbors(n_neighbors=1).fit(a).kneighbors(b)[1][:, 0]
        dxy = a[j] - b; near = np.hypot(*dxy.T) < 40
        b = b + np.median(dxy[near], axis=0)                        # global offset (~0.6 px)
        da, ja = NearestNeighbors(n_neighbors=1).fit(b).kneighbors(a)
        jb = NearestNeighbors(n_neighbors=1).fit(a).kneighbors(b)[1][:, 0]
        mutual = (jb[ja[:, 0]] == np.arange(len(a))) & (da[:, 0] <= 2.0)
        c = s.paper_initial_leiden.astype(str).to_numpy()[ja[mutual, 0]]
        lab[i[mutual]] = ann.loc[c, 'map_category'].to_numpy()
    anc = lab != ''
    cv = np.empty(anc.sum(), dtype=object); da_ = don[anc]
    for d in np.unique(da_):
        te = da_ == d
        clf = HistGradientBoostingClassifier(max_iter=300, class_weight='balanced', random_state=0) \
            .fit(F[anc][~te], lab[anc][~te])
        cv[te] = clf.predict(F[anc][te])
    print(f'myeloid anchors (mutual NN <= 2 px): {anc.sum()}/{len(my)} IBA1 cells ({100 * anc.mean():.1f}%); '
          f'leave-one-donor-out acc {accuracy_score(lab[anc], cv):.3f}, '
          f'balanced acc {balanced_accuracy_score(lab[anc], cv):.3f}', flush=True)
    clf = HistGradientBoostingClassifier(max_iter=300, class_weight='balanced', random_state=0).fit(F[anc], lab[anc])
    P = clf.predict_proba(F)
    pred = clf.classes_[P.argmax(1)].astype(object)
    A['myeloid_state'] = ''; A['myeloid_state_prob'] = np.nan; A['myeloid_state_source'] = ''
    A.iloc[my, A.columns.get_loc('myeloid_state')] = np.where(anc, lab, pred)
    A.iloc[my, A.columns.get_loc('myeloid_state_prob')] = np.where(anc, 1.0, P.max(1))
    A.iloc[my, A.columns.get_loc('myeloid_state_source')] = np.where(anc, 'anchor', 'predicted')
    print(pd.crosstab(A.iloc[my].donor_id, A.iloc[my].myeloid_state).to_string(), flush=True)
    return A


def normalise(A):
    raw = A[SHARED].to_numpy(float); out = np.empty_like(raw)
    for don in A.donor_id.unique():
        m = (A.donor_id == don).to_numpy(); v = raw[m]
        bg, top = np.percentile(v, 25, axis=0), np.percentile(v, 95, axis=0)
        out[m] = np.clip(v - bg, 0, None) / (top - bg + 1e-9)
    X = np.log1p(out * 100)
    Z = np.clip((X - X.mean(0)) / (X.std(0) + 1e-9), -10, 10)
    return raw, out, Z.astype(np.float32)


def to_anndata(A, tag):
    raw, cn, Z = normalise(A)
    obs = A.drop(columns=SHARED).copy()
    obs['condition'] = np.where(obs.donor_id.str.startswith('AD'), 'disease', 'control')
    obs['cell_type'] = obs.lineage_mask.map(MAJOR)
    obs.index = [f'{d}_{l[:3]}_{c}' for d, l, c in zip(obs.donor_id, obs.lineage_mask, obs.cell)]
    assert obs.index.is_unique
    for c in ['donor_id', 'condition', 'region', 'cell_type', 'lineage_mask', 'myeloid_state', 'myeloid_state_source']:
        obs[c] = obs[c].astype(str).astype('category')
    a = ad.AnnData(X=Z, obs=obs, var=pd.DataFrame(index=SHARED))
    a.layers['raw_intensity'] = raw.astype(np.float32); a.layers['channel_normalized'] = cn.astype(np.float32)
    a.layers['protein'] = Z
    xy = A[['x', 'y']].to_numpy(np.float32)
    a.obsm['spatial'] = xy; a.obsm['spatial_um'] = xy * PX_UM
    a.uns['wholeslice'] = dict(panel=SHARED, readout=READOUT, px_um=PX_UM, dedup_px=DEDUP_PX, variant=tag,
                               transform='per-donor p25/p95 channel norm -> log1p(x*100) -> global z clip 10',
                               excluded='Ctl-2 (AML control)')
    p = OUT / f'codex_wholeslice_{tag}.h5ad'; a.write_h5ad(p, compression='gzip')
    print(f'wrote {p.name}: {a.n_obs} cells x {a.n_vars} proteins', flush=True)
    return a


if __name__ == '__main__':
    A = load(); n0 = len(A)
    A, n_pairs = dedup(A)
    print(f'dedup: {n_pairs} cross-mask pairs within {DEDUP_PX} px -> {n0 - len(A)} instances removed '
          f'({100 * (n0 - len(A)) / n0:.2f}%), {len(A)} cells kept', flush=True)
    A = add_region(A); A = add_plaque(A); A = add_myeloid_states(A)
    A.to_csv(OUT / 'wholeslice_all_cells.csv.gz', index=False, compression='gzip')
    print(pd.crosstab(A.donor_id, A.lineage_mask, margins=True).to_string(), flush=True)
    to_anndata(A[A.lineage_mask != 'Oligodendrocyte'].reset_index(drop=True), 'all7_no_oligo')
    to_anndata(A[A.donor_id.isin(['AD-1', 'AD-4', 'Ctl-1'])].reset_index(drop=True), 'full3')
