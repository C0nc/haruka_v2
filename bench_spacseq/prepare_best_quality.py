#!/usr/bin/env python3
"""Best-quality SPAC-seq lung subset with full-tissue niche targets.

* slices M001 + M003 only (M002 has 36 NT anchors -> no usable reference)
* targets with >= 2 lesions (>= 5 anchors within 4 bins) in BOTH slices:
  Acvr1b, Fzd5, Gdf15, Lat, Slc17a5 (+ NT); every eligible single-guide tumour anchor is kept
* contexts = anchor + its 8 nearest bins (same rule as prepare_multislice_all_guides.assign_contexts)
* halo = the 20 nearest full-tissue bins of every context bin that are not contexts themselves,
  condition 'halo': they are present only so the k=20 niche targets are computed on the real tissue;
  the fit runs with --exclude_label halo, so halo bins are not in either training arm
* genes = the 3000 HVGs of the original functional-guides dataset
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import prepare_multislice_all_guides as base
TARGETS = ['Acvr1b', 'Fzd5', 'Gdf15', 'Lat', 'Slc17a5']; SLICES = ['M001', 'M003']; K_CTX = 8; K_NICHE = 20
OUT = HERE / 'data/spac_lung_best_quality_5targets_halo.h5ad'
genes = list(ad.read_h5ad(HERE / 'data/spac_lung_functional_guides_k8_hvg3000.h5ad', backed='r').var_names)
pieces = []
for sid in SLICES:
    d = base.load_slice(HERE / 'raw', sid)
    feat = d['single_feature']; v = feat >= 0
    tgt = np.full(len(feat), '', object); tgt[v] = d['feature_target'][feat[v]]
    gname = np.full(len(feat), '', object); gname[v] = d['names'][feat[v]]
    elig = d['single'] & d['tumor']
    anchors = {t: np.flatnonzero(elig & (tgt == t)) for t in ['NT'] + TARGETS}
    groups = base.assign_contexts(d, anchors, K_CTX)
    ctx = np.concatenate([groups[t] for t in ['NT'] + TARGETS]); lab = np.concatenate([[t] * len(groups[t]) for t in ['NT'] + TARGETS])
    _, nb = cKDTree(d['coords']).query(d['coords'][ctx], k=K_NICHE + 1)
    halo = np.setdiff1d(np.unique(nb), ctx)
    rows = np.concatenate([ctx, halo]); pert = np.concatenate([lab, ['halo'] * len(halo)])
    names = d['guide'].obs_names[rows]
    rna = ad.read_h5ad(d['root'] / 'filtered_gene_bc_matrix.h5', backed='r'); piece = rna[names].to_memory(); rna.file.close()
    piece.var_names_make_unique(); piece = piece[:, genes].copy()
    anchor_set = np.concatenate(list(anchors.values()))
    piece.obs_names = pd.Index([f'{sid}:{x}' for x in names])
    piece.obs['slice_id'] = sid
    piece.obs['perturbation'] = pert
    piece.obs['condition'] = np.where(pert == 'NT', 'control', np.where(pert == 'halo', 'halo', 'disease'))
    piece.obs['guide_name'] = gname[rows]
    piece.obs['is_anchor'] = np.isin(rows, anchor_set)
    piece.obs['cell_type'] = d['cells'].reindex(names)['cell_type'].fillna('Unknown').astype(str).to_numpy()
    piece.obsm['spatial'] = d['coords'][rows]
    piece.layers['counts'] = piece.X.copy()
    pieces.append(piece)
    print(sid, 'contexts', pd.Series(lab).value_counts().to_dict(), '| halo', len(halo),
          '| anchors', {t: len(i) for t, i in anchors.items()}, flush=True)
A = ad.concat(pieces, merge='same'); A.obs['slice_id'] = A.obs.slice_id.astype('category')
for c in ['perturbation', 'condition', 'guide_name', 'cell_type']: A.obs[c] = A.obs[c].astype(str).astype('category')
A.uns['design'] = dict(targets=TARGETS, slices=SLICES, k_context=K_CTX, k_niche=K_NICHE, halo='condition==halo, neighbours only')
A.write_h5ad(OUT, compression='gzip'); print(OUT, A.shape, A.obs.condition.value_counts().to_dict())
