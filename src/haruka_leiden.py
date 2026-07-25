"""Spatially-constrained Leiden clustering for Haruka representations (RNA).

Standard Leiden clusters a kNN graph built in expression/embedding space alone, so a domain
can be split across the tissue or a cluster can be spatially incoherent. Here the graph is a
convex blend of two kNN graphs -- one in embedding space, one in physical space --

    A = (1 - w) * A_embedding  +  w * A_spatial

so ``spatial_weight=0`` reproduces ordinary Leiden and ``spatial_weight=1`` clusters on
geometry alone. Spatial edges are built per slice, so a domain never bridges sections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

__all__ = ["spatial_leiden", "leiden_adata", "sweep_resolution"]


def _knn(X, k, seed=0):
    A = kneighbors_graph(X, n_neighbors=min(k, len(X) - 1), mode="connectivity",
                         include_self=False)
    return ((A + A.T) > 0).astype(np.float64)          # symmetrise; unweighted


def _spatial_knn(coords, k, groups=None):
    """kNN on coordinates, built WITHIN each group so no edge crosses a slice."""
    n = len(coords)
    if groups is None:
        return _knn(coords, k)
    A = sp.lil_matrix((n, n))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        Ag = _knn(coords[idx], k).tocoo()
        A[idx[Ag.row], idx[Ag.col]] = 1.0
    return A.tocsr()


def spatial_leiden(
    X,
    coords,
    groups=None,
    spatial_weight: float = 0.5,
    k_embed: int = 15,
    k_spatial: int = 8,
    resolution: float = 1.0,
    seed: int = 0,
    scale: bool = True,
    n_pcs: int | None = 50,
):
    """Return integer labels from Leiden on a blended embedding+spatial graph."""
    import igraph as ig
    import leidenalg

    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    if scale:
        X = StandardScaler().fit_transform(X)
    if n_pcs and n_pcs < X.shape[1]:
        from sklearn.decomposition import PCA
        X = PCA(n_components=n_pcs, random_state=seed).fit_transform(X)

    w = float(np.clip(spatial_weight, 0.0, 1.0))
    A = (1.0 - w) * _knn(X, k_embed)
    if w > 0:
        A = A + w * _spatial_knn(np.asarray(coords, dtype=np.float64), k_spatial, groups)
    A = sp.triu(A.tocsr(), k=1).tocoo()

    g = ig.Graph(n=A.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())), directed=False)
    g.es["weight"] = A.data.tolist()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(part.membership, dtype=int)


def leiden_adata(adata, use_rep, key_added, spatial_key="spatial", library_key="slice_id",
                 **kw):
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.shape[1] > 2:          # SPAC-style [in_tissue, row, col]; drop the constant col
        coords = coords[:, -2:]
    groups = adata.obs[library_key].astype(str).values if library_key in adata.obs else None
    lab = spatial_leiden(adata.obsm[use_rep], coords, groups=groups, **kw)
    adata.obs[key_added] = pd.Categorical(lab, categories=np.arange(lab.max() + 1))
    return adata.obs[key_added]


def sweep_resolution(adata, use_rep, resolutions=(0.2, 0.4, 0.6, 0.8, 1.0, 1.5), **kw):
    """Resolution -> n_clusters, for picking a resolution without guessing."""
    out = {}
    for r in resolutions:
        lab = leiden_adata(adata, use_rep, "_tmp_leiden", resolution=r, **kw)
        out[r] = int(np.asarray(lab).astype(int).max() + 1)
    if "_tmp_leiden" in adata.obs:
        del adata.obs["_tmp_leiden"]
    return out
