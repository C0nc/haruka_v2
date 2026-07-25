"""Fast drop-in replacement for cellcharter's GMM-based ``cc.tl.Cluster``.

The default cellcharter clustering path runs a full-covariance Gaussian mixture
through a PyTorch-Lightning Trainer. On Haruka's real input shape
(``latent_dim=256`` x ``n_layer=3`` aggregation -> 1024 columns) that mixture has
~525k covariance parameters per component, is badly over-parameterised, and
terminates after ~2 EM iterations, so its full-covariance advantage never
materialises while it still pays the full O(n*d^2) cost.

Reducing dimensionality first is what actually fixes both speed and quality, so
``FastCluster`` standardises and (optionally) PCA-reduces the representation
before clustering. ``method='gmm'`` still delegates to ``cc.tl.Cluster`` so the
original behaviour stays reachable; ``FastCluster(k, method='gmm', scale=False,
n_pcs=None)`` reproduces it exactly.

All estimators mirror ``cc.tl.Cluster``'s API::

    model = FastCluster(n_clusters=8)
    model.fit(adata, use_rep='X_cellcharter')
    adata.obs['domain'] = model.predict(adata, use_rep='X_cellcharter')
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

__all__ = [
    "FastCluster",
    "cluster_adata",
    "select_k",
    "torch_kmeans",
]

_VAR_FLOOR = 1e-6


def _as_dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))


def _get_rep(adata, use_rep):
    X = adata.X if use_rep is None else adata.obsm[use_rep]
    return _as_dense(X)


def _resolve_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pairwise_sqdist(X, C, chunk):
    """||x-c||^2 in chunks over rows of X. Returns (n, k) on X's device."""
    c_sq = (C * C).sum(1)
    out = torch.empty((X.shape[0], C.shape[0]), device=X.device, dtype=X.dtype)
    for i in range(0, X.shape[0], chunk):
        xb = X[i : i + chunk]
        d = (xb * xb).sum(1, keepdim=True) - 2.0 * (xb @ C.T) + c_sq.unsqueeze(0)
        out[i : i + chunk] = d.clamp_min_(0)
    return out


def _kmeans_plusplus(X, k, generator, chunk):
    n = X.shape[0]
    centers = torch.empty((k, X.shape[1]), device=X.device, dtype=X.dtype)
    first = torch.randint(n, (1,), generator=generator, device=X.device)
    centers[0] = X[first[0]]
    closest = _pairwise_sqdist(X, centers[0:1], chunk).squeeze(1)
    for j in range(1, k):
        total = float(closest.sum())
        if not np.isfinite(total) or total <= 0:
            # All points already coincide with a center; pick uniformly.
            pick = torch.randint(n, (1,), generator=generator, device=X.device)[0]
        else:
            probs = closest / closest.sum()
            pick = torch.multinomial(probs, 1, generator=generator)[0]
        centers[j] = X[pick]
        d = _pairwise_sqdist(X, centers[j : j + 1], chunk).squeeze(1)
        closest = torch.minimum(closest, d)
    return centers


def _accumulate(X, labels, k, chunk):
    """Per-cluster sums via one-hot matmul.

    ``index_add_`` would be the obvious choice but it uses atomics on CUDA, so
    the float summation order varies run to run. The resulting last-bit drift in
    inertia is enough to change which n_init restart wins, which silently breaks
    reproducibility. GEMM is deterministic for fixed shapes, so use that instead.
    """
    sums = torch.zeros((k, X.shape[1]), device=X.device, dtype=X.dtype)
    for i in range(0, X.shape[0], chunk):
        lb = labels[i : i + chunk]
        oh = torch.zeros((lb.shape[0], k), device=X.device, dtype=X.dtype)
        oh.scatter_(1, lb.unsqueeze(1), 1.0)
        sums += oh.T @ X[i : i + chunk]
    return sums


def _lloyd(X, centers, max_iter, tol_abs, chunk):
    k = centers.shape[0]
    labels = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for _ in range(max_iter):
        d = _pairwise_sqdist(X, centers, chunk)
        labels = d.argmin(1)
        new = _accumulate(X, labels, k, chunk)
        counts = torch.bincount(labels, minlength=k).to(X.dtype)
        empty = counts == 0
        if bool(empty.any()):
            # Re-seed empty clusters at the points furthest from their center,
            # otherwise the division below yields NaN and poisons every later step.
            far = d.gather(1, labels.unsqueeze(1)).squeeze(1).argsort(descending=True)
            for slot, idx in zip(torch.nonzero(empty).squeeze(1).tolist(), far.tolist()):
                new[slot] = X[idx]
                counts[slot] = 1.0
        new = new / counts.unsqueeze(1)
        shift = float(((new - centers) ** 2).sum())
        centers = new
        if shift <= tol_abs:
            break
    d = _pairwise_sqdist(X, centers, chunk)
    labels = d.argmin(1)
    inertia = float(d.gather(1, labels.unsqueeze(1)).sum())
    return centers, labels, inertia


def torch_kmeans(
    X,
    n_clusters,
    random_state=42,
    n_init=10,
    max_iter=300,
    tol=1e-4,
    device=None,
    chunk=100_000,
):
    """k-means++ / Lloyd on torch, chunked so large n does not OOM.

    ``tol`` follows sklearn's convention: it is scaled by the mean per-feature
    variance of the data and compared against the squared center shift.
    """
    device = device if device is not None else _resolve_device(True)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    if n_clusters > n:
        raise ValueError(f"n_clusters={n_clusters} exceeds n_samples={n}")

    tol_abs = float(Xt.var(0, unbiased=False).mean()) * tol

    best = None
    for run in range(n_init):
        g = torch.Generator(device=device)
        g.manual_seed(int(random_state) + 1000 * run)
        c0 = _kmeans_plusplus(Xt, n_clusters, g, chunk)
        centers, labels, inertia = _lloyd(Xt, c0, max_iter, tol_abs, chunk)
        if best is None or inertia < best[2]:
            best = (centers, labels, inertia)

    centers, labels, inertia = best
    return centers.cpu().numpy(), labels.cpu().numpy().astype(np.int64), inertia


class FastCluster:
    """Drop-in, faster stand-in for :class:`cellcharter.tl.Cluster`.

    Parameters
    ----------
    n_clusters
        Number of clusters.
    method
        ``'kmeans'`` (GPU torch k-means++ when CUDA is available, else sklearn),
        ``'minibatch_kmeans'``, or ``'gmm'`` (delegates to ``cc.tl.Cluster``).
    random_state
        Seed. Unlike ``cc.tl.Cluster``, this is actually honoured: torchgmm
        ignores its own ``random_state``, so the ``'gmm'`` path seeds the global
        torch RNG via ``pl.seed_everything`` immediately before fitting.
    use_gpu
        Use CUDA when available. Falls back to CPU silently otherwise.
    scale
        Standardise columns before clustering. Neighbour aggregation leaves
        hop-0 (the cell's own features) carrying ~72% of total variance, which
        would otherwise dominate the Euclidean distances k-means uses.
    n_pcs
        PCA dimensionality applied after scaling. ``None`` disables it.
    """

    def __init__(
        self,
        n_clusters=1,
        method="kmeans",
        random_state=42,
        use_gpu=True,
        scale=True,
        n_pcs=50,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        chunk=100_000,
        trainer_params=None,
        **kwargs,
    ):
        if method not in ("kmeans", "minibatch_kmeans", "gmm"):
            raise ValueError(f"unknown method {method!r}")
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.scale = scale
        self.n_pcs = n_pcs
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.chunk = chunk
        self.trainer_params = trainer_params
        self.kwargs = kwargs

        self._mean = None
        self._std = None
        self._pca = None
        self._estimator = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.labels_ = None

    # -- preprocessing -----------------------------------------------------

    def _fit_transform(self, X):
        if self.scale:
            self._mean = X.mean(0)
            std = X.std(0)
            # A 256-d latent commonly carries dead units; scaling a near-constant
            # column would amplify pure noise, so hold those at unit scale.
            std[std < _VAR_FLOOR] = 1.0
            self._std = std
            X = (X - self._mean) / self._std
        if self.n_pcs and self.n_pcs < X.shape[1]:
            from sklearn.decomposition import PCA

            self._pca = PCA(
                n_components=min(self.n_pcs, min(X.shape) - 1),
                random_state=self.random_state,
            )
            X = self._pca.fit_transform(X)
        return np.ascontiguousarray(X, dtype=np.float32)

    def _transform(self, X):
        if self.scale:
            if self._mean is None:
                raise RuntimeError("call fit() before predict()")
            X = (X - self._mean) / self._std
        if self._pca is not None:
            X = self._pca.transform(X)
        return np.ascontiguousarray(X, dtype=np.float32)

    # -- API ---------------------------------------------------------------

    def fit(self, adata, use_rep="X_cellcharter"):
        X = self._fit_transform(_get_rep(adata, use_rep))
        device = _resolve_device(self.use_gpu)

        if self.method == "kmeans":
            if device.type == "cuda":
                centers, labels, inertia = torch_kmeans(
                    X,
                    self.n_clusters,
                    random_state=self.random_state,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    device=device,
                    chunk=self.chunk,
                )
                self.cluster_centers_, self.labels_, self.inertia_ = centers, labels, inertia
            else:
                from sklearn.cluster import KMeans

                est = KMeans(
                    n_clusters=self.n_clusters,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                ).fit(X)
                self._estimator = est
                self.cluster_centers_ = est.cluster_centers_
                self.labels_ = est.labels_.astype(np.int64)
                self.inertia_ = float(est.inertia_)

        elif self.method == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans

            est = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                batch_size=4096,
                random_state=self.random_state,
            ).fit(X)
            self._estimator = est
            self.cluster_centers_ = est.cluster_centers_
            self.labels_ = est.labels_.astype(np.int64)
            self.inertia_ = float(est.inertia_)

        else:  # gmm
            import anndata as ad
            import cellcharter as cc
            import pytorch_lightning as pl

            tp = self.trainer_params or dict(
                accelerator="gpu" if device.type == "cuda" else "cpu",
                devices=1,
            )
            est = cc.tl.Cluster(
                n_clusters=self.n_clusters,
                trainer_params=tp,
                random_state=self.random_state,
                **self.kwargs,
            )
            tmp = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
            tmp.obsm["_r"] = X
            # torchgmm never reads random_state; the global torch RNG is the only
            # thing that makes this reproducible, so seed immediately before fit.
            pl.seed_everything(self.random_state, workers=True)
            est.fit(tmp, use_rep="_r")
            self._estimator = est
            self.labels_ = np.asarray(est.predict(tmp, use_rep="_r")).astype(np.int64)
            if "_cellcharter" in tmp.uns:
                adata.uns["_cellcharter"] = tmp.uns["_cellcharter"]

        return self

    def predict(self, adata, use_rep="X_cellcharter"):
        X = self._transform(_get_rep(adata, use_rep))

        if self.method == "gmm":
            import anndata as ad

            tmp = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
            tmp.obsm["_r"] = X
            labels = np.asarray(self._estimator.predict(tmp, use_rep="_r"))
        elif self._estimator is not None:
            labels = self._estimator.predict(X)
        else:
            if self.cluster_centers_ is None:
                raise RuntimeError("call fit() before predict()")
            device = _resolve_device(self.use_gpu)
            Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
            Ct = torch.as_tensor(self.cluster_centers_, dtype=torch.float32, device=device)
            labels = _pairwise_sqdist(Xt, Ct, self.chunk).argmin(1).cpu().numpy()

        # Must match cc.tl.Cluster.predict exactly (_gmm.py:320): a pandas
        # Categorical whose categories are np.arange(n_clusters), so that empty
        # clusters still register and squidpy picks a discrete palette.
        return pd.Categorical(np.asarray(labels).astype(np.int64), categories=np.arange(self.n_clusters))

    def fit_predict(self, adata, use_rep="X_cellcharter"):
        self.fit(adata, use_rep=use_rep)
        return pd.Categorical(np.asarray(self.labels_).astype(np.int64), categories=np.arange(self.n_clusters))


def select_k(
    adata,
    use_rep="X_cellcharter",
    k_range=range(2, 15),
    method="silhouette",
    max_cells=10_000,
    random_state=42,
    **cluster_kwargs,
):
    """Pick the number of clusters.

    Returns ``(best_k, scores)`` where ``scores`` maps k -> score. Silhouette is
    O(n^2), so the score is always computed on a subsample of at most
    ``max_cells`` cells; the clustering itself is fitted on that same subsample
    to keep the sweep cheap.
    """
    if method not in ("silhouette", "elbow"):
        raise ValueError(f"unknown method {method!r}")

    import anndata as ad
    from sklearn.metrics import silhouette_score

    X = _get_rep(adata, use_rep)
    rng = np.random.default_rng(random_state)
    if X.shape[0] > max_cells:
        sub = rng.choice(X.shape[0], max_cells, replace=False)
        X = X[np.sort(sub)]

    tmp = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
    tmp.obsm["_r"] = X

    ks = list(k_range)
    scores = {}
    for k in ks:
        model = FastCluster(n_clusters=k, random_state=random_state, **cluster_kwargs)
        labels = np.asarray(model.fit_predict(tmp, use_rep="_r"))
        if method == "silhouette":
            if len(np.unique(labels)) < 2:
                scores[k] = -1.0
            else:
                scores[k] = float(silhouette_score(model._transform(X), labels))
        else:
            scores[k] = float(model.inertia_)

    if method == "silhouette":
        best_k = max(scores, key=scores.get)
    else:
        # Knee = point furthest from the chord joining the first and last
        # (k, inertia) points, after rescaling both axes to [0, 1].
        kk = np.asarray(ks, dtype=np.float64)
        vv = np.asarray([scores[k] for k in ks], dtype=np.float64)
        if len(ks) < 3:
            best_k = ks[0]
        else:
            kn = (kk - kk.min()) / max(kk.ptp(), 1e-12)
            vn = (vv - vv.min()) / max(vv.ptp(), 1e-12)
            p0 = np.array([kn[0], vn[0]])
            p1 = np.array([kn[-1], vn[-1]])
            d = p1 - p0
            d = d / max(np.linalg.norm(d), 1e-12)
            pts = np.stack([kn, vn], 1) - p0
            dist = np.abs(pts[:, 0] * d[1] - pts[:, 1] * d[0])
            best_k = ks[int(dist.argmax())]

    return best_k, scores


def cluster_adata(
    adata,
    use_rep="X_cellcharter",
    key_added="cluster",
    n_clusters=8,
    method="kmeans",
    random_state=42,
    **kwargs,
):
    """Fit and write labels into ``adata.obs[key_added]`` as a Categorical."""
    model = FastCluster(
        n_clusters=n_clusters, method=method, random_state=random_state, **kwargs
    )
    adata.obs[key_added] = model.fit_predict(adata, use_rep=use_rep)
    return model
