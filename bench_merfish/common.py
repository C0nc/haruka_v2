"""Shared IO and the one clustering routine every method is scored through.

The point of putting the clusterer here rather than in each runner is that the
comparison is only meaningful if the downstream is IDENTICAL. Haruka's own
`cluster()` would route through `cellcharter.tl.Cluster` (torchgmm, GPU-only,
`random_state` not honoured); CellCharter's would too; CINEMA-OT ships none.
Scoring them with three different clusterers would confound the representation
with the partitioner. So every representation -- Haruka's, CellCharter's,
contrastiveVI's, CINEMA-OT's, plain PCA's -- goes through exactly this:

    z-score columns -> PCA(50) -> full-covariance GaussianMixture

which is the configuration the Haruka-v2 README settled on ("full-covariance GMM
on PCA-50 beats KMeans on the elongated background representation") reimplemented
in sklearn so it is deterministic and needs no GPU.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

N_PCS = 50


def embed(X: np.ndarray, n_pcs: int | None = N_PCS, seed: int = 0) -> np.ndarray:
    """z-score, then PCA to at most `n_pcs` components. `n_pcs=None` skips the PCA.

    Skipping matters for probes. PCA-50 of a neighbourhood-averaged expression
    matrix is almost entirely anatomy -- that is where the variance is -- and a
    fine-grained, low-variance factor such as the ligand-balance niche is
    truncated away by it. Probing the PCA-50 embedding then reports the factor as
    absent when it is merely off-axis: the niche scored AUROC 0.491 through PCA-50
    while ridge on the raw genes recovered it at R^2 0.30 on the same data.
    Clustering still goes through PCA-50 (that is the fixed protocol); presence-of-
    information probes should not.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X[:, np.var(X, axis=0) > 1e-12] if X.shape[1] > 1 else X
    if X.shape[1] == 0:
        return np.zeros((len(X), 1))
    X = StandardScaler().fit_transform(X)
    if n_pcs is None:
        return X
    k = min(n_pcs, X.shape[1], X.shape[0] - 1)
    if k < X.shape[1]:
        X = PCA(n_components=k, random_state=seed).fit_transform(X)
    return X


def gmm_on_embedding(Z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Full-covariance GMM on an already-embedded matrix."""
    gm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed,
        n_init=3, max_iter=300, reg_covar=1e-4,
    )
    return gm.fit_predict(Z)


def gmm_labels(X: np.ndarray, k: int, seed: int = 0, n_pcs: int = N_PCS) -> np.ndarray:
    """The benchmark's single clustering primitive: PCA-`n_pcs` + full-cov GMM."""
    return gmm_on_embedding(embed(X, n_pcs=n_pcs, seed=seed), k, seed)


def kmeans_on_embedding(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """K-means directly on the raw GAT readout (no PCA or post-hoc smoothing)."""
    Z = np.asarray(X, dtype=np.float64)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def leiden_labels(X: np.ndarray, resolution: float = 0.6, seed: int = 0,
                  n_neighbors: int = 15) -> np.ndarray:
    """SIMVI's clustering: kNN graph on the RAW latent, then Leiden.

    Verbatim from `simvi/model/simvi.py`::

        sc.pp.neighbors(adata_tmp, use_rep=z_label)
        sc.tl.leiden(adata_tmp, resolution=0.6)

    No PCA truncation and no Gaussian-component assumption. That matters here:
    on the architecture-matched build every latent carries its own factor
    linearly -- `salient_spatial` reads its domain at probe AUROC 0.858, `spatial`
    reads the niche at 0.916 -- while PCA-50 + full-covariance GMM clusters both
    at ARI ~0.000. The information is present and the Gaussian read-out cannot
    partition on it, so the clusterer is worth varying before concluding anything
    about the model.

    Leiden takes a resolution rather than a cluster count, so the number of
    communities is not forced to match the truth's class count. ARI handles that,
    but the count is returned by the caller for inspection: a 40-community
    partition scoring low against a 3-class truth means something different from
    a 3-community partition scoring low.
    """
    import anndata as _ad
    import scanpy as _sc

    tmp = _ad.AnnData(np.zeros((len(X), 1), dtype=np.float32))
    tmp.obsm["rep"] = np.asarray(X, dtype=np.float32)
    _sc.pp.neighbors(tmp, use_rep="rep", n_neighbors=n_neighbors, random_state=seed)
    _sc.tl.leiden(tmp, resolution=resolution, random_state=seed,
                  key_added="leiden", flavor="igraph", n_iterations=2,
                  directed=False)
    return tmp.obs["leiden"].to_numpy().astype(int)


def dependence(X, y, groups, discrete: bool, seed: int = 0) -> float:
    """Clusterer-free measure of how much a representation knows about a factor.

    WHY THIS REPLACES ARI AS THE PRIMARY METRIC. ARI presupposes the truth has
    clusters. Two of the four factors here are terciles of a CONTINUOUS score
    (residualised neighbourhood composition), and a tercile cut of a continuum has
    no natural modes -- so no partitioner can recover it. Measured: `spatial ->
    gt_niche` and `salient_spatial -> gt_sal_domain` score ~0.00 under BOTH
    PCA-50 + full-covariance GMM and SIMVI's Leiden, while linear probes on the
    same latents read 0.86-0.92. The information is plainly there; the metric
    could not see it, and every ~0.00 previously reported in those rows was an
    artefact of asking for clusters that do not exist.

    This measures prediction instead, always cross-validated leave-one-animal-out
    so nothing is credited to memorising a slice:

        discrete truth   -> 2 * (one-vs-rest macro AUROC - 0.5)
        continuous truth -> ridge R^2, clipped at 0

    Both land on [0, 1] with 0 = knows nothing, so diagonal and off-diagonal
    entries stay commensurate across factor types and the assignment margin
    remains meaningful. ARI is still reported for the discrete factors, where it
    is well posed.
    """
    if discrete:
        a = probe_auroc_multiclass(X, y, groups, seed=seed)
        return float("nan") if np.isnan(a) else max(0.0, 2.0 * (a - 0.5))

    from sklearn.linear_model import RidgeCV
    Z = embed(X, n_pcs=None, seed=seed)
    y = np.asarray(y, dtype=float)
    uniq = np.unique(groups)
    n_splits = min(len(uniq), 4)
    if n_splits < 2:
        return float("nan")
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Z, y, groups=groups):
        m = RidgeCV(alphas=np.logspace(-2, 3, 8)).fit(Z[tr], y[tr])
        scores.append(m.score(Z[te], y[te]))
    return float(max(0.0, np.mean(scores)))


def ari(labels, truth) -> float:
    return float(adjusted_rand_score(np.asarray(truth), np.asarray(labels)))


def nmi(labels, truth) -> float:
    return float(normalized_mutual_info_score(np.asarray(truth), np.asarray(labels)))


def probe_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0,
                n_pcs: int | None = None) -> float:
    """Cross-validated linear-probe AUROC, held out BY SLICE.

    Grouping by slice matters: cells within a slice share an animal, a section and
    a batch, so a random split would let the probe memorise slice identity and
    report a detection that does not generalise to a new animal. With 4 slices per
    arm this is a 4-fold leave-one-animal-out.

    This exists alongside the clustering metrics because clustering is a
    thresholded read-out -- at subtle effect sizes a representation can carry the
    domain linearly while no k-component GMM will split on it, and reporting only
    ARI would call that "no signal".
    """
    return probe_auroc_pre(embed(X, n_pcs=n_pcs, seed=seed), y, groups)


def probe_auroc_pre(Z: np.ndarray, y, groups) -> float:
    """`probe_auroc` on an already-embedded matrix."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    uniq = np.unique(groups)
    n_splits = min(len(uniq), 4)
    if n_splits < 2:
        return float("nan")
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Z, y, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Z[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.decision_function(Z[te])))
    return float(np.mean(scores)) if scores else float("nan")


def probe_auroc_multiclass(X, y, groups, seed: int = 0, n_pcs: int | None = None) -> float:
    """One-vs-rest macro AUROC over ALL classes, held out by slice.

    Replaces binarising a multi-class truth against its majority class, which is
    degenerate for an ORDERED label. `gt_niche` is terciles of a continuous score;
    with the three terciles exactly balanced the majority is decided by a tie-break
    on set iteration order, and when it lands on the MIDDLE tercile the target
    becomes "is in an extreme tercile" -- a U-shaped function of the score that no
    linear probe can separate. Measured on all three seeds:

        target != low tercile     AUROC 0.824 / 0.838 / 0.882
        target != MIDDLE tercile  AUROC 0.493 / 0.508 / 0.510   <- degenerate
        target != high tercile    AUROC 0.826 / 0.842 / 0.865

    The 0.50 row was read as "seed 1's niche factor is dead" for several rounds.
    It was an artefact of which class the tie-break happened to pick.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    if len(classes) == 2:
        return probe_auroc_pre(embed(X, n_pcs=n_pcs, seed=seed), y == classes[1], groups)
    Z = embed(X, n_pcs=n_pcs, seed=seed)
    n_splits = min(len(np.unique(groups)), 4)
    if n_splits < 2:
        return float("nan")
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Z, y, groups=groups):
        if len(np.unique(y[tr])) < len(classes) or len(np.unique(y[te])) < len(classes):
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Z[tr], y[tr])
        scores.append(roc_auc_score(y[te], clf.predict_proba(Z[te]),
                                    multi_class="ovr", average="macro"))
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# runner-side IO
# ---------------------------------------------------------------------------
def save_embeddings(out_dir: str, background=None, salient=None, spatial=None,
                    salient_spatial=None, obs_names=None, **meta):
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in (("background", background), ("salient", salient),
                      ("spatial", spatial), ("salient_spatial", salient_spatial)):
        if arr is not None:
            np.save(os.path.join(out_dir, f"{name}.npy"), np.asarray(arr, dtype=np.float32))
    if obs_names is not None:
        np.save(os.path.join(out_dir, "obs_names.npy"), np.asarray(obs_names, dtype=object),
                allow_pickle=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[save] {out_dir}: " + ", ".join(
        f"{n}{np.asarray(a).shape}" for n, a in
        (("bg", background), ("sal", salient), ("sp", spatial),
         ("salsp", salient_spatial)) if a is not None))


def load_embeddings(emb_dir: str):
    out = {}
    for name in ("background", "salient", "spatial", "salient_spatial"):
        p = os.path.join(emb_dir, f"{name}.npy")
        if os.path.exists(p):
            out[name] = np.load(p)
    p = os.path.join(emb_dir, "obs_names.npy")
    if os.path.exists(p):
        out["obs_names"] = np.load(p, allow_pickle=True)
    with open(os.path.join(emb_dir, "meta.json")) as fh:
        out["meta"] = json.load(fh)
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0
