"""Numerically robust SPARC scorer for sparse MENDER ring blocks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler

import score_multislice_all_guides as base


def cluster_metrics(x, labels, n_components, seed=0):
    y = LabelEncoder().fit_transform(np.asarray(labels).astype(str))
    x = np.asarray(x, dtype=np.float64)
    keep = np.std(x, axis=0) > 1e-8
    z = StandardScaler().fit_transform(x[:, keep])
    pred = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        n_init=5,
        reg_covar=1e-3,
        random_state=seed,
    ).fit_predict(z)
    return {
        "ARI": float(adjusted_rand_score(y, pred)),
        "NMI": float(normalized_mutual_info_score(y, pred)),
        "silhouette": float(silhouette_score(z, pred)),
    }


if __name__ == "__main__":
    base.cluster_metrics = cluster_metrics
    base.main()
