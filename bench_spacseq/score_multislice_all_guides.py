"""Score four-slot biology and slice leakage for multi-slice SPAC-seq Haruka."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
    )


def random_cv_ba(x, labels, seed=0):
    y = LabelEncoder().fit_transform(np.asarray(labels).astype(str))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = []
    for train, test in cv.split(x, y):
        model = classifier().fit(x[train], y[train])
        scores.append(balanced_accuracy_score(y[test], model.predict(x[test])))
    return float(np.mean(scores))


def leave_slice_out_ba(x, labels, slices):
    y = LabelEncoder().fit_transform(np.asarray(labels).astype(str))
    groups = np.asarray(slices).astype(str)
    folds, predictions = [], np.full(len(y), -1, dtype=int)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = classifier().fit(x[train], y[train])
        pred = model.predict(x[test])
        predictions[test] = pred
        folds.append({
            "slice": str(np.unique(groups[test])[0]),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        })
    return float(balanced_accuracy_score(y, predictions)), folds


def cluster_metrics(x, labels, n_components, seed=0):
    y = LabelEncoder().fit_transform(np.asarray(labels).astype(str))
    z = StandardScaler().fit_transform(x)
    pred = GaussianMixture(
        n_components=n_components, covariance_type="diag", n_init=5,
        reg_covar=1e-5, random_state=seed,
    ).fit_predict(z)
    return {
        "ARI": float(adjusted_rand_score(y, pred)),
        "NMI": float(normalized_mutual_info_score(y, pred)),
        "silhouette": float(silhouette_score(z, pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    emb_dir = Path(args.emb_dir)
    order = np.load(emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    reps = {
        name: np.load(emb_dir / f"{name}.npy")
        for name in ("background", "spatial", "salient", "salient_spatial")
    }
    counts = adata.layers["counts"]
    totals = np.asarray(counts.sum(axis=1)).ravel().clip(min=1)
    normalized = counts.multiply(1e4 / totals[:, None]) if sparse.issparse(counts) else counts * (1e4 / totals[:, None])
    normalized = normalized.copy()
    if sparse.issparse(normalized):
        normalized.data = np.log1p(normalized.data)
        normalized = normalized.toarray()
    else:
        normalized = np.log1p(normalized)
    reps["pca_expression"] = PCA(50, random_state=args.seed).fit_transform(
        StandardScaler().fit_transform(normalized)
    )

    obs = adata.obs
    anchor = obs.is_anchor.to_numpy(bool)
    target_anchor = anchor & (obs.perturbation.astype(str).to_numpy() != "NT")
    target_context = obs.perturbation.astype(str).to_numpy() != "NT"
    known_niche = obs.niche.astype(str).to_numpy() != "Unknown"
    target_labels = obs.perturbation.astype(str).to_numpy()
    slices = obs.slice_id.astype(str).to_numpy()
    result = {
        "data": args.data,
        "emb_dir": args.emb_dir,
        "n_cells": int(adata.n_obs),
        "n_anchor": int(anchor.sum()),
        "n_target_anchor": int(target_anchor.sum()),
        "n_perturbations": int(np.unique(target_labels[target_anchor]).size),
        "chance": {
            "perturbation": 1.0 / np.unique(target_labels[target_anchor]).size,
            "slice": 1.0 / np.unique(slices).size,
        },
        "representations": {},
    }
    for name, x in reps.items():
        rep = {"width": int(x.shape[1])}
        rep["slice_balanced_accuracy_anchor"] = random_cv_ba(
            x[anchor], slices[anchor], args.seed
        )
        rep["target_anchor_perturbation_balanced_accuracy"] = random_cv_ba(
            x[target_anchor], target_labels[target_anchor], args.seed
        )
        loso, folds = leave_slice_out_ba(
            x[target_anchor], target_labels[target_anchor], slices[target_anchor]
        )
        rep["target_anchor_perturbation_leave_slice_out_balanced_accuracy"] = loso
        rep["target_anchor_perturbation_leave_slice_out_folds"] = folds
        rep["target_anchor_perturbation_clustering"] = cluster_metrics(
            x[target_anchor], target_labels[target_anchor],
            np.unique(target_labels[target_anchor]).size, args.seed
        )
        rep["target_context_perturbation_balanced_accuracy"] = random_cv_ba(
            x[target_context], target_labels[target_context], args.seed
        )
        context_loso, context_folds = leave_slice_out_ba(
            x[target_context], target_labels[target_context], slices[target_context]
        )
        rep["target_context_perturbation_leave_slice_out_balanced_accuracy"] = context_loso
        rep["target_context_perturbation_leave_slice_out_folds"] = context_folds
        rep["cell_type_leave_slice_out_balanced_accuracy"], _ = leave_slice_out_ba(
            x, obs.cell_type.astype(str), slices
        )
        if known_niche.sum() > 0:
            rep["niche_leave_slice_out_balanced_accuracy"], _ = leave_slice_out_ba(
                x[known_niche], obs.niche.astype(str).to_numpy()[known_niche],
                slices[known_niche]
            )
        result["representations"][name] = rep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
