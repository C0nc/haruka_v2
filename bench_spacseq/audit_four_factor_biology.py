"""Audit biological and nuisance information in Haruka's four latent slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
    )


def cv_ba(x, labels, seed=0):
    y = LabelEncoder().fit_transform(np.asarray(labels).astype(str))
    counts = np.bincount(y)
    n_splits = min(5, int(counts.min()))
    if len(counts) < 2 or n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    scores = []
    for train, test in cv.split(x, y):
        model = classifier().fit(x[train], y[train])
        scores.append(balanced_accuracy_score(y[test], model.predict(x[test])))
    return float(np.mean(scores))


def leave_group_ba(x, labels, groups):
    encoder = LabelEncoder().fit(np.asarray(labels).astype(str))
    y = encoder.transform(np.asarray(labels).astype(str))
    groups = np.asarray(groups).astype(str)
    prediction = np.full(len(y), -1, dtype=int)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        if len(np.unique(y[train])) != len(encoder.classes_):
            return None
        model = classifier().fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    return float(balanced_accuracy_score(y, prediction))


def eta_squared(x, labels):
    """Fraction of standardized latent variance attributable to group centroids."""
    x = StandardScaler().fit_transform(np.asarray(x))
    labels = np.asarray(labels).astype(str)
    overall = x.mean(axis=0)
    total = float(np.square(x - overall).sum())
    between = 0.0
    for label in np.unique(labels):
        group = x[labels == label]
        between += len(group) * float(np.square(group.mean(axis=0) - overall).sum())
    return float(between / total) if total > 0 else None


def true_silhouette(x, labels, seed=0, max_cells=5000):
    labels = np.asarray(labels).astype(str)
    if len(np.unique(labels)) < 2:
        return None
    rng = np.random.default_rng(seed)
    if len(labels) > max_cells:
        take = np.sort(rng.choice(len(labels), max_cells, replace=False))
        x, labels = x[take], labels[take]
    return float(silhouette_score(StandardScaler().fit_transform(x), labels))


def cross_validated_r2(source, target, seed=0):
    source = StandardScaler().fit_transform(source)
    target = StandardScaler().fit_transform(target)
    prediction = np.zeros_like(target)
    for train, test in KFold(5, shuffle=True, random_state=seed).split(source):
        model = Ridge(alpha=10.0).fit(source[train], target[train])
        prediction[test] = model.predict(source[test])
    return float(r2_score(target, prediction, multioutput="variance_weighted"))


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
    obs = adata.obs
    slices = obs.slice_id.astype(str).to_numpy()
    anchors = obs.is_anchor.to_numpy(bool)
    perturbation = obs.perturbation.astype(str).to_numpy()
    target_anchor = anchors & (perturbation != "NT")
    known_niche = obs.niche.astype(str).to_numpy() != "Unknown"
    factors = {
        "slice": (np.ones(adata.n_obs, dtype=bool), slices),
        "condition": (np.ones(adata.n_obs, dtype=bool), obs.condition.astype(str).to_numpy()),
        "cell_type": (np.ones(adata.n_obs, dtype=bool), obs.cell_type.astype(str).to_numpy()),
        "niche": (known_niche, obs.niche.astype(str).to_numpy()),
        "guide_target_anchors": (target_anchor, perturbation),
    }

    result = {
        "data": args.data,
        "emb_dir": args.emb_dir,
        "n_cells": int(adata.n_obs),
        "n_target_anchors": int(target_anchor.sum()),
        "factor_chance": {},
        "representations": {},
        "cross_slot_linear_r2": {},
    }
    for factor, (mask, labels) in factors.items():
        result["factor_chance"][factor] = float(1 / len(np.unique(labels[mask])))

    for name, x in reps.items():
        entry = {"width": int(x.shape[1]), "factors": {}}
        for factor, (mask, labels) in factors.items():
            factor_entry = {
                "n": int(mask.sum()),
                "classes": int(len(np.unique(labels[mask]))),
                "eta_squared": eta_squared(x[mask], labels[mask]),
                "balanced_accuracy_cv": cv_ba(x[mask], labels[mask], args.seed),
                "balanced_accuracy_leave_slice_out": (
                    None if factor == "slice" else
                    leave_group_ba(x[mask], labels[mask], slices[mask])
                ),
                "true_label_silhouette": true_silhouette(
                    x[mask], labels[mask], args.seed
                ),
            }
            entry["factors"][factor] = factor_entry
        result["representations"][name] = entry

    for source_name, source in reps.items():
        result["cross_slot_linear_r2"][source_name] = {}
        for target_name, target in reps.items():
            if source_name != target_name:
                result["cross_slot_linear_r2"][source_name][target_name] = (
                    cross_validated_r2(source, target, args.seed)
                )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
