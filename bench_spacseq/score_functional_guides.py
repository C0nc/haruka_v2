"""Score target-function recovery while holding out individual sgRNA identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from score_multislice_all_guides import leave_slice_out_ba, random_cv_ba


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
    )


def eta_squared(x, labels):
    x = StandardScaler().fit_transform(x)
    labels = np.asarray(labels).astype(str)
    overall = x.mean(0)
    total = np.square(x - overall).sum()
    between = sum(
        (labels == label).sum()
        * np.square(x[labels == label].mean(0) - overall).sum()
        for label in np.unique(labels)
    )
    return float(between / total)


def leave_one_guide_out(x, functions, guides):
    encoder = LabelEncoder().fit(np.asarray(functions).astype(str))
    y = encoder.transform(np.asarray(functions).astype(str))
    guides = np.asarray(guides).astype(str)
    prediction = np.full(len(y), -1, dtype=int)
    folds = []
    for held_guide in sorted(np.unique(guides)):
        test = guides == held_guide
        train = ~test
        model = classifier().fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
        folds.append({
            "guide": held_guide,
            "function": str(np.unique(np.asarray(functions)[test])[0]),
            "accuracy": float((prediction[test] == y[test]).mean()),
            "n": int(test.sum()),
        })
    return float(balanced_accuracy_score(y, prediction)), folds


def within_function_guide_ba(x, functions, guides, seed=0):
    functions = np.asarray(functions).astype(str)
    guides = np.asarray(guides).astype(str)
    scores = {}
    for function in sorted(np.unique(functions)):
        take = functions == function
        y = LabelEncoder().fit_transform(guides[take])
        if len(np.unique(y)) < 2 or np.bincount(y).min() < 5:
            continue
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        prediction = np.full(len(y), -1, dtype=int)
        for train, test in cv.split(x[take], y):
            model = classifier().fit(x[take][train], y[train])
            prediction[test] = model.predict(x[take][test])
        scores[function] = float(balanced_accuracy_score(y, prediction))
    return {
        "mean": float(np.mean(list(scores.values()))),
        "per_function": scores,
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
    totals = np.asarray(counts.sum(1)).ravel().clip(min=1)
    normalized = counts.multiply(1e4 / totals[:, None])
    normalized.data = np.log1p(normalized.data)
    reps["pca_expression"] = PCA(50, random_state=args.seed).fit_transform(
        StandardScaler().fit_transform(
            normalized.toarray() if sparse.issparse(normalized) else normalized
        )
    )

    obs = adata.obs
    function = obs.function.astype(str).to_numpy()
    guide = obs.guide_identity.astype(str).to_numpy()
    slices = obs.slice_id.astype(str).to_numpy()
    target_anchor = obs.is_anchor.to_numpy(bool) & (function != "NT")
    f = function[target_anchor]
    g = guide[target_anchor]
    result = {
        "data": args.data,
        "emb_dir": args.emb_dir,
        "n_cells": int(adata.n_obs),
        "n_target_anchors": int(target_anchor.sum()),
        "n_functions": int(len(np.unique(f))),
        "n_guides": int(len(np.unique(g))),
        "chance_function": float(1 / len(np.unique(f))),
        "chance_within_function_guide": 0.5,
        "representations": {},
    }
    for name, all_x in reps.items():
        x = all_x[target_anchor]
        z = StandardScaler().fit_transform(x)
        pred = GaussianMixture(
            n_components=len(np.unique(f)), covariance_type="diag", n_init=10,
            reg_covar=1e-5, random_state=args.seed,
        ).fit_predict(z)
        guide_out, guide_folds = leave_one_guide_out(x, f, g)
        slice_out, _ = leave_slice_out_ba(x, f, slices[target_anchor])
        result["representations"][name] = {
            "width": int(x.shape[1]),
            "function_cv_balanced_accuracy": random_cv_ba(x, f, args.seed),
            "function_leave_slice_out_balanced_accuracy": slice_out,
            "function_leave_one_guide_out_balanced_accuracy": guide_out,
            "leave_one_guide_out_folds": guide_folds,
            "function_gmm_ari": float(adjusted_rand_score(f, pred)),
            "function_true_silhouette": float(silhouette_score(z, f)),
            "function_eta_squared": eta_squared(x, f),
            "guide_identity_eta_squared": eta_squared(x, g),
            "within_function_guide_identity_balanced_accuracy": (
                within_function_guide_ba(x, f, g, args.seed)
            ),
            "slice_cv_balanced_accuracy": random_cv_ba(
                x, slices[target_anchor], args.seed
            ),
        }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
