"""Annotate general intrinsic-response axes after nuisance residualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def residualize(target, nuisance):
    target = StandardScaler().fit_transform(target)
    nuisance = StandardScaler().fit_transform(nuisance)
    prediction = np.zeros_like(target)
    for train, test in KFold(5, shuffle=True, random_state=0).split(target):
        prediction[test] = Ridge(alpha=10).fit(
            nuisance[train], target[train]
        ).predict(nuisance[test])
    return target - prediction


def ranked(names, values, n=20):
    order = np.argsort(np.abs(values))[::-1][:n]
    return [{"gene": str(names[i]), "value": float(values[i])} for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.data)
    root = Path(args.emb_dir)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    function = adata.obs.function.astype(str).to_numpy()
    anchor = adata.obs.is_anchor.to_numpy(bool) & (function != "NT")
    guide = adata.obs.guide_identity.astype(str).to_numpy()[anchor]
    slices = adata.obs.slice_id.astype(str).to_numpy()[anchor]
    salient = np.load(root / "salient.npy")[anchor]
    salsp = np.load(root / "salient_spatial.npy")[anchor]
    categorical = pd.get_dummies(
        pd.DataFrame({"guide": guide, "slice": slices}), dtype=float
    ).to_numpy()
    residual = residualize(salient, np.column_stack([salsp, categorical]))

    counts = adata.layers["counts"][anchor]
    total = np.asarray(counts.sum(1)).ravel().clip(min=1)
    expression = counts.multiply(1e4 / total[:, None])
    if sparse.issparse(expression):
        expression.data = np.log1p(expression.data)
        expression = expression.toarray()
    expression = np.log1p(expression) if not sparse.issparse(counts) else expression
    variance = expression.var(0)
    variable = variance > 1e-8
    expression_z = StandardScaler().fit_transform(expression[:, variable])
    names = np.asarray(adata.var_names.astype(str))[variable]

    axes = PCA(4, random_state=0).fit_transform(residual)
    result = {"axes": {}, "two_state_expression_contrast": {}}
    for axis in range(axes.shape[1]):
        axis_z = StandardScaler().fit_transform(axes[:, [axis]]).ravel()
        correlation = expression_z.T @ axis_z / len(axis_z)
        result["axes"][f"PC{axis + 1}"] = ranked(names, correlation)

    labels = GaussianMixture(
        2, covariance_type="diag", n_init=10, random_state=0, reg_covar=1e-5
    ).fit_predict(StandardScaler().fit_transform(residual))
    effect = expression_z[labels == 1].mean(0) - expression_z[labels == 0].mean(0)
    result["two_state_expression_contrast"] = ranked(names, effect, 30)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
