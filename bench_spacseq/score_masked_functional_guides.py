"""Evaluate target function only on sgRNAs masked from model supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.data)
    order = np.load(Path(args.emb_dir) / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    design = adata.uns["functional_guide_design"]
    train_guide = {
        function: list(design["guides_by_target"][function])[0]
        for function in design["targets"]
    }
    test_guide = {
        function: list(design["guides_by_target"][function])[1]
        for function in design["targets"]
    }
    function = adata.obs.function.astype(str).to_numpy()
    guide = adata.obs.guide_identity.astype(str).to_numpy()
    anchor = adata.obs.is_anchor.to_numpy(bool)
    train = anchor & np.asarray([
        train_guide.get(f) == g for f, g in zip(function, guide)
    ])
    test = anchor & np.asarray([
        test_guide.get(f) == g for f, g in zip(function, guide)
    ])
    encoder = LabelEncoder().fit(sorted(train_guide))
    y_train = encoder.transform(function[train])
    y_test = encoder.transform(function[test])
    result = {
        "train_guides": train_guide,
        "masked_test_guides": test_guide,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "chance": float(1 / len(encoder.classes_)),
        "representations": {},
    }
    for name in ("background", "spatial", "salient", "salient_spatial"):
        x = np.load(Path(args.emb_dir) / f"{name}.npy")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
        ).fit(x[train], y_train)
        prediction = model.predict(x[test])
        z = StandardScaler().fit_transform(x[test])
        cluster = GaussianMixture(
            n_components=len(encoder.classes_), covariance_type="diag", n_init=10,
            random_state=0, reg_covar=1e-5,
        ).fit_predict(z)
        result["representations"][name] = {
            "masked_guide_function_balanced_accuracy": float(
                balanced_accuracy_score(y_test, prediction)
            ),
            "masked_guide_function_gmm_ari": float(
                adjusted_rand_score(y_test, cluster)
            ),
            "masked_guide_function_true_silhouette": float(
                silhouette_score(z, y_test)
            ),
        }
    output = Path(args.out)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
