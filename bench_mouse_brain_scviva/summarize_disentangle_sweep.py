"""Summarize recovery and cross-slot leakage for an A8 disentanglement sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def cluster_ari(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    pred = KMeans(
        n_clusters=len(np.unique(y)), random_state=seed, n_init=20
    ).fit_predict(np.asarray(x, dtype=np.float32))
    return float(adjusted_rand_score(y, pred))


def cv_ridge_r2(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Held-out mean R2 for standardized multivariate y predicted from x."""
    predictions = np.zeros_like(y, dtype=np.float64)
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in folds.split(x):
        x_scaler = StandardScaler().fit(x[train])
        y_scaler = StandardScaler().fit(y[train])
        x_train = x_scaler.transform(x[train])
        y_train = y_scaler.transform(y[train])
        x_test = x_scaler.transform(x[test])
        model = Ridge(alpha=len(train) * 0.01).fit(x_train, y_train)
        predictions[test] = y_scaler.inverse_transform(model.predict(x_test))
    return float(r2_score(y, predictions, multioutput="uniform_average"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--method", action="append", nargs=2, metavar=("NAME", "EMB_DIR"), required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data, backed="r")
    rows = []
    for name, directory in args.method:
        directory = Path(directory)
        names = np.load(directory / "obs_names.npy", allow_pickle=True).astype(str)
        positions = pd.Index(data.obs_names.astype(str)).get_indexer(names)
        if (positions < 0).any():
            raise ValueError(f"Rows for {name} do not align to the dataset")
        obs = data.obs.iloc[positions]
        target = obs["condition"].astype(str).to_numpy() == "disease"
        fine = obs["gt_background_subregion"].astype(str).to_numpy()
        broad = obs["region"].astype(str).to_numpy()
        domain = obs["gt_sal_domain"].astype(str).to_numpy()
        z = np.load(directory / "background.npy")
        s = np.load(directory / "salient.npy")
        z_sp = np.load(directory / "spatial.npy")
        s_sp = np.load(directory / "salient_spatial.npy")
        with open(directory / "direct_kmeans_ari.json") as handle:
            scores = json.load(handle)["scores"]
        direct_path = directory.parents[2] / "haruka_direct" / directory.parent.name
        # Sweep paths use a different method leaf name but the same dataset parent.
        direct_path = directory.parents[2] / "haruka_direct" / directory.parent.name / "haruka_direct_decoders_seed0"
        direct_scores = None
        if (direct_path / "direct_decoder_kmeans_ari.json").exists():
            with open(direct_path / "direct_decoder_kmeans_ari.json") as handle:
                direct_scores = json.load(handle)["scores"]
        rows.append(
            {
                "variant": name,
                "background_fine_ari": scores["background_subregion"]["ari"],
                "background_broad_ari": cluster_ari(z_sp, broad, args.seed),
                "salient_domain_ari": scores["salient_domain"]["ari"],
                "z_salient_leakage_ari": cluster_ari(z_sp[target], domain[target], args.seed),
                "s_background_fine_leakage_ari": cluster_ari(s_sp[target], fine[target], args.seed),
                "s_background_broad_leakage_ari": cluster_ari(s_sp[target], broad[target], args.seed),
                "cv_r2_z_from_s": cv_ridge_r2(s[target], z[target], args.seed),
                "cv_r2_s_from_z": cv_ridge_r2(z[target], s[target], args.seed),
                "decoder_background_ari": (
                    direct_scores["background_subregion"]["ari"] if direct_scores else np.nan
                ),
                "decoder_salient_ari": (
                    direct_scores["salient_domain"]["ari"] if direct_scores else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
