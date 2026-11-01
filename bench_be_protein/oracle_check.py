"""Verify that each simulated protein program is perfectly identifiable in paired deltas."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parent


def score(delta: np.ndarray, truth: np.ndarray) -> float:
    pred = GaussianMixture(
        n_components=len(np.unique(truth)),
        covariance_type="full",
        reg_covar=1e-6,
        random_state=0,
    ).fit_predict(delta.astype(np.float64))
    return float(adjusted_rand_score(truth, pred))


def main() -> None:
    rows = []
    for path in sorted((ROOT / "data_pairs").glob("be_*_paired.h5ad")):
        a = ad.read_h5ad(path)
        n = a.n_obs // 2
        delta = np.asarray(a.X[n:]) - np.asarray(a.X[:n])
        meta = a.uns["be_protein_simulation"]
        state_names = np.concatenate(list(meta["state_proteins"].values()))
        domain_names = np.concatenate(list(meta["domain_proteins"].values()))
        state_cols = a.var_names.get_indexer(state_names)
        domain_cols = a.var_names.get_indexer(domain_names)
        rows.append(
            {
                "section": meta["section"],
                "paired_delta_salient_intrinsic_ARI": score(
                    delta[:, state_cols],
                    a.obs.iloc[n:]["gt_salient_intrinsic"].to_numpy(),
                ),
                "paired_delta_salient_spatial_ARI": score(
                    delta[:, domain_cols],
                    a.obs.iloc[n:]["gt_salient_domain"].to_numpy(),
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data_pairs" / "oracle_paired_delta_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
