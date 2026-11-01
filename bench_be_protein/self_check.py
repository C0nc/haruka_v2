"""Validate independence, geometry, and protein-emission contracts."""

from __future__ import annotations

import argparse
import glob
import json

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def neighbor_agreement(coords, labels, k=12):
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(
        return_distance=False
    )[:, 1:]
    return float(np.mean(np.asarray(labels)[neighbors] == np.asarray(labels)[:, None]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", nargs="+", default=glob.glob("bench_be_protein/data_pairs/*.h5ad")
    )
    parser.add_argument("--out", default="bench_be_protein/data_pairs/self_check.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.data):
        a = ad.read_h5ad(path)
        n = a.n_obs // 2
        x = np.asarray(a.layers["protein"], dtype=np.float32)
        coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
        control, condition = x[:n], x[n:]
        if not np.array_equal(coords[:n], coords[n:]):
            raise RuntimeError(f"{path}: paired coordinates differ")
        obs = a.obs.iloc[n:]
        bg_i = obs["gt_background_composition"].astype(str).to_numpy()
        bg_s = obs["gt_background_niche"].astype(str).to_numpy()
        sal_i = obs["gt_salient_intrinsic"].astype(str).to_numpy()
        sal_s = obs["gt_salient_domain"].astype(str).to_numpy()
        nmis = [
            normalized_mutual_info_score(sal_i, bg_i),
            normalized_mutual_info_score(sal_i, bg_s),
            normalized_mutual_info_score(sal_s, bg_i),
            normalized_mutual_info_score(sal_s, bg_s),
            normalized_mutual_info_score(sal_i, sal_s),
        ]
        row = {
            "section": obs["section"].iloc[0],
            "n_cells_original": n,
            "max_factor_nmi": float(max(nmis)),
            "background_niche_knn_agreement": neighbor_agreement(coords[:n], bg_s),
            "background_composition_knn_agreement": neighbor_agreement(coords[:n], bg_i),
            "salient_domain_knn_agreement": neighbor_agreement(coords[:n], sal_s),
            "salient_intrinsic_knn_agreement": neighbor_agreement(coords[:n], sal_i),
            "changed_proteins": int(np.any(condition != control, axis=0).sum()),
            "unchanged_proteins": int(np.all(condition == control, axis=0).sum()),
            "mean_abs_effect": float(np.mean(np.abs(condition - control))),
            "control_finite": bool(np.isfinite(control).all()),
            "condition_finite": bool(np.isfinite(condition).all()),
        }
        if row["max_factor_nmi"] > 0.01:
            raise RuntimeError(f"{path}: factor leakage {row}")
        if row["salient_domain_knn_agreement"] <= 0.5:
            raise RuntimeError(f"{path}: salient domain is not spatially coherent {row}")
        if row["salient_intrinsic_knn_agreement"] >= 0.45:
            raise RuntimeError(f"{path}: salient intrinsic state is spatially clustered {row}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
