"""Direct K-means controls for the paired composition simulation.

These are diagnostic data baselines rather than learned models. They quantify
what the observed local cell-type composition and the exact paired composition
difference can recover under the same K-means readout used for Haruka.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

MERFISH = os.path.join(os.path.dirname(__file__), "..", "bench_merfish")
sys.path.insert(0, MERFISH)

from run_haruka_cvi_scviva import neighbor_index  # noqa: E402
from run_haruka_shared_decoder import annotated_composition  # noqa: E402


def cluster_ari(x, truth, seed, n_init):
    n_clusters = len(np.unique(truth))
    pred = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=n_init
    ).fit_predict(np.asarray(x, dtype=np.float32))
    return float(adjusted_rand_score(truth, pred))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k_niche", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_init", type=int, default=20)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    target = data.obs["condition"].astype(str).to_numpy() == "disease"
    control_rows = np.flatnonzero(~target)
    target_rows = np.flatnonzero(target)
    if len(control_rows) != len(target_rows):
        raise ValueError("Expected equal paired arms")

    labels = data.obs["observed_celltype"].astype(str).to_numpy()
    categories = np.unique(labels)
    neighbors = neighbor_index(data, args.k_niche, "slice_id")
    composition = annotated_composition(labels, neighbors, categories)
    control_composition = composition[control_rows]
    target_composition = composition[target_rows]
    paired_background = np.vstack([control_composition, control_composition])
    paired_delta = target_composition - control_composition

    bg_truth = data.obs["gt_background_subregion"].astype(str).to_numpy()
    sal_truth = data.obs["gt_sal_domain"].astype(str).to_numpy()[target]
    result = {
        "protocol": "raw 20-NN annotated composition -> KMeans (no scaling or PCA)",
        "section": str(data.obs["timepoint"].iloc[0]),
        "scores": {
            "observed_local_composition": {
                "background_subregion_ari": cluster_ari(
                    composition, bg_truth, args.seed, args.n_init
                ),
                "salient_domain_ari": cluster_ari(
                    target_composition, sal_truth, args.seed, args.n_init
                ),
            },
            "paired_composition_oracle": {
                "background_subregion_ari": cluster_ari(
                    paired_background, bg_truth, args.seed, args.n_init
                ),
                "salient_domain_ari": cluster_ari(
                    paired_delta, sal_truth, args.seed, args.n_init
                ),
            },
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
