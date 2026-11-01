"""Score paired mouse-brain spatial outputs directly with K-means.

This scorer intentionally performs no scaling, PCA, graph smoothing, or GMM.
It clusters each method's saved native spatial representation exactly as stored.
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args()

    data = ad.read_h5ad(args.data, backed="r")
    obs_names_path = os.path.join(args.emb_dir, "obs_names.npy")
    saved_names = (
        np.load(obs_names_path, allow_pickle=True).astype(str)
        if os.path.exists(obs_names_path)
        else data.obs_names.astype(str).to_numpy()
    )
    positions = pd.Index(data.obs_names.astype(str)).get_indexer(saved_names)
    if (positions < 0).any():
        raise ValueError("Embedding rows are not aligned to the paired dataset")
    obs = data.obs.iloc[positions]
    target = obs["condition"].astype(str).to_numpy() == "disease"

    meta_path = os.path.join(args.emb_dir, "meta.json")
    with open(meta_path) as handle:
        meta = json.load(handle)

    questions = (
        (
            "background_subregion",
            np.load(os.path.join(args.emb_dir, "spatial.npy")),
            obs["gt_background_subregion"].astype(str).to_numpy(),
            np.ones(len(obs), dtype=bool),
        ),
        (
            "salient_domain",
            np.load(os.path.join(args.emb_dir, "salient_spatial.npy")),
            obs["gt_sal_domain"].astype(str).to_numpy(),
            target,
        ),
    )
    scores = {}
    for name, representation, truth, mask in questions:
        y = truth[mask]
        x = np.asarray(representation[mask], dtype=np.float32)
        n_clusters = len(np.unique(y))
        pred = KMeans(
            n_clusters=n_clusters,
            random_state=args.seed,
            n_init=args.n_init,
        ).fit_predict(x)
        scores[name] = {
            "ari": float(adjusted_rand_score(y, pred)),
            "n_cells": int(mask.sum()),
            "n_clusters": int(n_clusters),
            "n_features": int(x.shape[1]),
        }

    result = {
        "method": meta.get("method"),
        "protocol": "raw saved spatial outputs -> KMeans (no scaling, PCA, or GMM)",
        "section": str(obs["timepoint"].iloc[0]),
        "scores": scores,
    }
    payload = json.dumps(result, indent=2)
    print(payload, flush=True)
    output = args.out or os.path.join(args.emb_dir, "direct_kmeans_ari.json")
    with open(output, "w") as handle:
        handle.write(payload + "\n")


if __name__ == "__main__":
    main()
