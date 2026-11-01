"""Spatial-only PCA + full-covariance GMM benchmark.

The two fixed questions are:
  * spatial -> gt_arch_domain, evaluated on all cells
  * salient_spatial -> gt_sal_domain, evaluated on target cells only
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

from common import ari, embed, gmm_on_embedding, load_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--target_label", default="disease")
    parser.add_argument("--background_truth", default="gt_arch_domain")
    parser.add_argument("--salient_truth", default="gt_sal_domain")
    parser.add_argument("--out")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.emb_dir)
    names = loaded.get("obs_names", adata.obs_names).astype(str)
    pos = pd.Index(adata.obs_names.astype(str)).get_indexer(pd.Index(names))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} embedding rows are absent from data")
    obs = adata.obs.iloc[pos]
    target = (obs[args.condition_key].astype(str) == args.target_label).to_numpy()

    contract = (
        ("background_spatial", "spatial", args.background_truth, np.ones(len(obs), bool)),
        ("salient_spatial", "salient_spatial", args.salient_truth, target),
    )
    result = {
        "method": loaded["meta"].get("method", os.path.basename(args.emb_dir)),
        "seed": args.seed,
        "protocol": "StandardScaler -> PCA<=50 -> full-covariance GMM",
        "scores": {},
    }
    for label, rep_key, truth_key, mask in contract:
        if rep_key not in loaded:
            raise KeyError(f"Missing {rep_key}.npy in {args.emb_dir}")
        if truth_key not in obs:
            raise KeyError(f"Missing obs truth column {truth_key}")
        truth = obs[truth_key].astype(str).to_numpy()[mask]
        z = embed(loaded[rep_key][mask], seed=args.seed)
        pred = gmm_on_embedding(z, len(np.unique(truth)), seed=args.seed)
        result["scores"][label] = {
            "ari": ari(pred, truth),
            "n_cells": int(mask.sum()),
            "n_clusters": int(len(np.unique(truth))),
            "truth": truth_key,
        }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    out = args.out or os.path.join(args.emb_dir, "spatial_ari.json")
    with open(out, "w") as handle:
        handle.write(payload + "\n")


if __name__ == "__main__":
    main()
