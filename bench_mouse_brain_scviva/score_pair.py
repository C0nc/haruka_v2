"""Score the two requested spatial ARIs for one mouse-brain section pair."""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.emb_dir)
    saved = pd.Index(loaded.get("obs_names", data.obs_names).astype(str))
    pos = pd.Index(data.obs_names.astype(str)).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Embedding rows are not aligned to the paired dataset")
    obs = data.obs.iloc[pos]
    target = obs["condition"].astype(str).to_numpy() == "disease"

    questions = (
        ("background_subregion", loaded["spatial"],
         obs["gt_background_subregion"].astype(str).to_numpy(), np.ones(len(obs), bool)),
        ("salient_domain", loaded["salient_spatial"],
         obs["gt_sal_domain"].astype(str).to_numpy(), target),
    )
    scores = {}
    for name, rep, truth, mask in questions:
        y = truth[mask]
        z = embed(rep[mask], seed=args.seed)
        pred = gmm_on_embedding(z, len(np.unique(y)), args.seed)
        scores[name] = {
            "ari": ari(pred, y),
            "n_cells": int(mask.sum()),
            "n_clusters": int(len(np.unique(y))),
        }
    result = {
        "section": str(data.obs["timepoint"].iloc[0]),
        "method": loaded["meta"].get("method"),
        "protocol": "StandardScaler -> PCA<=50 -> full-covariance GMM",
        "scores": scores,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    out = args.out or os.path.join(args.emb_dir, "brain_pair_ari.json")
    with open(out, "w") as handle:
        handle.write(payload + "\n")


if __name__ == "__main__":
    main()
