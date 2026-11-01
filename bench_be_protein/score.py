"""Score all four BE protein factors with the shared PCA + GMM protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

MERFISH = os.path.join(os.path.dirname(__file__), "..", "bench_merfish")
sys.path.insert(0, MERFISH)
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


ASSIGNMENT = {
    "background": "gt_background_composition",
    "spatial": "gt_background_niche",
    "salient": "gt_salient_intrinsic",
    "salient_spatial": "gt_salient_domain",
}
TRUTHS = list(ASSIGNMENT.values())
TREATED_ONLY = {"gt_salient_intrinsic", "gt_salient_domain"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.emb_dir)
    saved = pd.Index(loaded["obs_names"].astype(str))
    pos = pd.Index(a.obs_names.astype(str)).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Embedding rows do not align to BE pair")
    obs = a.obs.iloc[pos]
    treated = obs["condition"].astype(str).to_numpy() == "disease"

    scores = {}
    diagonal, off_diagonal = [], []
    for rep_name, assigned_truth in ASSIGNMENT.items():
        rep = np.asarray(loaded[rep_name])
        for truth_name in TRUTHS:
            mask = treated if truth_name in TREATED_ONLY else np.ones(len(obs), bool)
            truth = obs[truth_name].astype(str).to_numpy()[mask]
            z = embed(rep[mask], seed=args.seed)
            pred = gmm_on_embedding(z, len(np.unique(truth)), args.seed)
            value = ari(pred, truth)
            scores[f"ARI__{rep_name}__{truth_name}"] = value
            if truth_name == assigned_truth:
                diagonal.append(value)
            else:
                off_diagonal.append(value)

    result = {
        "section": str(obs["section"].iloc[0]),
        "method": loaded["meta"].get("method"),
        "protocol": "StandardScaler -> PCA<=50 -> full-covariance GMM",
        "n_cells": len(obs),
        "diagonal": {
            rep: scores[f"ARI__{rep}__{truth}"]
            for rep, truth in ASSIGNMENT.items()
        },
        "diagonal_mean": float(np.mean(diagonal)),
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "assignment_margin": float(np.mean(diagonal) - np.mean(off_diagonal)),
        "matrix": scores,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    out = args.out or os.path.join(args.emb_dir, "be_four_factor_ari.json")
    with open(out, "w") as handle:
        handle.write(payload + "\n")


if __name__ == "__main__":
    main()
