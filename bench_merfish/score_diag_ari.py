"""Fast fixed-protocol ARI scorer for the four assigned 2x2 diagonals.

This intentionally skips dependence probes and the 12 off-diagonal cells.  It is
for screening training/simulation knobs; finalists must still run score_2x2.py.
"""

from __future__ import annotations

import argparse
import json

import anndata as ad
import numpy as np
import pandas as pd

from common import ari, embed, gmm_on_embedding, load_embeddings
from score_2x2 import ASSIGNMENT, TREATED_ONLY, own_block_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.emb_dir)
    pos = pd.Index(adata.obs_names).get_indexer(
        pd.Index(loaded.get("obs_names", adata.obs_names).astype(str))
    )
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} embedding rows are absent from data")
    obs = adata.obs.iloc[pos]
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    reps = {
        name: loaded[name]
        for name in ASSIGNMENT
        if name in loaded
    }
    reps = own_block_only(reps, loaded["meta"])

    scores = {}
    for rep_name, truth_name in ASSIGNMENT.items():
        if rep_name not in reps:
            continue
        mask = treated if truth_name in TREATED_ONLY else np.ones(len(obs), dtype=bool)
        truth = obs[truth_name].astype(str).to_numpy()[mask]
        z = embed(reps[rep_name][mask], seed=args.seed)
        labels = gmm_on_embedding(z, len(np.unique(truth)), args.seed)
        scores[rep_name] = ari(labels, truth)
    print(json.dumps(scores, sort_keys=True))


if __name__ == "__main__":
    main()
