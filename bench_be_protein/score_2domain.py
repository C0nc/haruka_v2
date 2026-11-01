"""Score only background-domain and salient-domain recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np

MERFISH = os.path.join(os.path.dirname(__file__), "..", "bench_merfish")
sys.path.insert(0, MERFISH)
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


def cluster_ari(rep, truth, seed):
    z = embed(np.asarray(rep), seed=seed)
    pred = gmm_on_embedding(z, len(np.unique(truth)), seed)
    return ari(pred, truth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.emb_dir)
    saved = loaded["obs_names"].astype(str)
    pos = a.obs_names.astype(str).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Embedding rows do not align to BE pair")
    obs = a.obs.iloc[pos]
    treated = obs["condition"].astype(str).to_numpy() == "disease"

    background_ari = cluster_ari(
        loaded["spatial"],
        obs["gt_background_domain"].astype(str).to_numpy(),
        args.seed,
    )
    salient_ari = cluster_ari(
        np.asarray(loaded["salient_spatial"])[treated],
        obs.loc[treated, "gt_salient_domain"].astype(str).to_numpy(),
        args.seed,
    )
    result = {
        "section": str(obs["section"].iloc[0]),
        "method": loaded["meta"].get("method"),
        "protocol": "StandardScaler -> PCA<=50 -> full-covariance GMM",
        "background_domain_ari": background_ari,
        "salient_domain_ari": salient_ari,
        "two_domain_mean_ari": float((background_ari + salient_ari) / 2),
        "background_truth": "neigh_name",
        "salient_truth": "simulated condition-specific spatial domain",
        "composition_role": "neigh_sub used for balanced generation; not scored",
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    with open(os.path.join(args.emb_dir, "be_two_domain_ari.json"), "w") as handle:
        handle.write(payload + "\n")


if __name__ == "__main__":
    main()
