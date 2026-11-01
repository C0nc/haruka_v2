"""Evaluate a common density-normalized residual smoother without label training."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


def graph_mean(x, coords, samples, ks=(12, 36, 72)):
    out = np.zeros_like(x, dtype=np.float32)
    samples = np.asarray(samples).astype(str)
    for sample in np.unique(samples):
        rows = np.flatnonzero(samples == sample)
        width = min(max(ks) + 1, len(rows))
        nbr = NearestNeighbors(n_neighbors=width).fit(coords[rows]).kneighbors(
            return_distance=False
        )[:, 1:]
        blocks = []
        for k in ks:
            take = nbr[:, : min(k, nbr.shape[1])]
            blocks.append(x[rows][take].mean(1))
        out[rows] = np.mean(blocks, axis=0)
    return out


def score(x, truth):
    z = embed(x, seed=0)
    return ari(gmm_on_embedding(z, len(np.unique(truth)), 0), truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    args = parser.parse_args()
    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.emb_dir)
    coords = np.asarray(a.obsm["spatial"])
    samples = a.obs["slice_id"].astype(str).to_numpy()
    target = a.obs["condition"].astype(str).to_numpy() == "disease"
    z = np.asarray(loaded["background"], dtype=np.float32)
    s = np.asarray(loaded["salient"], dtype=np.float32)
    z_graph = graph_mean(z, coords, samples)
    s_graph = graph_mean(s, coords, samples)
    background = a.obs["gt_background_domain"].astype(str).to_numpy()
    salient = a.obs.loc[target, "gt_salient_domain"].astype(str).to_numpy()
    result = {"section": str(a.obs["section"].iloc[0]), "scores": {}}
    for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        z_fused = (1 - alpha) * z + alpha * z_graph
        s_fused = (1 - alpha) * s + alpha * s_graph
        result["scores"][str(alpha)] = {
            "background": score(z_fused, background),
            "salient": score(s_fused[target], salient),
        }
    out = os.path.join(args.emb_dir, "graph_residual_ablation.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
