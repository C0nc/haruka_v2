"""Create stable density-normalized spatial slots from paired Haruka latents."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402


def graph_mean(x, coords, samples, ks=(12, 36, 72)):
    out = np.zeros_like(x, dtype=np.float32)
    samples = np.asarray(samples).astype(str)
    for sample in np.unique(samples):
        rows = np.flatnonzero(samples == sample)
        width = min(max(ks) + 1, len(rows))
        nbr = NearestNeighbors(n_neighbors=width).fit(coords[rows]).kneighbors(
            return_distance=False
        )[:, 1:]
        means = [
            x[rows][nbr[:, : min(k, nbr.shape[1])]].mean(1)
            for k in ks
        ]
        out[rows] = np.mean(means, axis=0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    z = np.asarray(loaded["background"], dtype=np.float32)
    s = np.asarray(loaded["salient"], dtype=np.float32)
    coords = np.asarray(a.obsm["spatial"])
    samples = a.obs["slice_id"].astype(str).to_numpy()
    target = a.obs["condition"].astype(str).to_numpy() == "disease"
    spatial = ((1 - args.alpha) * z + args.alpha * graph_mean(z, coords, samples)).astype(
        np.float32
    )
    salient_spatial = (
        (1 - args.alpha) * s + args.alpha * graph_mean(s, coords, samples)
    ).astype(np.float32)
    salient_spatial[~target] = 0.0
    save_embeddings(
        args.output_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=loaded["obs_names"],
        method="haruka_paired_graph_protein_mse",
        likelihood="exact summed MSE plus paired counterfactual-delta MSE",
        graph_alpha=args.alpha,
        graph_scales=[12, 36, 72],
        supervised_labels=False,
        postfit=False,
        source_training_dir=os.path.abspath(args.input_dir),
    )


if __name__ == "__main__":
    main()
