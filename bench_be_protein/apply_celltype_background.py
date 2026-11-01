"""Fuse paired Haruka protein latents with observed fine-cell-type composition."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        out[rows] = np.mean(
            [x[rows][nbr[:, : min(k, nbr.shape[1])]].mean(1) for k in ks], axis=0
        )
    return out


def celltype_composition(labels, coords, samples, ks=(12, 36, 72)):
    onehot = OneHotEncoder(sparse_output=False).fit_transform(labels[:, None]).astype(
        np.float32
    )
    return np.concatenate(
        [graph_mean(onehot, coords, samples, ks=(k,)) for k in ks], axis=1
    ).astype(np.float32)


def compress(x, dim=16):
    x = StandardScaler().fit_transform(x)
    return PCA(n_components=min(dim, x.shape[1]), random_state=0).fit_transform(x).astype(
        np.float32
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mode", choices=["balanced", "raw_concat", "celltype_only", "expression_only"],
        default="balanced",
    )
    parser.add_argument("--background_alpha", type=float, default=0.75)
    parser.add_argument("--salient_alpha", type=float, default=0.25)
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    z = np.asarray(loaded["background"], dtype=np.float32)
    s = np.asarray(loaded["salient"], dtype=np.float32)
    coords = np.asarray(a.obsm["spatial"])
    samples = a.obs["slice_id"].astype(str).to_numpy()
    target = a.obs["condition"].astype(str).to_numpy() == "disease"
    cell_type = a.obs["cell_type_label"].astype(str).to_numpy()
    z_residual = (
        (1 - args.background_alpha) * z
        + args.background_alpha * graph_mean(z, coords, samples)
    ).astype(np.float32)
    ct = celltype_composition(cell_type, coords, samples)
    if args.mode == "balanced":
        spatial = np.concatenate([compress(z_residual), compress(ct)], axis=1)
    elif args.mode == "raw_concat":
        spatial = np.concatenate([z_residual, ct], axis=1)
    elif args.mode == "celltype_only":
        spatial = ct
    else:
        spatial = z_residual

    salient_spatial = (
        (1 - args.salient_alpha) * s
        + args.salient_alpha * graph_mean(s, coords, samples)
    ).astype(np.float32)
    salient_spatial[~target] = 0.0
    save_embeddings(
        args.output_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=loaded["obs_names"],
        method=f"haruka_paired_celltype_{args.mode}_protein_mse",
        cell_composition_annotation="Cell Type",
        background_truth_not_used_for_training="neigh_name",
        background_fusion=args.mode,
        background_graph_alpha=args.background_alpha,
        salient_graph_alpha=args.salient_alpha,
        graph_scales=[12, 36, 72],
        supervised_cell_type_composition=True,
        supervised_niche_or_salient_labels=False,
        postfit=False,
    )


if __name__ == "__main__":
    main()
