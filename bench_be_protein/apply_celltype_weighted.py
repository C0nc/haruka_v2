"""Expression-dominant Cell Type composition fusion for background niches."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402
from apply_celltype_background import (  # noqa: E402
    celltype_composition,
    compress,
    graph_mean,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expression_dim", type=int, required=True)
    parser.add_argument("--composition_dim", type=int, required=True)
    args = parser.parse_args()
    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    z = np.asarray(loaded["background"], dtype=np.float32)
    s = np.asarray(loaded["salient"], dtype=np.float32)
    coords = np.asarray(a.obsm["spatial"])
    samples = a.obs["slice_id"].astype(str).to_numpy()
    target = a.obs["condition"].astype(str).to_numpy() == "disease"
    z_residual = (0.25 * z + 0.75 * graph_mean(z, coords, samples)).astype(np.float32)
    ct = celltype_composition(
        a.obs["cell_type_label"].astype(str).to_numpy(), coords, samples
    )
    spatial = np.concatenate(
        [compress(z_residual, args.expression_dim), compress(ct, args.composition_dim)],
        axis=1,
    )
    salient_spatial = (0.75 * s + 0.25 * graph_mean(s, coords, samples)).astype(
        np.float32
    )
    salient_spatial[~target] = 0.0
    save_embeddings(
        args.output_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=loaded["obs_names"],
        method="haruka_paired_celltype_expression_weighted_protein_mse",
        cell_composition_annotation="Cell Type",
        expression_dim=args.expression_dim,
        composition_dim=args.composition_dim,
        background_truth_not_used_for_training="neigh_name",
        graph_scales=[12, 36, 72],
        supervised_niche_or_salient_labels=False,
        postfit=False,
    )


if __name__ == "__main__":
    main()
