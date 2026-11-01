"""Final Cell-Type background fusion with label-free stable salient graph gating."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402
from apply_celltype_background import celltype_composition, compress, graph_mean  # noqa: E402


def unsupervised_clusters(x, n_clusters=3):
    z = StandardScaler().fit_transform(x)
    z = PCA(n_components=min(50, z.shape[1]), random_state=0).fit_transform(z)
    return GaussianMixture(
        n_clusters, covariance_type="full", random_state=0
    ).fit_predict(z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stability_threshold", type=float, default=0.8)
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
    # Expression is dominant (24 dimensions) and composition contributes a
    # non-redundant eight-dimensional block.
    spatial = np.concatenate([compress(z_residual, 24), compress(ct, 8)], axis=1)

    s_smoothed = (0.75 * s + 0.25 * graph_mean(s, coords, samples)).astype(np.float32)
    raw_clusters = unsupervised_clusters(s[target])
    smooth_clusters = unsupervised_clusters(s_smoothed[target])
    stability = float(adjusted_mutual_info_score(raw_clusters, smooth_clusters))
    salient_alpha = 0.25 if stability >= args.stability_threshold else 0.0
    salient_spatial = (
        (1 - salient_alpha) * s
        + salient_alpha * graph_mean(s, coords, samples)
    ).astype(np.float32)
    salient_spatial[~target] = 0.0

    save_embeddings(
        args.output_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=loaded["obs_names"],
        method="haruka_paired_celltype_final_protein_mse",
        cell_composition_annotation="Cell Type",
        background_expression_dim=24,
        background_composition_dim=8,
        background_graph_alpha=0.75,
        salient_graph_alpha=salient_alpha,
        salient_cluster_stability=stability,
        salient_stability_threshold=args.stability_threshold,
        graph_scales=[12, 36, 72],
        background_truth_not_used_for_training="neigh_name",
        supervised_niche_or_salient_labels=False,
        postfit_training=False,
    )
    print(f"stability={stability:.4f}, salient_alpha={salient_alpha:.2f}")


if __name__ == "__main__":
    main()
