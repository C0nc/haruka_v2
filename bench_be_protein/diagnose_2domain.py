"""Diagnose signal, clipping, latent separation, and spatial smoothing in BE pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "bench_merfish"))
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


def cluster_ari(x, truth, seed):
    z = embed(np.asarray(x), seed=seed)
    return ari(gmm_on_embedding(z, len(np.unique(truth)), seed), truth)


def knn_agreement(coords, labels, k=12):
    indices = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(
        return_distance=False
    )[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


def eta_squared(x, labels):
    overall = x.mean(0)
    between = 0.0
    for group in np.unique(labels):
        take = labels == group
        between += take.mean() * np.square(x[take].mean(0) - overall).sum()
    total = np.square(x - overall).sum(1).mean()
    return float(between / max(total, 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    n = a.n_obs // 2
    raw_control = np.asarray(a.X[:n], dtype=np.float32)
    raw_condition = np.asarray(a.X[n:], dtype=np.float32)
    x_control = (np.clip(raw_control, -5, 5) + 5) / 5
    x_condition = (np.clip(raw_condition, -5, 5) + 5) / 5
    raw_delta = raw_condition - raw_control
    transformed_delta = x_condition - x_control
    expected_delta = raw_delta / 5
    changed = np.abs(raw_delta) > 1e-7
    retention = float(
        np.abs(transformed_delta[changed]).mean()
        / np.abs(expected_delta[changed]).mean()
    )

    niche = a.obs.iloc[n:]["gt_background_domain"].astype(str).to_numpy()
    salient = a.obs.iloc[n:]["gt_salient_domain"].astype(str).to_numpy()
    coords = np.asarray(a.obsm["spatial"][:n])
    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out = {
        "section": str(a.obs["section"].iloc[0]),
        "effect_transform_retention": retention,
        "changed_entries_clipped_fraction": float(
            np.mean(np.abs(transformed_delta[changed] - expected_delta[changed]) > 1e-6)
        ),
        "condition_salient_eta_squared_all_proteins": eta_squared(x_condition, salient),
        "paired_delta_salient_eta_squared": eta_squared(transformed_delta, salient),
        "background_12nn_agreement": knn_agreement(coords, niche),
        "salient_12nn_agreement": knn_agreement(coords, salient),
    }
    methods = {
        "haruka": "haruka_mender_mse_seed0",
        "contrastivevi": "contrastivevi_protein_seed0",
    }
    for label, directory in methods.items():
        emb_dir = os.path.join(args.result_root, tag, directory)
        loaded = load_embeddings(emb_dir)
        out[f"{label}_cell_background_ari"] = cluster_ari(
            loaded["background"],
            a.obs["gt_background_domain"].astype(str).to_numpy(),
            args.seed,
        )
        out[f"{label}_cell_salient_ari"] = cluster_ari(
            np.asarray(loaded["salient"])[n:], salient, args.seed
        )
        score = json.load(open(os.path.join(emb_dir, "be_two_domain_ari.json")))
        out[f"{label}_spatial_background_ari"] = score["background_domain_ari"]
        out[f"{label}_spatial_salient_ari"] = score["salient_domain_ari"]
    path = os.path.join(args.result_root, tag, "two_domain_diagnostics.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
