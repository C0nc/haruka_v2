"""Locate the background-scVIVA failure: target construction or decoder fit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402
from run_haruka_cvi_scviva import (  # noqa: E402
    fit_reference,
    neighbor_index,
    niche_features,
    pseudo_states,
)


def cluster_ari(rep: np.ndarray, truth: np.ndarray, seed: int) -> float:
    pred = gmm_on_embedding(embed(rep, seed=seed), len(np.unique(truth)), seed)
    return ari(pred, truth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--decoder_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.decoder_dir)
    cfg = SimpleNamespace(
        condition_key="condition", batch_key="timepoint", n_latent=32,
        reference_epochs=75, batch_size=2048, seed=args.seed,
    )
    z_ref, _ = fit_reference(data, cfg)
    states, z_scaled = pseudo_states(z_ref, np.ones(data.n_obs, bool), 12, args.seed)
    neighbors = neighbor_index(data, 20, "slice_id")
    composition, _ = niche_features(z_scaled, states, neighbors, 12)
    predicted = loaded["spatial"][:, :12]
    truth = data.obs["gt_background_subregion"].astype(str).to_numpy()
    celltype = data.obs["celltype"].astype(str).to_numpy()
    same_region = (truth[neighbors] == truth[:, None]).mean()
    result = {
        "section": str(data.obs["timepoint"].iloc[0]),
        "n_cells": int(data.n_obs),
        "neighbor_same_subregion_fraction": float(same_region),
        "pseudo_state_subregion_ari": float(adjusted_rand_score(truth, states)),
        "pseudo_state_subregion_nmi": float(normalized_mutual_info_score(truth, states)),
        "pseudo_state_celltype_ari": float(adjusted_rand_score(celltype, states)),
        "pseudo_state_celltype_nmi": float(normalized_mutual_info_score(celltype, states)),
        "reference_z_subregion_gmm_ari": cluster_ari(z_ref, truth, args.seed),
        "true_composition_subregion_gmm_ari": cluster_ari(composition, truth, args.seed),
        "predicted_composition_subregion_gmm_ari": cluster_ari(predicted, truth, args.seed),
        "decoder_composition_mse": float(np.mean((predicted - composition) ** 2)),
        "decoder_composition_variance_weighted_r2": float(
            r2_score(composition, predicted, multioutput="variance_weighted")
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
