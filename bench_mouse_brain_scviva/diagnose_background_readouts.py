"""Compare label-free Haruka background readouts against brain subregions."""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--decoder_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.decoder_dir)
    saved = pd.Index(loaded["obs_names"].astype(str))
    pos = pd.Index(data.obs_names.astype(str)).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Embedding rows do not align to data")
    truth = data.obs.iloc[pos]["gt_background_subregion"].astype(str).to_numpy()
    n_clusters = len(np.unique(truth))
    n_bg = int(loaded["meta"]["n_bg_states"])
    readouts = {
        "intrinsic_z": loaded["background"],
        "decoder_composition": loaded["spatial"][:, :n_bg],
        "decoder_full": loaded["spatial"],
        "cellcharter_z": np.load(os.path.join(args.decoder_dir, "spatial_cellcharter.npy")),
    }
    result = {name: ari(gmm_on_embedding(embed(rep, seed=args.seed), n_clusters, args.seed), truth)
              for name, rep in readouts.items()}
    result["section"] = str(data.obs["timepoint"].iloc[0])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
