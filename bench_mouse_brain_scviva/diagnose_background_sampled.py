"""Fast diagnostic comparison of Haruka background readouts."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import ari, embed, gmm_on_embedding, load_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--decoder_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_cells", type=int, default=20_000)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.decoder_dir)
    saved = pd.Index(loaded["obs_names"].astype(str))
    pos = pd.Index(data.obs_names.astype(str)).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Embedding rows do not align to data")
    obs = data.obs.iloc[pos]
    control = obs["condition"].astype(str).to_numpy() == "control"
    truth_all = obs["gt_background_subregion"].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed)
    classes = np.unique(truth_all[control])
    per_class = max(1, args.max_cells // len(classes))
    chosen = []
    for label in classes:
        candidates = np.flatnonzero(control & (truth_all == label))
        chosen.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
    chosen = np.asarray(sorted(chosen), dtype=int)
    truth = truth_all[chosen]
    n_bg = int(loaded["meta"]["n_bg_states"])
    readouts = {
        "intrinsic_z": loaded["background"],
        "decoder_composition": loaded["spatial"][:, :n_bg],
        "decoder_full": loaded["spatial"],
        "cellcharter_z": np.load(os.path.join(args.decoder_dir, "spatial_cellcharter.npy")),
    }
    result = {}
    for name, rep in readouts.items():
        pred = gmm_on_embedding(embed(rep[chosen], seed=args.seed), len(classes), args.seed)
        result[name] = ari(pred, truth)
    result["section"] = str(data.obs["timepoint"].iloc[0])
    result["diagnostic_cells"] = int(len(chosen))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
