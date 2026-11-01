"""Add fair CellCharter spatial slots to a two-slot protein baseline."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
MERFISH = os.path.join(HERE, "..", "bench_merfish")
sys.path.insert(0, MERFISH)
from common import load_embeddings, save_embeddings  # noqa: E402
from run_baselines_2x2 import _cc_aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    saved = pd.Index(loaded["obs_names"].astype(str))
    pos = pd.Index(a.obs_names.astype(str)).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Embedding rows do not align to protein data")
    a = a[pos].copy()
    background = np.asarray(loaded["background"], dtype=np.float32)
    salient = np.asarray(loaded["salient"], dtype=np.float32)
    target = a.obs["condition"].astype(str).to_numpy() == "disease"
    salient_zero = salient.copy()
    salient_zero[~target] = 0.0
    spatial = _cc_aggregate(a, background, "cinema_bg")
    salient_spatial = _cc_aggregate(a, salient_zero, "cinema_sal")
    save_embeddings(
        args.output_dir, background=background, salient=salient,
        spatial=spatial, salient_spatial=salient_spatial,
        obs_names=loaded["obs_names"],
        **{
            **loaded["meta"],
            "method": f"{loaded['meta'].get('method', 'baseline')}_cellcharter",
            "source_result": args.input_dir,
            "spatial_readout": "CellCharter three-hop aggregation",
        },
    )


if __name__ == "__main__":
    main()
