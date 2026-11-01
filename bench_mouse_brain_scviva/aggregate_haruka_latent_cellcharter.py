"""Apply CellCharter aggregation separately to saved Haruka z and s latents."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402
from run_baselines_2x2 import _cc_aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    saved = pd.Index(loaded["obs_names"].astype(str))
    positions = pd.Index(data.obs_names.astype(str)).get_indexer(saved)
    if (positions < 0).any():
        raise ValueError("Haruka latent rows do not align to the paired dataset")
    aligned = data[positions].copy()

    z = np.asarray(loaded["background"], dtype=np.float32)
    s = np.asarray(loaded["salient"], dtype=np.float32)
    z_cellcharter = _cc_aggregate(aligned, z, "haruka_z")
    s_cellcharter = _cc_aggregate(aligned, s, "haruka_s")

    save_embeddings(
        args.output_dir,
        background=z,
        salient=s,
        spatial=z_cellcharter,
        salient_spatial=s_cellcharter,
        obs_names=loaded["obs_names"],
        **{
            **loaded["meta"],
            "method": "haruka_z_s_cellcharter",
            "source_result": os.path.abspath(args.input_dir),
            "spatial_readout": (
                "separate three-hop CellCharter aggregation of z and s; "
                "own block removed"
            ),
        },
    )


if __name__ == "__main__":
    main()
