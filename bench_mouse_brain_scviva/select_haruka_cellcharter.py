"""Export Haruka intrinsic latents with its saved CellCharter spatial readouts."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    loaded = load_embeddings(args.input_dir)
    save_embeddings(
        args.output_dir,
        background=loaded["background"],
        salient=loaded["salient"],
        spatial=np.load(os.path.join(args.input_dir, "spatial_cellcharter.npy")),
        salient_spatial=np.load(os.path.join(args.input_dir, "salient_spatial_cellcharter.npy")),
        obs_names=loaded["obs_names"],
        **{
            **loaded["meta"],
            "method": "haruka_cvi_cellcharter_readout",
            "source_result": args.input_dir,
            "spatial_readout": "CellCharter three-hop aggregation of Haruka z and s",
        },
    )


if __name__ == "__main__":
    main()
