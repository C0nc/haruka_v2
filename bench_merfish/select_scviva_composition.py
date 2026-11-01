"""Export the predefined niche-composition blocks for spatial clustering."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import load_embeddings, save_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    loaded = load_embeddings(args.input_dir)
    meta = loaded["meta"]
    n_bg = int(meta["n_bg_states"])
    n_sal = int(meta["n_sal_states"])
    spatial = loaded["spatial"][:, :n_bg]
    salient_spatial = loaded["salient_spatial"][:, :n_sal]
    save_embeddings(
        args.output_dir,
        background=loaded.get("background"),
        salient=loaded.get("salient"),
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=loaded.get("obs_names"),
        **{
            **meta,
            "method": "haruka_cvi_scviva_composition",
            "source_result": args.input_dir,
            "spatial_readout": "predefined niche-composition decoder blocks",
            "selection_uses_ground_truth": False,
        },
    )


if __name__ == "__main__":
    main()
