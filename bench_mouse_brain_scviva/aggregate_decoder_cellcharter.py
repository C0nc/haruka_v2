"""Apply CellCharter neighborhood aggregation to Haruka decoder outputs."""

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
    parser.add_argument("--block", choices=("composition", "full"), default="composition")
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    saved = pd.Index(loaded["obs_names"].astype(str))
    pos = pd.Index(data.obs_names.astype(str)).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Decoder rows do not align to the paired dataset")
    a = data[pos].copy()
    bg = loaded["spatial"]
    sal = loaded["salient_spatial"]
    if args.block == "composition":
        bg = bg[:, : int(loaded["meta"]["n_bg_states"])]
        sal = sal[:, : int(loaded["meta"]["n_sal_states"])]
    bg_cc = _cc_aggregate(a, bg, f"decoder_bg_{args.block}")
    sal_cc = _cc_aggregate(a, sal, f"decoder_sal_{args.block}")
    save_embeddings(
        args.output_dir,
        background=loaded["background"],
        salient=loaded["salient"],
        spatial=bg_cc,
        salient_spatial=sal_cc,
        obs_names=loaded["obs_names"],
        **{
            **loaded["meta"],
            "method": f"haruka_decoder_{args.block}_cellcharter",
            "source_result": args.input_dir,
            "decoder_block": args.block,
            "spatial_readout": "three-hop CellCharter aggregation of decoder output; own block removed",
        },
    )


if __name__ == "__main__":
    main()
