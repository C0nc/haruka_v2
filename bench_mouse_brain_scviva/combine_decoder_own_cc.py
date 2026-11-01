"""Concatenate a decoder's own output with its saved CellCharter hop blocks."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_dir", required=True)
    parser.add_argument("--cc_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--block", choices=("composition", "full"), required=True)
    args = parser.parse_args()
    decoder = load_embeddings(args.decoder_dir)
    cc = load_embeddings(args.cc_dir)
    bg, sal = decoder["spatial"], decoder["salient_spatial"]
    if args.block == "composition":
        bg = bg[:, : int(decoder["meta"]["n_bg_states"])]
        sal = sal[:, : int(decoder["meta"]["n_sal_states"])]
    save_embeddings(
        args.output_dir,
        background=decoder["background"],
        salient=decoder["salient"],
        spatial=np.concatenate([bg, cc["spatial"]], axis=1),
        salient_spatial=np.concatenate([sal, cc["salient_spatial"]], axis=1),
        obs_names=decoder["obs_names"],
        **{
            **decoder["meta"],
            "method": f"haruka_decoder_{args.block}_own_plus_cellcharter",
            "decoder_block": args.block,
            "spatial_readout": "decoder own block concatenated with three CellCharter hop blocks",
        },
    )


if __name__ == "__main__":
    main()
