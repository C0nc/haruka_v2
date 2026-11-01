"""Aggregate independently scored mouse-brain section pairs."""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    pattern = os.path.join(args.root, "*", "haruka_cvi_scviva_composition_seed0", "brain_pair_ari.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            result = json.load(handle)
        rows.append(
            {
                "section": result["section"],
                "background_subregion_ari": result["scores"]["background_subregion"]["ari"],
                "salient_domain_ari": result["scores"]["salient_domain"]["ari"],
                "result_json": os.path.abspath(path),
            }
        )
    if not rows:
        raise RuntimeError(f"No score files found under {args.root}")
    frame = pd.DataFrame(rows).sort_values("section")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame[["background_subregion_ari", "salient_domain_ari"]].agg(["mean", "std"])
    summary.to_csv(os.path.splitext(args.out)[0] + "_mean_std.csv")
    print(frame.to_string(index=False))
    print("\nMean +/- SD")
    print(summary.to_string())


if __name__ == "__main__":
    main()
