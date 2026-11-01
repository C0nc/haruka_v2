"""Aggregate fair-subset brain ARIs across spatial methods."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


METHOD_PATHS = {
    "Haruka-scVIVA": ("haruka_composition", "haruka_cvi_scviva_composition_seed0"),
    "Haruka+CellCharter": ("haruka_cellcharter", "haruka_cvi_cellcharter_seed0"),
    "ContrastiveVI+CellCharter": ("baselines", "contrastivevi_2x2_seed0"),
    "CINEMA-OT+CellCharter": ("cinema_raw", "cinemaot_seed0"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for section in ("A8", "A9", "B17", "B3", "B5"):
        tag = f"mouse_brain_{section}_paired_5k"
        for method, (branch, leaf) in METHOD_PATHS.items():
            path = os.path.join(args.root, branch, tag, leaf, "brain_pair_ari.json")
            with open(path) as handle:
                result = json.load(handle)
            rows.append({
                "section": section,
                "method": method,
                "background_subregion_ari": result["scores"]["background_subregion"]["ari"],
                "salient_domain_ari": result["scores"]["salient_domain"]["ari"],
                "result_json": os.path.abspath(path),
            })
    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby("method")[["background_subregion_ari", "salient_domain_ari"]].agg(["mean", "std"])
    summary.to_csv(os.path.splitext(args.out)[0] + "_mean_std.csv")
    print(frame.to_string(index=False))
    print("\nMean +/- SD")
    print(summary.to_string())


if __name__ == "__main__":
    main()
