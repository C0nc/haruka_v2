"""Summarize full-density decoder-composition and MENDER-composition results."""

from __future__ import annotations

import glob
import json
import os

import pandas as pd


ROOT = os.path.dirname(__file__)
MENDER_ROOT = os.path.join(ROOT, "results_mender")
ORIGINAL = os.path.join(ROOT, "results_composition", "summary.csv")


def main() -> None:
    rows = []
    pattern = os.path.join(
        MENDER_ROOT, "mouse_brain_*_paired",
        "haruka_decoder_composition_mender_seed0", "brain_pair_ari.json",
    )
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            result = json.load(handle)
        rows.append({
            "section": result["section"],
            "background_subregion_ari": result["scores"]["background_subregion"]["ari"],
            "salient_domain_ari": result["scores"]["salient_domain"]["ari"],
            "result_json": os.path.abspath(path),
        })
    mender = pd.DataFrame(rows).sort_values("section")
    mender.to_csv(os.path.join(MENDER_ROOT, "summary.csv"), index=False)
    mender[["background_subregion_ari", "salient_domain_ari"]].agg(
        ["mean", "std"]
    ).to_csv(os.path.join(MENDER_ROOT, "summary_mean_std.csv"))

    original = pd.read_csv(ORIGINAL).drop(columns=["result_json"], errors="ignore")
    comparison = original.merge(mender.drop(columns="result_json"), on="section", suffixes=(
        "_decoder", "_mender",
    ))
    for metric in ("background_subregion_ari", "salient_domain_ari"):
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_mender"] - comparison[f"{metric}_decoder"]
        )
    comparison.to_csv(os.path.join(MENDER_ROOT, "paired_comparison.csv"), index=False)
    print(mender.to_string(index=False))
    print("\nMean/std:\n", mender[[
        "background_subregion_ari", "salient_domain_ari"
    ]].agg(["mean", "std"]))
    print("\nMean paired deltas:\n", comparison[[
        "background_subregion_ari_delta", "salient_domain_ari_delta"
    ]].mean())


if __name__ == "__main__":
    main()
