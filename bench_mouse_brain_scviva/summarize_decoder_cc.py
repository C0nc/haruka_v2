"""Summarize CellCharter aggregation of Haruka decoder outputs."""

from __future__ import annotations

import json
import os

import pandas as pd


ROOT = os.path.join(os.path.dirname(__file__), "results_fair_5k")
METHODS = {
    "decoder composition": ("haruka_composition", "haruka_cvi_scviva_composition_seed0"),
    "decoder composition + CellCharter": (
        "haruka_decoder_cc_composition", "haruka_decoder_composition_cc_seed0"
    ),
    "decoder full + CellCharter": (
        "haruka_decoder_cc_full", "haruka_decoder_full_cc_seed0"
    ),
}


def main() -> None:
    rows = []
    for section in ("A8", "A9", "B17", "B3", "B5"):
        tag = f"mouse_brain_{section}_paired_5k"
        for method, (branch, leaf) in METHODS.items():
            path = os.path.join(ROOT, branch, tag, leaf, "brain_pair_ari.json")
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
    frame.to_csv(os.path.join(ROOT, "decoder_cellcharter_comparison.csv"), index=False)
    summary = frame.groupby("method")[["background_subregion_ari", "salient_domain_ari"]].agg(["mean", "std"])
    summary.to_csv(os.path.join(ROOT, "decoder_cellcharter_comparison_mean_std.csv"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
