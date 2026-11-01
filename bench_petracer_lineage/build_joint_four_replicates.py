#!/usr/bin/env python3
"""Combine the four PEtracer pairs into one control + four-condition study."""

from pathlib import Path

import anndata as ad
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data_m1t1_pairs_22k"
OUTPUT = ROOT / "data_m1t1_joint" / "petracer_M1T1_four_condition_110k.h5ad"


def main() -> None:
    pieces = []
    for index in range(1, 5):
        path = INPUT / f"petracer_M1T1_M1-{index}_paired_22k.h5ad"
        pair = ad.read_h5ad(path)
        condition = pair.obs["condition"].astype(str)
        if index == 1:
            control = pair[condition == "control"].copy()
            control.obs["graph_id"] = pd.Categorical(["M1T1_control"] * control.n_obs)
            control.obs["sample"] = pd.Categorical(["M1T1"] * control.n_obs)
            pieces.append(control)
        target = pair[condition == "disease"].copy()
        target.obs["sample"] = pd.Categorical([f"M1-{index}"] * target.n_obs)
        pieces.append(target)
    joint = ad.concat(pieces, join="inner", merge="same", index_unique=None)
    if not joint.obs_names.is_unique:
        joint.obs_names_make_unique()
    for key in ["condition", "graph_id", "sample", "gt_background_celltype"]:
        joint.obs[key] = joint.obs[key].astype("category")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joint.write_h5ad(OUTPUT, compression="gzip")
    print(joint)
    print(pd.crosstab(joint.obs["graph_id"], joint.obs["condition"]))
    print(OUTPUT)


if __name__ == "__main__":
    main()
