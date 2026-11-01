"""Build the corrected individual-sgRNA multi-slice dataset with all NT controls."""

from __future__ import annotations

import numpy as np

import prepare_multislice_all_guides as base
from prepare_multislice_all_guides import main


original_load_slice = base.load_slice


def load_slice_with_guide_classes(raw_root, slice_id):
    data = original_load_slice(raw_root, slice_id)
    data["feature_target"] = np.asarray([
        "NT" if name.lower().startswith("sgnon-targeting") else name
        for name in data["names"]
    ], dtype=object)
    return data


def all_nt_balanced_guide_indices(data, targets, n_per_group, seed):
    rng = np.random.default_rng(seed)
    feature = data["single_feature"]
    label = np.full(len(feature), "", dtype=object)
    valid = feature >= 0
    label[valid] = data["feature_target"][feature[valid]]
    eligible = data["single"] & data["tumor"]
    result = {"NT": np.flatnonzero(eligible & (label == "NT"))}
    for guide in targets:
        idx = np.flatnonzero(eligible & (label == guide))
        if len(idx) < n_per_group:
            raise ValueError(f"{guide}: only {len(idx)} anchors, need {n_per_group}")
        result[guide] = np.sort(rng.choice(idx, n_per_group, replace=False))
    return result


base.load_slice = load_slice_with_guide_classes
base.balanced_anchor_indices = all_nt_balanced_guide_indices


if __name__ == "__main__":
    main()
