"""Build the multi-slice dataset while retaining every eligible NT anchor."""

from __future__ import annotations

import numpy as np

from prepare_multislice_all_guides import main
import prepare_multislice_all_guides as base


def all_nt_balanced_target_indices(data, targets, n_per_group, seed):
    """Use all NT controls and a balanced cross-slice sample per target class."""
    rng = np.random.default_rng(seed)
    feature = data["single_feature"]
    target = np.full(len(feature), "", dtype=object)
    valid = feature >= 0
    target[valid] = data["feature_target"][feature[valid]]
    eligible = data["single"] & data["tumor"]
    result = {"NT": np.flatnonzero(eligible & (target == "NT"))}
    if len(result["NT"]) == 0:
        raise ValueError("slice has no eligible NT anchors")
    for label in targets:
        idx = np.flatnonzero(eligible & (target == label))
        if len(idx) < n_per_group:
            raise ValueError(f"{label}: only {len(idx)} anchors, need {n_per_group}")
        result[label] = np.sort(rng.choice(idx, n_per_group, replace=False))
    return result


base.balanced_anchor_indices = all_nt_balanced_target_indices


if __name__ == "__main__":
    main()
