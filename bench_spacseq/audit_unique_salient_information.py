"""Cross-fitted residual audit of unique intrinsic and spatial information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from score_multislice_all_guides import leave_slice_out_ba, random_cv_ba
from score_functional_guides import within_function_guide_ba


def cross_fitted_residual(target, nuisance, seed=0):
    target = StandardScaler().fit_transform(target)
    nuisance = StandardScaler().fit_transform(nuisance)
    prediction = np.zeros_like(target)
    for train, test in KFold(5, shuffle=True, random_state=seed).split(target):
        model = Ridge(alpha=10.0).fit(nuisance[train], target[train])
        prediction[test] = model.predict(nuisance[test])
    return target - prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.data)
    root = Path(args.emb_dir)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    salient = np.load(root / "salient.npy")
    spatial = np.load(root / "salient_spatial.npy")
    reps = {
        "intrinsic_raw": salient,
        "spatial_raw": spatial,
        "intrinsic_unique_after_spatial": cross_fitted_residual(
            salient, spatial, args.seed
        ),
        "spatial_unique_after_intrinsic": cross_fitted_residual(
            spatial, salient, args.seed
        ),
        "intrinsic_plus_spatial": np.column_stack([salient, spatial]),
    }
    obs = adata.obs
    function = obs.function.astype(str).to_numpy()
    guide = obs.guide_identity.astype(str).to_numpy()
    slices = obs.slice_id.astype(str).to_numpy()
    anchor = obs.is_anchor.to_numpy(bool)
    target_anchor = anchor & (function != "NT")
    known_niche = obs.niche.astype(str).to_numpy() != "Unknown"
    result = {"representations": {}}
    for name, x in reps.items():
        function_loso, _ = leave_slice_out_ba(
            x[target_anchor], function[target_anchor], slices[target_anchor]
        )
        cell_loso, _ = leave_slice_out_ba(
            x, obs.cell_type.astype(str).to_numpy(), slices
        )
        niche_loso, _ = leave_slice_out_ba(
            x[known_niche], obs.niche.astype(str).to_numpy()[known_niche],
            slices[known_niche]
        )
        condition_loso, _ = leave_slice_out_ba(
            x, obs.condition.astype(str).to_numpy(), slices
        )
        result["representations"][name] = {
            "width": int(x.shape[1]),
            "function_cv_ba": random_cv_ba(
                x[target_anchor], function[target_anchor], args.seed
            ),
            "function_leave_slice_out_ba": function_loso,
            "within_function_guide_identity_ba": within_function_guide_ba(
                x[target_anchor], function[target_anchor], guide[target_anchor],
                args.seed
            )["mean"],
            "cell_type_leave_slice_out_ba": cell_loso,
            "niche_leave_slice_out_ba": niche_loso,
            "condition_leave_slice_out_ba": condition_loso,
            "slice_cv_ba": random_cv_ba(x, slices, args.seed),
        }
    output = Path(args.out)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
