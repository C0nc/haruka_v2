"""Score cell-autonomous responders against state-matched NT controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ECM_GENES = ["Col1a1", "Col1a2", "Col3a1", "Col6a3", "Mmp2", "Dpt", "Tnc", "Thbs2", "Serpinf1"]
EPITHELIAL_GENES = ["Krt8", "Krt18", "Krt19", "Epcam", "Lcn2"]


def robust_deviation(value, reference):
    median = np.median(reference)
    mad = np.median(np.abs(reference - median))
    scale = max(1.4826 * mad, 1e-3)
    return float((value - median) / scale)


def leave_function_out_r2(x, y, functions):
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    prediction = np.zeros_like(y)
    for train, test in LeaveOneGroupOut().split(x, y, functions):
        prediction[test] = Ridge(alpha=10).fit(x[train], y[train]).predict(x[test])
    return float(r2_score(y, prediction))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--k_nt", type=int, default=25)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    root = Path(args.emb_dir)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    obs = adata.obs.copy()
    anchor = obs.is_anchor.to_numpy(bool)
    function_all = obs.function.astype(str).to_numpy()
    control_all = function_all == "NT"
    use = anchor
    function = function_all[use]
    guide = obs.guide_identity.astype(str).to_numpy()[use]
    slices = obs.slice_id.astype(str).to_numpy()[use]
    names = adata.obs_names.to_numpy()[use]
    control = control_all[use]
    background = np.load(root / "background.npy")[use]
    salient = np.load(root / "salient.npy")[use]
    salsp = np.load(root / "salient_spatial.npy")[use]

    background = StandardScaler().fit(background[control]).transform(background)
    salient = StandardScaler().fit(salient[control]).transform(salient)
    response_raw = np.zeros(len(function), dtype=float)
    response_robust = np.zeros(len(function), dtype=float)
    delta = np.zeros_like(salient)
    matched_n = np.zeros(len(function), dtype=int)
    for slice_id in np.unique(slices):
        pool = np.flatnonzero(control & (slices == slice_id))
        members = np.flatnonzero(slices == slice_id)
        if len(pool) < 3:
            raise RuntimeError(f"slice {slice_id} has only {len(pool)} NT anchors")
        k = min(args.k_nt + 1, len(pool))
        neighbor = NearestNeighbors(n_neighbors=k).fit(background[pool])
        for row in members:
            candidate = pool[neighbor.kneighbors(background[[row]], return_distance=False)[0]]
            candidate = candidate[candidate != row][:args.k_nt]
            if len(candidate) < 2:
                continue
            center = salient[candidate].mean(0)
            distances = np.linalg.norm(salient[candidate] - center, axis=1)
            distance = np.linalg.norm(salient[row] - center)
            response_raw[row] = distance
            response_robust[row] = robust_deviation(distance, distances)
            delta[row] = salient[row] - center
            matched_n[row] = len(candidate)

    response_percentile = np.zeros(len(function), dtype=float)
    for slice_id in np.unique(slices):
        null = response_robust[control & (slices == slice_id)]
        for row in np.flatnonzero(slices == slice_id):
            response_percentile[row] = (
                np.sum(null <= response_robust[row]) + 1
            ) / (len(null) + 1)

    coherence = np.full(len(function), np.nan)
    for target in sorted(set(function) - {"NT"}):
        index = np.flatnonzero(function == target)
        total = delta[index].sum(0)
        for row in index:
            direction = total - delta[row]
            denominator = np.linalg.norm(delta[row]) * np.linalg.norm(direction)
            coherence[row] = float(delta[row] @ direction / denominator) if denominator else 0.0

    # Independent RNA annotation of the general epithelial-to-ECM state.
    counts = adata.layers["counts"][use]
    totals = np.asarray(counts.sum(1)).ravel().clip(min=1)
    expression = counts.multiply(1e4 / totals[:, None])
    if sparse.issparse(expression):
        expression.data = np.log1p(expression.data)
        expression = expression.toarray()
    else:
        expression = np.log1p(expression)
    expression = StandardScaler().fit_transform(expression)
    gene_index = {str(gene): i for i, gene in enumerate(adata.var_names)}
    ecm = [gene_index[g] for g in ECM_GENES if g in gene_index]
    epithelial = [gene_index[g] for g in EPITHELIAL_GENES if g in gene_index]
    general_state = expression[:, ecm].mean(1) - expression[:, epithelial].mean(1)

    responder = (~control) & (response_percentile >= 0.95) & (coherence >= 0.0)
    frame = pd.DataFrame({
        "obs_name": names,
        "slice_id": slices,
        "function": function,
        "guide_identity": guide,
        "is_control": control,
        "matched_nt": matched_n,
        "intrinsic_response_distance": response_raw,
        "intrinsic_response_robust_z": response_robust,
        "intrinsic_response_percentile": response_percentile,
        "within_function_direction_coherence": coherence,
        "responder": responder,
        "ecm_minus_epithelial_score": general_state,
    })

    target_frame = frame[~frame.is_control].copy()
    def summarize(group):
        is_response = group.responder.to_numpy(bool)
        return pd.Series({
            "n": len(group),
            "responder_fraction": is_response.mean(),
            "median_response_percentile": group.intrinsic_response_percentile.median(),
            "median_response_robust_z": group.intrinsic_response_robust_z.median(),
            "median_coherence": group.within_function_direction_coherence.median(),
            "ecm_epithelial_all": group.ecm_minus_epithelial_score.mean(),
            "ecm_epithelial_responders": (
                group.loc[is_response, "ecm_minus_epithelial_score"].mean()
                if is_response.any() else np.nan
            ),
            "ecm_epithelial_nonresponders": (
                group.loc[~is_response, "ecm_minus_epithelial_score"].mean()
                if (~is_response).any() else np.nan
            ),
        })
    by_function = target_frame.groupby("function").apply(summarize, include_groups=False)
    by_guide = target_frame.groupby(["function", "guide_identity"]).apply(
        summarize, include_groups=False
    )
    function_target = function[~control]
    state = general_state[~control]
    target_reps = {
        "background": background[~control],
        "salient_intrinsic": salient[~control],
        "salient_spatial": salsp[~control],
    }
    result = {
        "score_definition": (
            "salient-intrinsic distance from same-slice NT anchors matched in "
            "background state; empirical percentile against the slice NT null"
        ),
        "responder_definition": "NT-null percentile >=0.95 and within-function coherence >=0",
        "control_false_positive_fraction": float(
            (response_percentile[control] >= 0.95).mean()
        ),
        "n_controls": int(control.sum()),
        "n_targets": int((~control).sum()),
        "ecm_epithelial_genes": {
            "ecm": [gene for gene in ECM_GENES if gene in gene_index],
            "epithelial": [gene for gene in EPITHELIAL_GENES if gene in gene_index],
        },
        "general_state_leave_function_out_R2": {
            name: leave_function_out_r2(x, state, function_target)
            for name, x in target_reps.items()
        },
        "response_state_spearman": {
            "rho": float(spearmanr(
                target_frame.intrinsic_response_percentile,
                target_frame.ecm_minus_epithelial_score,
            ).statistic),
            "pvalue": float(spearmanr(
                target_frame.intrinsic_response_percentile,
                target_frame.ecm_minus_epithelial_score,
            ).pvalue),
        },
        "by_function": by_function.reset_index().to_dict(orient="records"),
        "by_guide": by_guide.reset_index().to_dict(orient="records"),
    }
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prefix.with_suffix(".cells.csv"), index=False)
    prefix.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    by_function.to_csv(prefix.with_suffix(".by_function.csv"))
    by_guide.to_csv(prefix.with_suffix(".by_guide.csv"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
