#!/usr/bin/env python3
"""Spatiotemporal MOSCOT with Haruka background geometry and salient masses.

For each paired Less Affected -> More Affected core:

* physical coordinates define the quadratic (spatial) FGW term;
* Haruka z-CellCharter defines the linear background-state term;
* the source marginal is uniform;
* the target marginal is a positive rank transform of ||s-CellCharter||.

A matched uniform-marginal fit is run as a sensitivity control. Published
annotations and s-CC clusters are used only after fitting to summarize plans.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from moscot.problems.spatiotemporal import SpatioTemporalProblem


TRANSITION_SPECS = {
    "lineage_to_lineage": ("lineage", "lineage"),
    "cell_type_to_cell_type": ("cell_type", "cell_type"),
    "CNiche_to_CNiche": ("CNiche", "CNiche"),
    "TNiche_to_TNiche": ("TNiche", "TNiche"),
    "cell_type_to_s_cc_cluster": ("cell_type", "s_cc_cluster"),
    "CNiche_to_s_cc_cluster": ("CNiche", "s_cc_cluster"),
    "TNiche_to_s_cc_cluster": ("TNiche", "s_cc_cluster"),
}


def aggregate_plan(
    coupling: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate a source-by-target coupling into probabilities/enrichments."""
    source_categories = np.unique(source_labels.astype(str))
    target_categories = np.unique(target_labels.astype(str))
    source_codes = pd.Categorical(source_labels.astype(str), categories=source_categories).codes
    target_codes = pd.Categorical(target_labels.astype(str), categories=target_categories).codes
    grouped = np.zeros((len(source_categories), len(target_categories)), dtype=np.float64)
    for source_code in range(len(source_categories)):
        target_mass = coupling[source_codes == source_code].sum(axis=0)
        np.add.at(grouped[source_code], target_codes, target_mass)
    row_sum = grouped.sum(axis=1, keepdims=True)
    probability = np.divide(grouped, row_sum, out=np.zeros_like(grouped), where=row_sum > 0)
    baseline = coupling.sum(axis=0)
    target_baseline = np.zeros(len(target_categories), dtype=np.float64)
    np.add.at(target_baseline, target_codes, baseline)
    target_baseline /= max(target_baseline.sum(), 1e-12)
    enrichment = np.log2((probability + 1e-8) / (target_baseline[None, :] + 1e-8))
    return (
        pd.DataFrame(probability, index=source_categories, columns=target_categories),
        pd.DataFrame(enrichment, index=source_categories, columns=target_categories),
    )


def average_matrices(matrices: list[pd.DataFrame], *, absent_target_is_zero: bool) -> pd.DataFrame:
    rows = sorted(set().union(*(set(x.index) for x in matrices)))
    columns = sorted(set().union(*(set(x.columns) for x in matrices)))
    aligned = []
    for matrix in matrices:
        if absent_target_is_zero:
            frame = matrix.reindex(index=rows, columns=columns, fill_value=0.0)
            frame.loc[pd.Index(rows).difference(matrix.index)] = np.nan
        else:
            frame = matrix.reindex(index=rows, columns=columns)
        aligned.append(frame)
    stacked = np.stack([x.to_numpy(float) for x in aligned])
    valid = np.sum(~np.isnan(stacked), axis=0)
    mean = np.divide(
        np.nansum(stacked, axis=0),
        valid,
        out=np.full(valid.shape, np.nan, dtype=np.float64),
        where=valid > 0,
    )
    return pd.DataFrame(mean, index=rows, columns=columns)


def positive_salient_mass(s_cc: np.ndarray, strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Map multidimensional s-CC to a robust, strictly positive cell marginal.

    The marginal is monotone in ||s-CC|| but rank-based, so one extreme cell
    cannot dominate OT. ``strength=0`` is uniform and ``strength=1`` spans
    approximately 0..2 before normalization. We cap strength below one to
    retain nonzero support for every target cell.
    """
    if not 0.0 <= strength < 1.0:
        raise ValueError("marginal strength must be in [0, 1)")
    response = np.linalg.norm(np.asarray(s_cc, dtype=np.float64), axis=1)
    ranks = pd.Series(response).rank(method="average", pct=True).to_numpy(float)
    mass = (1.0 - strength) + strength * 2.0 * ranks
    mass /= mass.sum()
    return response, mass


def fit_pair(
    block: ad.AnnData,
    *,
    weighted: bool,
    alpha: float,
    epsilon: float,
    rank: int,
    device: str,
) -> tuple[np.ndarray, bool]:
    problem = SpatioTemporalProblem(block)
    problem.prepare(
        time_key="severity_time",
        spatial_key="spatial",
        joint_attr="X_z_cc",
        normalize_spatial=True,
        policy="sequential",
        cost={"xy": "sq_euclidean", "x": "sq_euclidean", "y": "sq_euclidean"},
        a="source_mass" if weighted else False,
        b="target_mass" if weighted else False,
    )
    problem.solve(
        alpha=alpha,
        epsilon=epsilon,
        rank=rank,
        max_iterations=300,
        threshold=1e-3,
        device=device,
    )
    solution = problem.solutions[(0, 1)]
    coupling = np.asarray(solution.transport_matrix, dtype=np.float64)
    return coupling, bool(solution.converged)


def heatmap(matrix: pd.DataFrame, path: Path, title: str, *, difference: bool = False) -> None:
    width = max(9, 0.48 * matrix.shape[1] + 4)
    height = max(7, 0.30 * matrix.shape[0] + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix,
        cmap="vlag" if difference else "rocket",
        center=0.0 if difference else None,
        ax=ax,
        cbar_kws={"label": "weighted - uniform probability" if difference else "transition probability"},
    )
    ax.set_xlabel("More Affected destination state")
    ax.set_ylabel("Less Affected source state")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--marginal-strength", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = args.out_dir / "per_pair"
    pair_dir.mkdir(exist_ok=True)

    source = ad.read_h5ad(args.landmarks)
    pooled: dict[str, dict[str, list[pd.DataFrame]]] = {
        mode: {name: [] for name in TRANSITION_SPECS} for mode in ("uniform", "salient_weighted")
    }
    audit: list[dict[str, object]] = []
    marginal_rows: list[pd.DataFrame] = []

    for donor in source.obs["donor"].astype(str).unique():
        donor_block = source[source.obs["donor"].astype(str) == donor]
        references = donor_block.obs.loc[
            donor_block.obs["condition"].astype(str) == "reference", "slice_id"
        ].astype(str).unique()
        if len(references) != 1:
            raise ValueError(f"Donor {donor} requires exactly one Less Affected reference")
        reference = references[0]
        targets = donor_block.obs.loc[
            donor_block.obs["condition"].astype(str) == "fibrosis", "slice_id"
        ].astype(str).unique()
        for target in targets:
            mask = donor_block.obs["slice_id"].astype(str).isin([reference, target])
            block = donor_block[mask].copy()
            block.obs["severity_time"] = np.where(
                block.obs["slice_id"].astype(str) == reference, 0, 1
            ).astype(int)
            source_mask = block.obs["severity_time"].to_numpy() == 0
            target_mask = ~source_mask
            response, target_mass = positive_salient_mass(
                np.asarray(block.obsm["X_s_cc"])[target_mask], args.marginal_strength
            )
            block.obs["source_mass"] = 1.0
            block.obs["target_mass"] = 1.0
            block.obs.loc[source_mask, "source_mass"] = 1.0 / source_mask.sum()
            block.obs.loc[target_mask, "target_mass"] = target_mass
            pair_name = f"{donor}__{reference}_to_{target}"

            marginal_rows.append(
                pd.DataFrame(
                    {
                        "pair": pair_name,
                        "obs_name": block.obs_names[target_mask],
                        "response_norm_s_cc": response,
                        "target_marginal": target_mass,
                        "s_cc_cluster": block.obs.loc[target_mask, "s_cc_cluster"].astype(str).to_numpy(),
                        "cell_type": block.obs.loc[target_mask, "cell_type"].astype(str).to_numpy(),
                    }
                )
            )
            pair_audit: dict[str, object] = {
                "donor": donor,
                "less_slice": reference,
                "more_slice": target,
                "n_less_landmarks": int(source_mask.sum()),
                "n_more_landmarks": int(target_mask.sum()),
                "marginal_strength": args.marginal_strength,
                "target_mass_min": float(target_mass.min()),
                "target_mass_max": float(target_mass.max()),
                "target_mass_effective_n": float(1.0 / np.sum(target_mass**2)),
            }
            matrices_by_mode: dict[str, dict[str, pd.DataFrame]] = {}
            for mode, weighted in (("uniform", False), ("salient_weighted", True)):
                coupling, converged = fit_pair(
                    block,
                    weighted=weighted,
                    alpha=args.alpha,
                    epsilon=args.epsilon,
                    rank=args.rank,
                    device=args.device,
                )
                if coupling.shape != (int(source_mask.sum()), int(target_mask.sum())):
                    raise ValueError(f"Unexpected coupling shape {coupling.shape} for {pair_name}")
                pair_audit[f"{mode}_converged"] = converged
                matrices_by_mode[mode] = {}
                less_obs = block.obs.loc[source_mask]
                more_obs = block.obs.loc[target_mask]
                for name, (source_key, target_key) in TRANSITION_SPECS.items():
                    probability, enrichment = aggregate_plan(
                        coupling,
                        less_obs[source_key].astype(str).to_numpy(),
                        more_obs[target_key].astype(str).to_numpy(),
                    )
                    pooled[mode][name].append(probability)
                    matrices_by_mode[mode][name] = probability
                    probability.to_csv(pair_dir / f"{pair_name}__{mode}__{name}_probability.csv")
                    enrichment.to_csv(pair_dir / f"{pair_name}__{mode}__{name}_log2_enrichment.csv")
            for name in TRANSITION_SPECS:
                uniform, weighted = matrices_by_mode["uniform"][name].align(
                    matrices_by_mode["salient_weighted"][name], join="outer", fill_value=0.0
                )
                (weighted - uniform).to_csv(pair_dir / f"{pair_name}__salient_minus_uniform__{name}.csv")
            audit.append(pair_audit)

    pooled_dir = args.out_dir / "pooled"
    pooled_dir.mkdir(exist_ok=True)
    pooled_mean: dict[str, dict[str, pd.DataFrame]] = {mode: {} for mode in pooled}
    for mode in pooled:
        for name, matrices in pooled[mode].items():
            frame = average_matrices(matrices, absent_target_is_zero=True)
            pooled_mean[mode][name] = frame
            frame.to_csv(pooled_dir / f"{mode}__{name}_probability.csv")

    for name in TRANSITION_SPECS:
        uniform, weighted = pooled_mean["uniform"][name].align(
            pooled_mean["salient_weighted"][name], join="outer", fill_value=0.0
        )
        difference = weighted - uniform
        difference.to_csv(pooled_dir / f"salient_minus_uniform__{name}.csv")

    for name, label in (
        ("CNiche_to_CNiche", "cellular niche to cellular niche"),
        ("CNiche_to_s_cc_cluster", "background cellular niche to salient response state"),
        ("cell_type_to_s_cc_cluster", "background cell type to salient response state"),
    ):
        uniform, weighted = pooled_mean["uniform"][name].align(
            pooled_mean["salient_weighted"][name], join="outer", fill_value=0.0
        )
        heatmap(
            weighted,
            args.out_dir / f"salient_weighted__{name}.png",
            f"s-CC-marginal MOSCOT: {label}",
        )
        heatmap(
            weighted - uniform,
            args.out_dir / f"salient_minus_uniform__{name}.png",
            f"Effect of s-CC marginal: {label}",
            difference=True,
        )

    marginals = pd.concat(marginal_rows, ignore_index=True)
    marginals.to_csv(args.out_dir / "salient_target_marginals.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=marginals, x="pair", y="target_marginal", color="#7a5195", ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("More Affected cell masses derived from s-CellCharter response strength")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(args.out_dir / "salient_target_marginals.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "moscot_problem": "SpatioTemporalProblem",
        "time_key": "severity_time (Less Affected=0, More Affected=1)",
        "quadratic_spatial_term": "obsm['spatial']",
        "linear_background_term": "obsm['X_z_cc']",
        "source_marginal": "uniform",
        "target_marginal": "positive rank transform of L2 norm of obsm['X_s_cc']",
        "marginal_formula": "(1-strength) + 2*strength*percentile_rank(||s_cc||), normalized",
        "uniform_control": True,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "rank": args.rank,
        "audit": audit,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
