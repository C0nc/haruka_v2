#!/usr/bin/env python3
"""Infer less-to-more affected spatial state transitions from z-CC MOSCOT plans.

MOSCOT is fit with spatial coordinates and background z-CC only. Published
annotations and salient s-CC clusters are used strictly after fitting to
aggregate the cell-level transport plans.
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
from moscot.problems.space import AlignmentProblem


def aggregate_plan(
    coupling_less_more: np.ndarray,
    less_labels: np.ndarray,
    more_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-stochastic transition probabilities and target enrichment."""
    source_categories = np.unique(less_labels.astype(str))
    target_categories = np.unique(more_labels.astype(str))
    source_lookup = {value: i for i, value in enumerate(source_categories)}
    target_lookup = {value: i for i, value in enumerate(target_categories)}
    source_codes = np.asarray([source_lookup[x] for x in less_labels.astype(str)])
    target_codes = np.asarray([target_lookup[x] for x in more_labels.astype(str)])
    grouped = np.zeros((len(source_categories), len(target_categories)), dtype=np.float64)
    for source_code in range(len(source_categories)):
        source_rows = source_codes == source_code
        mass_to_cells = coupling_less_more[source_rows].sum(axis=0)
        np.add.at(grouped[source_code], target_codes, mass_to_cells)
    row_sum = grouped.sum(axis=1, keepdims=True)
    probability = np.divide(grouped, row_sum, out=np.zeros_like(grouped), where=row_sum > 0)
    target_baseline = coupling_less_more.sum(axis=0)
    target_group_mass = np.zeros(len(target_categories), dtype=np.float64)
    np.add.at(target_group_mass, target_codes, target_baseline)
    target_group_mass /= max(target_group_mass.sum(), 1e-12)
    enrichment = np.log2((probability + 1e-8) / (target_group_mass[None, :] + 1e-8))
    return (
        pd.DataFrame(probability, index=source_categories, columns=target_categories),
        pd.DataFrame(enrichment, index=source_categories, columns=target_categories),
    )


def average_matrices(
    matrices: list[pd.DataFrame], *, absent_target_is_zero: bool
) -> pd.DataFrame:
    rows = sorted(set().union(*(set(x.index) for x in matrices)))
    columns = sorted(set().union(*(set(x.columns) for x in matrices)))
    aligned = []
    for matrix in matrices:
        if absent_target_is_zero:
            frame = matrix.reindex(index=rows, columns=columns, fill_value=0.0)
            missing_source = pd.Index(rows).difference(matrix.index)
            frame.loc[missing_source] = np.nan
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


def save_long(matrix: pd.DataFrame, path: Path, value_name: str) -> None:
    matrix.rename_axis("less_state").rename_axis("more_state", axis=1).stack(
        future_stack=True
    ).rename(value_name).reset_index().to_csv(path, index=False)


def top_transitions(probability: pd.DataFrame, enrichment: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    rows = []
    for source in probability.index:
        order = probability.loc[source].sort_values(ascending=False).head(n)
        for rank, (target, value) in enumerate(order.items(), start=1):
            rows.append(
                {
                    "less_state": source,
                    "rank": rank,
                    "more_state": target,
                    "probability": value,
                    "log2_enrichment_over_target_frequency": enrichment.loc[source, target],
                }
            )
    return pd.DataFrame(rows)


def heatmap(matrix: pd.DataFrame, path: Path, title: str, cmap: str = "mako", center=None) -> None:
    width = max(9, 0.48 * matrix.shape[1] + 4)
    height = max(7, 0.30 * matrix.shape[0] + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(matrix, cmap=cmap, center=center, ax=ax, cbar_kws={"label": "transition probability"})
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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=-1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = args.out_dir / "per_pair"
    pair_dir.mkdir(exist_ok=True)

    source = ad.read_h5ad(args.landmarks)
    transition_specs = {
        "lineage_to_lineage": ("lineage", "lineage"),
        "cell_type_to_cell_type": ("cell_type", "cell_type"),
        "CNiche_to_CNiche": ("CNiche", "CNiche"),
        "TNiche_to_TNiche": ("TNiche", "TNiche"),
        "cell_type_to_s_cc_cluster": ("cell_type", "s_cc_cluster"),
        "CNiche_to_s_cc_cluster": ("CNiche", "s_cc_cluster"),
        "TNiche_to_s_cc_cluster": ("TNiche", "s_cc_cluster"),
    }
    pooled_probabilities: dict[str, list[pd.DataFrame]] = {key: [] for key in transition_specs}
    pooled_enrichments: dict[str, list[pd.DataFrame]] = {key: [] for key in transition_specs}
    audit = []

    for donor in source.obs["donor"].astype(str).unique():
        block = source[source.obs["donor"].astype(str) == donor].copy()
        reference_slices = block.obs.loc[
            block.obs["condition"].astype(str) == "reference", "slice_id"
        ].astype(str).unique()
        if len(reference_slices) != 1:
            raise ValueError(f"Donor {donor} requires one Less Affected reference")
        reference = reference_slices[0]
        problem = AlignmentProblem(block)
        problem.prepare(
            batch_key="slice_id",
            spatial_key="spatial",
            joint_attr="X_z_cc",
            policy="star",
            reference=reference,
            normalize_spatial=True,
            cost={"xy": "sq_euclidean", "x": "sq_euclidean", "y": "sq_euclidean"},
        )
        problem.solve(
            alpha=args.alpha,
            epsilon=args.epsilon,
            rank=args.rank,
            max_iterations=300,
            threshold=1e-3,
            device="gpu",
        )
        for more_slice in block.obs.loc[
            block.obs["condition"].astype(str) == "fibrosis", "slice_id"
        ].astype(str).unique():
            key = (more_slice, reference)
            solution = problem.solutions[key]
            # Star policy solves More -> Less. Transpose for requested Less -> More.
            coupling_less_more = np.asarray(solution.transport_matrix, dtype=np.float64).T
            less = block.obs[block.obs["slice_id"].astype(str) == reference]
            more = block.obs[block.obs["slice_id"].astype(str) == more_slice]
            if coupling_less_more.shape != (len(less), len(more)):
                raise ValueError(f"Unexpected coupling shape for {key}: {coupling_less_more.shape}")
            pair_name = f"{donor}__{reference}_to_{more_slice}"
            for transition_name, (less_key, more_key) in transition_specs.items():
                probability, enrichment = aggregate_plan(
                    coupling_less_more,
                    less[less_key].astype(str).to_numpy(),
                    more[more_key].astype(str).to_numpy(),
                )
                pooled_probabilities[transition_name].append(probability)
                pooled_enrichments[transition_name].append(enrichment)
                save_long(
                    probability,
                    pair_dir / f"{pair_name}__{transition_name}_probability.csv",
                    "probability",
                )
            audit.append(
                {
                    "donor": donor,
                    "less_slice": reference,
                    "more_slice": more_slice,
                    "n_less_landmarks": len(less),
                    "n_more_landmarks": len(more),
                    "solver_converged": bool(solution.converged),
                    "transport_orientation": "transposed More->Less star solution; row-normalized Less->More",
                }
            )

    pooled_dir = args.out_dir / "pooled"
    pooled_dir.mkdir(exist_ok=True)
    pooled = {}
    for name in transition_specs:
        probability = average_matrices(
            pooled_probabilities[name], absent_target_is_zero=True
        )
        enrichment = average_matrices(
            pooled_enrichments[name], absent_target_is_zero=False
        )
        pooled[name] = probability
        probability.to_csv(pooled_dir / f"{name}_probability.csv")
        enrichment.to_csv(pooled_dir / f"{name}_log2_enrichment.csv")
        top_transitions(probability, enrichment).to_csv(
            pooled_dir / f"{name}_top_transitions.csv", index=False
        )

    heatmap(
        pooled["CNiche_to_CNiche"],
        args.out_dir / "less_to_more_CNiche_transition.png",
        "GSE250346 MOSCOT: Less Affected to More Affected cellular-niche transition",
    )
    heatmap(
        pooled["TNiche_to_TNiche"],
        args.out_dir / "less_to_more_TNiche_transition.png",
        "GSE250346 MOSCOT: Less Affected to More Affected transcriptional-niche transition",
    )
    heatmap(
        pooled["CNiche_to_s_cc_cluster"],
        args.out_dir / "less_CNiche_to_more_scc_transition.png",
        "Background cellular niche to salient fibrosis-response state",
        cmap="rocket",
    )
    cell_to_s = pooled["cell_type_to_s_cc_cluster"]
    abundance = source.obs["cell_type"].astype(str).value_counts()
    top_cell_types = [x for x in abundance.index if x in cell_to_s.index][:25]
    heatmap(
        cell_to_s.loc[top_cell_types],
        args.out_dir / "less_celltype_to_more_scc_transition.png",
        "Background cell type to salient fibrosis-response state (top 25 cell types)",
        cmap="rocket",
    )
    heatmap(
        pooled["lineage_to_lineage"],
        args.out_dir / "less_to_more_lineage_transition.png",
        "GSE250346 MOSCOT: Less Affected to More Affected lineage correspondence",
    )
    (args.out_dir / "transition_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
