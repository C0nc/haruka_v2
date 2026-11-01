#!/usr/bin/env python3
"""Test whether initial salient state predicts paired fibrosis fate.

The Haruka fit is Unaffected versus all fibrosis.  This analysis is strictly
post-hoc: within the five donors with paired Less/More Affected cores, it uses
background z-CC plus physical coordinates to fit Less -> More MOSCOT.  More-
affected s-CC supplies the target marginal.  Initial less-affected Alveolar
FBs are then scored for transport toward Activated Fibrotic FBs versus staying
Alveolar, and their initial z-CC/s-CC states are compared donor-held-out.
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
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from run_moscot_salient_marginal import fit_pair, positive_salient_mass


SOURCE_TYPE = "Alveolar FBs"
ACTIVATED_FATE = "Activated Fibrotic FBs"
STABLE_FATE = "Alveolar FBs"


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(pvalues)
    result = np.full(len(pvalues), np.nan)
    if not valid.any():
        return result
    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def donor_heldout_classifier(
    features: np.ndarray, labels: np.ndarray, donors: np.ndarray, name: str
) -> tuple[dict, np.ndarray]:
    predictions = np.full(len(labels), np.nan)
    for train, test in LeaveOneGroupOut().split(features, labels, donors):
        if len(np.unique(labels[train])) < 2:
            continue
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=0
        ).fit(scaler.transform(features[train]), labels[train])
        predictions[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    valid = np.isfinite(predictions)
    return (
        {
            "representation": name,
            "n_cells": int(valid.sum()),
            "n_donors": int(pd.Series(donors[valid]).nunique()),
            "roc_auc": roc_auc_score(labels[valid], predictions[valid]),
            "average_precision": average_precision_score(labels[valid], predictions[valid]),
            "balanced_accuracy": balanced_accuracy_score(
                labels[valid], predictions[valid] >= 0.5
            ),
            "validation": "leave-one-donor-out; within-donor fate quartiles",
        },
        predictions,
    )


def load_log_normalized(
    data_path: Path, row_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = ad.read_h5ad(data_path, backed="r")
    order = np.argsort(row_indices)
    sorted_rows = row_indices[order]
    block = data[sorted_rows].to_memory()
    matrix = block.layers["counts"] if "counts" in block.layers else block.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float64)
        library = np.asarray(matrix.sum(axis=1)).ravel()
        matrix = matrix.multiply((10_000.0 / np.maximum(library, 1.0))[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = np.log1p(matrix * (10_000.0 / np.maximum(matrix.sum(1), 1.0))[:, None])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return matrix[inverse].astype(np.float32), block.var_names.astype(str).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--haruka-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--marginal-strength", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--evidence-threshold", type=float, default=0.10)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = ad.read_h5ad(args.landmarks)
    fate_records: list[pd.DataFrame] = []
    fit_audit: list[dict] = []
    for donor in source.obs["donor"].astype(str).unique():
        donor_block = source[source.obs["donor"].astype(str) == donor]
        references = donor_block.obs.loc[
            donor_block.obs["condition"].astype(str) == "reference", "slice_id"
        ].astype(str).unique()
        if len(references) != 1:
            raise ValueError(f"Donor {donor} requires one Less Affected core")
        reference = references[0]
        targets = donor_block.obs.loc[
            donor_block.obs["condition"].astype(str) == "fibrosis", "slice_id"
        ].astype(str).unique()
        for target in targets:
            keep = donor_block.obs["slice_id"].astype(str).isin([reference, target])
            block = donor_block[keep].copy()
            block.obs["severity_time"] = np.where(
                block.obs["slice_id"].astype(str) == reference, 0, 1
            ).astype(int)
            source_mask = block.obs["severity_time"].to_numpy() == 0
            target_mask = ~source_mask
            _, target_mass = positive_salient_mass(
                np.asarray(block.obsm["X_s_cc"])[target_mask], args.marginal_strength
            )
            block.obs["source_mass"] = 1.0
            block.obs["target_mass"] = 1.0
            block.obs.loc[source_mask, "source_mass"] = 1.0 / source_mask.sum()
            block.obs.loc[target_mask, "target_mass"] = target_mass
            coupling, converged = fit_pair(
                block, weighted=True, alpha=args.alpha, epsilon=args.epsilon,
                rank=args.rank, device=args.device,
            )

            less = block.obs.loc[source_mask]
            more = block.obs.loc[target_mask]
            source_fb = less["cell_type"].astype(str).to_numpy() == SOURCE_TYPE
            destination = more["cell_type"].astype(str).to_numpy()
            row_mass = coupling[source_fb].sum(axis=1)
            p_activated = coupling[source_fb][:, destination == ACTIVATED_FATE].sum(axis=1)
            p_stable = coupling[source_fb][:, destination == STABLE_FATE].sum(axis=1)
            p_activated = np.divide(
                p_activated, row_mass, out=np.zeros_like(p_activated), where=row_mass > 0
            )
            p_stable = np.divide(
                p_stable, row_mass, out=np.zeros_like(p_stable), where=row_mass > 0
            )
            evidence = p_activated + p_stable
            fate_score = np.divide(
                p_activated, evidence, out=np.full_like(evidence, np.nan), where=evidence > 0
            )
            chosen = less.iloc[np.flatnonzero(source_fb)]
            fate_records.append(
                pd.DataFrame(
                    {
                        "obs_name": chosen.index.astype(str),
                        "source_index": chosen["source_index"].astype(int).to_numpy(),
                        "donor": donor,
                        "less_slice": reference,
                        "more_slice": target,
                        "p_activated_fibrotic": p_activated,
                        "p_alveolar": p_stable,
                        "terminal_evidence": evidence,
                        "activated_fate_score": fate_score,
                    }
                )
            )
            fit_audit.append(
                {
                    "donor": donor,
                    "less_slice": reference,
                    "more_slice": target,
                    "converged": converged,
                    "n_initial_alveolar_fb": int(source_fb.sum()),
                }
            )

    per_pair = pd.concat(fate_records, ignore_index=True)
    per_pair.to_csv(args.out_dir / "initial_alveolar_fb_fate_per_pair.csv.gz", index=False)
    # TILD117 has two More Affected cores.  Average its two future readouts so
    # every initial cell occurs only once in statistical tests.
    unique = (
        per_pair.groupby("obs_name", as_index=False, observed=True)
        .agg(
            source_index=("source_index", "first"),
            donor=("donor", "first"),
            less_slice=("less_slice", "first"),
            p_activated_fibrotic=("p_activated_fibrotic", "mean"),
            p_alveolar=("p_alveolar", "mean"),
            terminal_evidence=("terminal_evidence", "mean"),
            n_future_cores=("more_slice", "nunique"),
        )
    )
    unique["activated_fate_score"] = unique["p_activated_fibrotic"] / np.maximum(
        unique["p_activated_fibrotic"] + unique["p_alveolar"], 1e-12
    )
    unique = unique[unique["terminal_evidence"] >= args.evidence_threshold].copy()
    unique["fate_group"] = "middle"
    for donor, rows in unique.groupby("donor", observed=True).groups.items():
        values = unique.loc[rows, "activated_fate_score"]
        low, high = values.quantile([0.25, 0.75])
        unique.loc[rows[values <= low], "fate_group"] = "stable_alveolar"
        unique.loc[rows[values >= high], "fate_group"] = "fibrotic_primed"

    landmark_index = pd.Index(source.obs_names.astype(str))
    positions = landmark_index.get_indexer(unique["obs_name"])
    if (positions < 0).any():
        raise RuntimeError("Initial cells do not map back to landmark representations")
    z_cc = np.asarray(source.obsm["X_z_cc"])[positions]
    s_cc = np.asarray(source.obsm["X_s_cc"])[positions]
    extremes = unique["fate_group"].isin(["stable_alveolar", "fibrotic_primed"]).to_numpy()
    y = (unique.loc[extremes, "fate_group"] == "fibrotic_primed").astype(int).to_numpy()
    donors = unique.loc[extremes, "donor"].astype(str).to_numpy()
    feature_sets: list[tuple[str, np.ndarray]] = [
        ("z_cc", z_cc[extremes]),
        ("s_cc", s_cc[extremes]),
    ]
    if args.haruka_dir is not None:
        source_indices = unique.loc[extremes, "source_index"].astype(int).to_numpy()
        for name, filename in (
            ("z_intrinsic", "background.npy"),
            ("s_intrinsic", "salient.npy"),
            ("background_composition", "background_composition.npy"),
            ("salient_composition_shift", "salient_composition_shift.npy"),
        ):
            feature_sets.append(
                (name, np.asarray(np.load(args.haruka_dir / filename, mmap_mode="r")[source_indices]))
            )
    metrics: list[dict] = []
    for name, features in feature_sets:
        result, prediction = donor_heldout_classifier(features, y, donors, name)
        metrics.append(result)
        unique.loc[extremes, f"primed_probability_{name}"] = prediction
    pd.DataFrame(metrics).to_csv(args.out_dir / "initial_state_donor_heldout_metrics.csv", index=False)
    unique.to_csv(args.out_dir / "initial_alveolar_fb_fate_unique.csv.gz", index=False)

    expression, genes = load_log_normalized(
        args.data, unique.loc[extremes, "source_index"].astype(int).to_numpy()
    )
    extreme_obs = unique.loc[extremes].reset_index(drop=True)
    pseudobulk_records: list[dict] = []
    donor_deltas: list[np.ndarray] = []
    valid_donors: list[str] = []
    for donor in sorted(extreme_obs["donor"].astype(str).unique()):
        donor_mask = extreme_obs["donor"].astype(str).to_numpy() == donor
        stable = donor_mask & (extreme_obs["fate_group"].to_numpy() == "stable_alveolar")
        primed = donor_mask & (extreme_obs["fate_group"].to_numpy() == "fibrotic_primed")
        if not stable.any() or not primed.any():
            continue
        stable_mean = expression[stable].mean(axis=0)
        primed_mean = expression[primed].mean(axis=0)
        donor_deltas.append(primed_mean - stable_mean)
        valid_donors.append(donor)
        for group, values, n_cells in (
            ("stable_alveolar", stable_mean, int(stable.sum())),
            ("fibrotic_primed", primed_mean, int(primed.sum())),
        ):
            for gene, value in zip(genes, values):
                pseudobulk_records.append(
                    {"donor": donor, "fate_group": group, "gene": gene,
                     "mean_log_expression": value, "n_cells": n_cells}
                )
    delta = np.vstack(donor_deltas)
    gene_rows: list[dict] = []
    for j, gene in enumerate(genes):
        values = delta[:, j]
        statistic, pvalue = (
            wilcoxon(values) if np.any(values != 0) else (np.nan, np.nan)
        )
        gene_rows.append(
            {
                "gene": gene,
                "median_primed_minus_stable": float(np.median(values)),
                "mean_primed_minus_stable": float(np.mean(values)),
                "positive_donor_fraction": float(np.mean(values > 0)),
                "wilcoxon_statistic": statistic,
                "pvalue": pvalue,
                "n_donors": len(values),
            }
        )
    gene_table = pd.DataFrame(gene_rows)
    gene_table["fdr"] = bh_fdr(gene_table["pvalue"].to_numpy())
    gene_table = gene_table.sort_values("median_primed_minus_stable", ascending=False)
    gene_table.to_csv(args.out_dir / "initial_state_gene_programs.csv", index=False)
    pseudobulk = pd.DataFrame(pseudobulk_records)
    pseudobulk.to_csv(args.out_dir / "initial_state_expression_pseudobulk.csv.gz", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_frame = pd.DataFrame(metrics)
    axes[0].bar(
        metric_frame["representation"], metric_frame["roc_auc"],
        color=sns.color_palette("deep", len(metric_frame)),
    )
    axes[0].axhline(0.5, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Donor-held-out ROC AUC")
    axes[0].set_title("Initial-state prediction of future fate")
    axes[0].tick_params(axis="x", rotation=35)
    from sklearn.decomposition import PCA
    for ax, features, title in (
        (axes[1], z_cc, "Initial Alveolar FB z-CC"),
        (axes[2], s_cc, "Initial Alveolar FB s-CC"),
    ):
        xy = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(features))
        points = ax.scatter(
            xy[:, 0], xy[:, 1], c=unique["activated_fate_score"], cmap="magma",
            s=9, rasterized=True,
        )
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(points, ax=ax, fraction=0.045, label="Activated fate score")
    fig.suptitle("Less-affected Alveolar FB initial state and paired More-affected fate")
    fig.tight_layout()
    fig.savefig(args.out_dir / "initial_state_fate_prediction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    top_genes = gene_table.head(15)["gene"].tolist()
    bottom_genes = gene_table.tail(10)["gene"].tolist()
    selected = top_genes + bottom_genes
    matrix = (
        pseudobulk[pseudobulk["gene"].isin(selected)]
        .pivot_table(index="gene", columns=["donor", "fate_group"], values="mean_log_expression")
        .reindex(selected)
    )
    row_z = matrix.sub(matrix.mean(1), axis=0).div(matrix.std(1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(row_z, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "gene-wise z-score"})
    ax.set_title("Initial expression programs: fibrotic-primed vs stable Alveolar FBs")
    ax.set_xlabel("Donor and inferred future fate")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(args.out_dir / "initial_state_gene_program_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "haruka_training_contrast": "Unaffected control vs all fibrosis condition",
        "paired_analysis": "Less Affected to More Affected within donor",
        "source_cell_type": SOURCE_TYPE,
        "terminal_score": f"P({ACTIVATED_FATE}) / [P({ACTIVATED_FATE}) + P({STABLE_FATE})]",
        "evidence_threshold": args.evidence_threshold,
        "duplicate_source_handling": "average fate probabilities across future cores",
        "fate_groups": "within-donor bottom/top quartiles",
        "transport": {"alpha": args.alpha, "epsilon": args.epsilon, "rank": args.rank,
                      "marginal_strength": args.marginal_strength},
        "haruka_dir": str(args.haruka_dir) if args.haruka_dir is not None else None,
        "fit_audit": fit_audit,
        "n_unique_initial_cells": len(unique),
        "n_extreme_cells": int(extremes.sum()),
        "n_expression_donors": len(valid_donors),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"metadata": metadata, "metrics": metrics,
                      "top_genes": gene_table.head(10)[["gene", "median_primed_minus_stable", "positive_donor_fraction"]].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
