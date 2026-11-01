#!/usr/bin/env python3
"""Background-guided initial-state associations for KRASi residual tumor fates.

This mirrors the Haruka preprint's initial-state/fate logic, but the experiment
has no animal-matched control/treated sections.  Results are therefore unpaired
cross-section associations and must not be described as lineage transitions.
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
from pynndescent import NNDescent
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TUMOR_TYPES = ["Tumor_epithelial", "Tumor_epithelial_Itgb6", "Tumor_EMT"]


def sample_per_slice(obs: pd.DataFrame, mask: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    slices = obs["slice_id"].astype(str).to_numpy()
    parts: list[np.ndarray] = []
    for sid in sorted(np.unique(slices[mask])):
        rows = np.flatnonzero(mask & (slices == sid))
        if len(rows) > n:
            rows = rng.choice(rows, n, replace=False)
        parts.append(np.sort(rows))
    return np.concatenate(parts)


def load_log_expression(data: ad.AnnData, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(rows)
    block = data[rows[order]].to_memory()
    matrix = block.layers["counts"] if "counts" in block.layers else block.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float32)
        lib = np.asarray(matrix.sum(1)).ravel()
        matrix = matrix.multiply((10_000.0 / np.maximum(lib, 1.0))[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
        matrix = np.log1p(matrix * (10_000.0 / np.maximum(matrix.sum(1), 1.0))[:, None])
    inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
    return matrix[inverse], block.var_names.astype(str).to_numpy()


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]; order = np.argsort(pv); ranked = pv[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    restored = np.empty_like(adj); restored[order] = np.minimum(adj, 1)
    out[valid] = restored
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cc-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-slice", type=int, default=6000)
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--salient-marginal-strength", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fig_dir = args.out_dir / "figures"; table_dir = args.out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); table_dir.mkdir(parents=True, exist_ok=True)

    data = ad.read_h5ad(args.data, backed="r")
    obs = data.obs.copy()
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(names, obs.index.astype(str).to_numpy()):
        raise ValueError("CellCharter rows do not align with AnnData")
    z_cc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    s_cc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    condition = obs["condition_label"].astype(str).to_numpy()
    celltype = obs["celltype"].astype(str).to_numpy()
    tumor = np.isin(celltype, TUMOR_TYPES)
    control_rows = sample_per_slice(obs, tumor & (condition == "Control"), args.per_slice, args.seed)
    treated_rows = sample_per_slice(obs, tumor & (condition == "KRASi_treated"), args.per_slice, args.seed + 1)

    scaler = StandardScaler().fit(np.asarray(z_cc[np.r_[control_rows, treated_rows]]))
    control_z = scaler.transform(np.asarray(z_cc[control_rows])).astype(np.float32)
    treated_z = scaler.transform(np.asarray(z_cc[treated_rows])).astype(np.float32)
    index = NNDescent(treated_z, n_neighbors=max(45, args.neighbors), metric="euclidean",
                      random_state=args.seed, n_jobs=-1)
    neighbor_index, distances = index.query(control_z, k=args.neighbors)
    target_types = celltype[treated_rows][neighbor_index]
    scale = max(float(np.median(distances[distances > 0])), 1e-6)
    weights = np.exp(-distances / scale)
    target_salient = np.linalg.norm(np.asarray(s_cc[treated_rows]), axis=1)
    lo, hi = np.quantile(target_salient, [0.05, 0.95])
    marginal = np.clip((target_salient - lo) / max(hi - lo, 1e-6), 0, 1)
    weights *= 1.0 + args.salient_marginal_strength * marginal[neighbor_index]
    probabilities = {}
    for fate in TUMOR_TYPES:
        probabilities[fate] = (weights * (target_types == fate)).sum(1) / np.maximum(weights.sum(1), 1e-12)
    evidence = probabilities["Tumor_epithelial"] + probabilities["Tumor_epithelial_Itgb6"]
    itgb6_score = probabilities["Tumor_epithelial_Itgb6"] / np.maximum(evidence, 1e-12)

    result = obs.iloc[control_rows][["slice_id", "celltype", "compartment"]].copy()
    result.insert(0, "source_index", control_rows)
    for key, values in probabilities.items(): result[f"p_{key}"] = values
    result["canonical_itgb6_evidence"] = evidence
    result["itgb6_residual_fate_score"] = itgb6_score
    result["initial_group"] = "middle"
    for sid, rows in result.groupby("slice_id", observed=True).groups.items():
        values = result.loc[rows, "itgb6_residual_fate_score"]
        low, high = values.quantile([0.25, 0.75])
        result.loc[rows[values <= low], "initial_group"] = "canonical_associated"
        result.loc[rows[values >= high], "initial_group"] = "itgb6_residual_associated"
    result.to_csv(table_dir / "control_initial_state_associations.csv.gz", index=False, compression="gzip")

    extremes = result["initial_group"].isin(["canonical_associated", "itgb6_residual_associated"]).to_numpy()
    extreme_rows = control_rows[extremes]
    expression, genes = load_log_expression(data, extreme_rows)
    extreme_obs = result.loc[extremes].reset_index(drop=True)
    deltas = []
    slice_ids = []
    pseudobulk_records = []
    for sid in sorted(extreme_obs["slice_id"].astype(str).unique()):
        block = extreme_obs["slice_id"].astype(str).to_numpy() == sid
        low = block & (extreme_obs["initial_group"].to_numpy() == "canonical_associated")
        high = block & (extreme_obs["initial_group"].to_numpy() == "itgb6_residual_associated")
        if not low.any() or not high.any(): continue
        low_mean = expression[low].mean(0); high_mean = expression[high].mean(0)
        deltas.append(high_mean - low_mean); slice_ids.append(sid)
        for group, values, n_cells in [("canonical_associated", low_mean, int(low.sum())),
                                        ("itgb6_residual_associated", high_mean, int(high.sum()))]:
            pseudobulk_records.extend({"slice_id": sid, "initial_group": group, "gene": gene,
                                       "mean_log_expression": float(value), "n_cells": n_cells}
                                      for gene, value in zip(genes, values))
    delta = np.vstack(deltas)
    records = []
    for j, gene in enumerate(genes):
        values = delta[:, j]
        stat, pvalue = wilcoxon(values) if np.any(values != 0) else (np.nan, np.nan)
        records.append({"gene": gene, "mean_itgb6_minus_canonical": float(values.mean()),
                        "median_itgb6_minus_canonical": float(np.median(values)),
                        "positive_section_fraction": float(np.mean(values > 0)),
                        "wilcoxon_statistic": stat, "pvalue": pvalue, "n_control_sections": len(values)})
    gene_table = pd.DataFrame(records)
    gene_table["fdr"] = bh_fdr(gene_table["pvalue"].to_numpy())
    gene_table = gene_table.sort_values("mean_itgb6_minus_canonical", ascending=False)
    gene_table.to_csv(table_dir / "initial_state_gene_programs.csv", index=False)
    pseudobulk = pd.DataFrame(pseudobulk_records)
    pseudobulk.to_csv(table_dir / "initial_state_expression_pseudobulk.csv.gz", index=False, compression="gzip")

    source_xy = PCA(n_components=2, random_state=args.seed).fit_transform(control_z)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    points = axes[0].scatter(source_xy[:, 0], source_xy[:, 1], c=itgb6_score, cmap="magma",
                             s=2, linewidths=0, rasterized=True)
    axes[0].set_title("Control tumor background state\ncolored by Itgb6-residual association")
    axes[0].set_xticks([]); axes[0].set_yticks([]); fig.colorbar(points, ax=axes[0], fraction=0.045)
    transition = pd.DataFrame({"source": celltype[control_rows]})
    for fate in TUMOR_TYPES: transition[fate] = probabilities[fate]
    transition_mean = transition.groupby("source", observed=True)[TUMOR_TYPES].mean()
    sns.heatmap(transition_mean, cmap="viridis", annot=True, fmt=".2f", ax=axes[1],
                cbar_kws={"label": "weighted association probability"})
    axes[1].set_title("Control state → treated residual state")
    top = gene_table.head(12)["gene"].tolist() + gene_table.tail(8)["gene"].tolist()
    matrix = (pseudobulk[pseudobulk["gene"].isin(top)]
              .pivot_table(index="gene", columns=["slice_id", "initial_group"], values="mean_log_expression")
              .reindex(top))
    row_z = matrix.sub(matrix.mean(1), axis=0).div(matrix.std(1).replace(0, 1), axis=0)
    sns.heatmap(row_z, cmap="vlag", center=0, ax=axes[2], cbar_kws={"label": "gene-wise z-score"})
    axes[2].set_title("Control initial-state expression programs")
    fig.suptitle("Background-guided KRASi residual-state association (unpaired)")
    fig.tight_layout(); fig.savefig(fig_dir / "initial_state_residual_fate_summary.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    coords = np.asarray(data.obsm["spatial"])
    control_slices = sorted(result["slice_id"].astype(str).unique())
    fig, axes = plt.subplots(1, len(control_slices), figsize=(6 * len(control_slices), 5), squeeze=False)
    for i, sid in enumerate(control_slices):
        take = result["slice_id"].astype(str).to_numpy() == sid
        rows = control_rows[take]
        points = axes[0, i].scatter(coords[rows, 0], coords[rows, 1], c=itgb6_score[take], cmap="magma",
                                    vmin=0, vmax=1, s=3, linewidths=0, rasterized=True)
        axes[0, i].set_title(f"{sid}: inferred initial association"); axes[0, i].set_aspect("equal"); axes[0, i].axis("off")
    fig.colorbar(points, ax=axes.ravel().tolist(), fraction=0.018, label="Itgb6 residual-state association")
    fig.savefig(fig_dir / "control_spatial_initial_state_score.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    transition_mean.to_csv(table_dir / "control_to_treated_tumor_state_matrix.csv")
    metadata = {
        "analysis": "preprint-style initial-state to fate association",
        "joint_space": "background z-CC",
        "target_marginal": f"treated s-CC magnitude, strength={args.salient_marginal_strength}",
        "matching": f"approximate {args.neighbors}-nearest treated tumor states per control tumor cell",
        "source": "control tumor cells", "terminal_states": TUMOR_TYPES,
        "critical_limitation": "control and treated sections are unpaired and each arm is one Xenium run; this is association, not lineage, fate, or causal transition",
        "n_control_tumor_sample": len(control_rows), "n_treated_tumor_sample": len(treated_rows),
        "control_sections": sorted(obs.iloc[control_rows]["slice_id"].astype(str).unique().tolist()),
        "treated_sections": sorted(obs.iloc[treated_rows]["slice_id"].astype(str).unique().tolist()),
    }
    (args.out_dir / "initial_state_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"metadata": metadata,
                      "top_initial_genes": gene_table.head(15)[["gene", "mean_itgb6_minus_canonical", "positive_section_fraction"]].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
