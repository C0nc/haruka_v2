#!/usr/bin/env python3
"""Initial molecular correlates of Haruka/CellRank terminal-fate allocation.

For each AD replicate and main biological cluster, the earliest 30% of cells by
s-CC DPT are selected.  Feature expression is associated with the continuous
CellRank probability of reaching the plaque-proximal terminal state.  Groupwise
Spearman correlations are combined by a fixed-effect Fisher-z meta-analysis.

This is an exploratory predictive association analysis, not a causal test.
The same molecular data contributed to Haruka training, so effect sizes should
be validated on independent samples before biological interpretation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, rankdata, t as student_t


ROOT = Path("/home/yancui/haruka_v2")
RESULTS = ROOT / "ad_analysis/scc_dpt_cellrank2"
DATA = {
    "starmap": ROOT / "bench_real_discovery/data/ad.h5ad",
    "codex": ROOT / (
        "bench_codex_cns_2026/data/processed/"
        "codex_cns_myeloid_primary_no_aml_no_combat.h5ad"
    ),
}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    values = p[ok]
    order = np.argsort(values)
    ranked = values[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.minimum(q, 1.0)
    out[ok] = restored
    return out


def vectorized_spearman(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ry = rankdata(y).astype(float)
    ry -= ry.mean()
    rx = rankdata(x, axis=0).astype(float)
    rx -= rx.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.square(rx).sum(axis=0) * np.square(ry).sum())
    rho = np.divide(rx.T @ ry, denom, out=np.full(x.shape[1], np.nan), where=denom > 0)
    rho = np.clip(rho, -0.999999, 0.999999)
    dof = max(len(y) - 2, 1)
    stat = rho * np.sqrt(dof / np.maximum(1.0 - np.square(rho), 1e-12))
    p = 2 * student_t.sf(np.abs(stat), dof)
    return rho, p


def group_associations(dataset: str, scores: pd.DataFrame, early_fraction: float) -> pd.DataFrame:
    a = ad.read_h5ad(DATA[dataset], backed="r")
    index = pd.Series(np.arange(a.n_obs), index=a.obs_names.astype(str))
    features = np.asarray(a.var_names.astype(str))
    rows = []
    for (replicate, cluster), group in scores.groupby(["replicate", "main_cluster"], sort=True):
        cutoff = group.dpt_pseudotime.quantile(early_fraction)
        early = group[group.dpt_pseudotime <= cutoff].copy()
        early = early[early.cell_id.astype(str).isin(index.index)]
        if len(early) < 30:
            continue
        pos = index.loc[early.cell_id.astype(str)].to_numpy(dtype=int)
        order = np.argsort(pos)
        raw = a[pos[order], :].X
        x_sorted = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        x = np.asarray(x_sorted[inverse], dtype=float)
        y = early.plaque_fate_probability.to_numpy(dtype=float)
        rho, p = vectorized_spearman(x, y)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "replicate": str(replicate),
                    "main_cluster": str(cluster),
                    "feature": features,
                    "n_early_cells": len(early),
                    "early_dpt_cutoff": float(cutoff),
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )
        )
    a.file.close()
    out = pd.concat(rows, ignore_index=True)
    out["q_value_within_group"] = out.groupby(["replicate", "main_cluster"], sort=False)["p_value"].transform(
        lambda x: bh_fdr(x.to_numpy())
    )
    return out


def meta_analyze(groupwise: pd.DataFrame) -> pd.DataFrame:
    records = []
    for feature, frame in groupwise.groupby("feature", sort=False):
        valid = frame[np.isfinite(frame.spearman_rho)].copy()
        if valid.empty:
            continue
        z = np.arctanh(np.clip(valid.spearman_rho.to_numpy(), -0.999999, 0.999999))
        weight = np.maximum(valid.n_early_cells.to_numpy(dtype=float) - 3.0, 1.0)
        pooled_z = float(np.sum(weight * z) / np.sum(weight))
        test_z = pooled_z * np.sqrt(np.sum(weight))
        records.append(
            {
                "dataset": valid.dataset.iloc[0],
                "feature": feature,
                "meta_spearman_rho": float(np.tanh(pooled_z)),
                "meta_p_value": float(2 * norm.sf(abs(test_z))),
                "n_groups": len(valid),
                "n_early_cells": int(valid.n_early_cells.sum()),
                "positive_group_fraction": float((valid.spearman_rho > 0).mean()),
                "median_group_rho": float(valid.spearman_rho.median()),
            }
        )
    out = pd.DataFrame(records)
    out["meta_q_value"] = bh_fdr(out.meta_p_value.to_numpy())
    out["terminal_association"] = np.where(
        out.meta_spearman_rho >= 0, "Plaque-proximal terminal", "Alternative terminal"
    )
    return out.sort_values("meta_spearman_rho", ascending=False).reset_index(drop=True)


def plot_meta(dataset: str, meta: pd.DataFrame, out_dir: Path):
    if dataset == "codex":
        selected = meta.sort_values("meta_spearman_rho").copy()
        height = 6.5
    else:
        positive = meta.nlargest(15, "meta_spearman_rho")
        negative = meta.nsmallest(15, "meta_spearman_rho")
        selected = pd.concat([negative, positive]).drop_duplicates("feature").sort_values("meta_spearman_rho")
        height = 10
    colors = np.where(selected.meta_spearman_rho >= 0, "#20A6B8", "#61B15A")
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(selected.feature, selected.meta_spearman_rho, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Meta-analyzed Spearman ρ in early-pseudotime cells")
    ax.set_ylabel("")
    ax.set_title(
        f"{dataset.upper()}: initial molecular correlates of CellRank terminal allocation\n"
        "positive → plaque-proximal fate; negative → alternative fate"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_initial_molecular_terminal_predictors.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_group_heatmap(dataset: str, groupwise: pd.DataFrame, meta: pd.DataFrame, out_dir: Path):
    if dataset == "codex":
        features = meta.feature.tolist()
    else:
        features = pd.concat([meta.nlargest(10, "meta_spearman_rho"), meta.nsmallest(10, "meta_spearman_rho")]).feature.tolist()
    sub = groupwise[groupwise.feature.isin(features)].copy()
    sub["group"] = sub.replicate.astype(str) + " | " + sub.main_cluster.astype(str)
    matrix = sub.pivot(index="feature", columns="group", values="spearman_rho").reindex(features)
    fig, ax = plt.subplots(figsize=(max(14, 0.55 * matrix.shape[1]), max(6, 0.42 * matrix.shape[0])))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-0.6, vmax=0.6, ax=ax, cbar_kws={"label": "Early-cell Spearman ρ"})
    ax.set_xlabel("Replicate | main cluster")
    ax.set_ylabel("")
    ax.set_title(f"{dataset.upper()}: replicate- and cluster-resolved initial molecular associations")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_initial_molecular_terminal_predictor_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=RESULTS)
    parser.add_argument("--early-fraction", type=float, default=0.30)
    args = parser.parse_args()
    scores = pd.read_csv(args.root / "cellrank_cell_scores.csv.gz", dtype={"cell_id": str, "main_cluster": str})
    all_groupwise, all_meta = [], []
    for dataset in ["starmap", "codex"]:
        print(f"Analyzing {dataset}", flush=True)
        groupwise = group_associations(dataset, scores[scores.dataset == dataset], args.early_fraction)
        meta = meta_analyze(groupwise)
        groupwise.to_csv(args.root / f"{dataset}_initial_molecular_terminal_associations_by_group.csv.gz", index=False)
        meta.to_csv(args.root / f"{dataset}_initial_molecular_terminal_associations_meta.csv", index=False)
        top = pd.concat([meta.nlargest(10, "meta_spearman_rho"), meta.nsmallest(10, "meta_spearman_rho")])
        top.to_csv(args.root / f"{dataset}_top_initial_molecular_terminal_predictors.csv", index=False)
        plot_meta(dataset, meta, args.root)
        plot_group_heatmap(dataset, groupwise, meta, args.root)
        all_groupwise.append(groupwise)
        all_meta.append(meta)
    pd.concat(all_meta, ignore_index=True).to_csv(args.root / "both_ad_initial_molecular_terminal_associations_meta.csv", index=False)
    print(f"Saved initial molecular terminal-fate analysis to {args.root}", flush=True)


if __name__ == "__main__":
    main()
