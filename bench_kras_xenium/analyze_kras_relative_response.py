#!/usr/bin/env python3
"""Reproducible treated-tumor relative response analysis for Haruka z-CC/s-CC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from umap import UMAP


TUMOR_TYPES = {"Tumor_epithelial", "Tumor_epithelial_Itgb6", "Tumor_EMT"}


def grouped_percentile(values: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    temp = frame[["slice_id", "celltype"]].copy()
    temp["value"] = values
    return temp.groupby(["slice_id", "celltype"], observed=True)["value"].rank(pct=True).to_numpy()


def leave_section_out(x: np.ndarray, y: np.ndarray, sections: np.ndarray, name: str):
    pred = np.full(len(y), np.nan, dtype=np.float32)
    rows = []
    for section in sorted(np.unique(sections)):
        train, test = sections != section, sections == section
        if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
            continue
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, class_weight="balanced"))
        model.fit(x[train], y[train])
        score = model.decision_function(x[test])
        pred[test] = score
        rows.append({
            "representation": name, "held_out_slice": section, "n_test": int(test.sum()),
            "roc_auc": float(roc_auc_score(y[test], score)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], score >= 0)),
        })
    return pd.DataFrame(rows), pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cc-dir", type=Path, required=True)
    p.add_argument("--haruka-dir", type=Path, required=True)
    p.add_argument("--global-assignments", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    figdir, tabdir = args.out_dir / "figures", args.out_dir / "tables"
    figdir.mkdir(parents=True, exist_ok=True); tabdir.mkdir(parents=True, exist_ok=True)

    source_rows = np.load(args.cc_dir / "source_rows.npy").astype(int)
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    handle = ad.read_h5ad(args.data, backed="r")
    obs = handle.obs.iloc[source_rows].copy()
    obs.index = obs.index.astype(str)
    spatial = np.asarray(handle.obsm["spatial"], dtype=np.float32)[source_rows]
    handle.file.close()
    if not np.array_equal(names, obs.index.to_numpy()) or not obs["condition_label"].astype(str).eq("KRASi_treated").all():
        raise ValueError("CellCharter rows must be aligned KRASi-treated cells only")

    tumor_local = np.flatnonzero(obs["celltype"].astype(str).isin(TUMOR_TYPES).to_numpy())
    tumor = obs.iloc[tumor_local].copy()
    full_rows = source_rows[tumor_local]
    comp = np.load(args.haruka_dir / "salient_composition_shift.npy", mmap_mode="r")
    expr = np.load(args.haruka_dir / "salient_neighbor_mean_shift.npy", mmap_mode="r")
    comp_norm = np.linalg.norm(np.asarray(comp[full_rows], dtype=np.float32), axis=1)
    expr_norm = np.linalg.norm(np.asarray(expr[full_rows], dtype=np.float32), axis=1)
    tumor["composition_shift_norm"] = comp_norm
    tumor["niche_expression_shift_norm"] = expr_norm
    tumor["composition_shift_percentile"] = grouped_percentile(comp_norm, tumor)
    tumor["niche_expression_shift_percentile"] = grouped_percentile(expr_norm, tumor)
    mean_percentile = 0.5 * (tumor["composition_shift_percentile"] + tumor["niche_expression_shift_percentile"])
    tumor["response_percentile"] = grouped_percentile(mean_percentile.to_numpy(), tumor)
    tumor["response_score"] = tumor["response_percentile"]
    tumor["response_class"] = np.where(
        tumor["response_percentile"] <= 0.25, "relative_nonresponder",
        np.where(tumor["response_percentile"] >= 0.75, "relative_responder", "intermediate"),
    )
    # Some source AnnData objects already expose ``cell_id`` in ``obs``.
    # Canonicalize it to the observation index without inserting a duplicate.
    tumor["cell_id"] = tumor.index.astype(str)
    tumor["source_row"] = full_rows

    assignment = pd.read_csv(args.global_assignments, index_col=0)
    assignment.index = assignment.index.astype(str)
    tumor["s_cc_cluster"] = assignment.loc[tumor.cell_id, "s_cc_cluster"].to_numpy()
    tumor["z_cc_cluster"] = assignment.loc[tumor.cell_id, "z_cc_cluster"].to_numpy()
    tumor.to_csv(tabdir / "treated_tumor_relative_responder_assignments.csv.gz", index=False, compression="gzip")

    extreme = tumor.response_class.ne("intermediate").to_numpy()
    y = tumor.loc[extreme, "response_class"].eq("relative_responder").astype(int).to_numpy()
    sections = tumor.loc[extreme, "slice_id"].astype(str).to_numpy()
    zcc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    scc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    metric_parts, prediction_parts = [], []
    for name, matrix in [("z_cc", zcc), ("s_cc", scc)]:
        scores, pred = leave_section_out(np.asarray(matrix[tumor_local[extreme]], dtype=np.float32), y, sections, name)
        metric_parts.append(scores)
        prediction_parts.append(pd.DataFrame({
            "cell_id": tumor.loc[extreme, "cell_id"], "representation": name,
            "truth": y, "prediction": pred, "slice_id": sections,
        }))
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_csv(tabdir / "zcc_scc_relative_responder_leave_slice_out_metrics.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        tabdir / "zcc_scc_relative_responder_predictions.csv.gz", index=False, compression="gzip"
    )

    rows = []
    extreme_frame = tumor.loc[extreme].copy()
    extreme_frame["is_responder"] = y
    for cluster, inside in extreme_frame.groupby("s_cc_cluster", observed=True):
        outside = extreme_frame[extreme_frame.s_cc_cluster != cluster]
        table = [[inside.is_responder.sum(), len(inside) - inside.is_responder.sum()],
                 [outside.is_responder.sum(), len(outside) - outside.is_responder.sum()]]
        odds, pvalue = fisher_exact(table)
        rows.append({"s_cc_cluster": cluster, "n_extreme": len(inside),
                     "responder_fraction": inside.is_responder.mean(), "fisher_odds_ratio": odds,
                     "pvalue": pvalue})
    enrichment = pd.DataFrame(rows)
    enrichment["fdr_bh"] = multipletests(enrichment.pvalue, method="fdr_bh")[1]
    enrichment.to_csv(tabdir / "s_cc_cluster_relative_responder_enrichment.csv", index=False)

    sample = np.arange(len(tumor))
    if len(sample) > 40000:
        sample = np.sort(np.random.default_rng(args.seed).choice(sample, 40000, replace=False))
    xy = UMAP(n_neighbors=30, min_dist=0.25, random_state=args.seed, n_jobs=1).fit_transform(
        StandardScaler().fit_transform(np.asarray(scc[tumor_local[sample]], dtype=np.float32))
    )
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    q = axes[0].scatter(xy[:, 0], xy[:, 1], c=tumor.iloc[sample].response_percentile, cmap="magma", s=3, linewidths=0)
    fig.colorbar(q, ax=axes[0], label="Relative response percentile")
    for ax, key, title in [(axes[1], "s_cc_cluster", "s-CC response niche"), (axes[2], "slice_id", "Treated section")]:
        sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=tumor.iloc[sample][key].astype(str), s=5, linewidth=0, ax=ax, legend="brief")
        ax.set_title(title)
    axes[0].set_title("Decoded treatment-response magnitude")
    for ax in axes: ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
    fig.tight_layout(); fig.savefig(figdir / "treated_tumor_scc_relative_responder_umap.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    sections_unique = sorted(tumor.slice_id.astype(str).unique())
    fig, axes = plt.subplots(1, len(sections_unique), figsize=(6 * len(sections_unique), 5.5), squeeze=False)
    for ax, section in zip(axes[0], sections_unique):
        take = tumor.slice_id.astype(str).eq(section).to_numpy()
        pts = ax.scatter(spatial[tumor_local[take], 0], spatial[tumor_local[take], 1],
                         c=tumor.loc[take, "response_percentile"], cmap="magma", vmin=0, vmax=1,
                         s=3, linewidths=0, rasterized=True)
        ax.set_title(section); ax.set_aspect("equal"); ax.axis("off")
    fig.colorbar(pts, ax=axes.ravel().tolist(), label="Relative response percentile")
    fig.suptitle("KRASi-treated tumor response heterogeneity")
    fig.savefig(figdir / "treated_tumor_relative_responder_spatial.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary = metrics.groupby("representation")[["roc_auc", "balanced_accuracy"]].agg(["mean", "std"])
    for ax, metric in zip(axes, ["roc_auc", "balanced_accuracy"]):
        values = summary[metric]
        ax.bar(values.index, values["mean"], yerr=values["std"], color=["#4c72b0", "#dd8452"])
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1); ax.set_ylim(0, 1); ax.set_title(metric)
    fig.tight_layout(); fig.savefig(figdir / "scc_cluster_responder_enrichment_and_zs_benchmark.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    metadata = {
        "candidate_population": "KRASi_treated tumor cells only", "control_responder_labels": 0,
        "score": "mean percentile of decoded composition-shift norm and decoded niche-expression-shift norm, reranked within slice and tumor cell type",
        "classes": "bottom quartile relative_nonresponder; middle half intermediate; top quartile relative_responder",
        "validation": "leave-one-treated-slice-out z-CC versus condition-only s-CC",
        "limitation": "model-derived relative treatment-effect magnitude, not measured longitudinal shrinkage",
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    metric_summary = metrics.groupby("representation", as_index=False).agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    print(json.dumps({"n_tumor": len(tumor), "metrics": metric_summary.to_dict("records")}, default=float, indent=2))


if __name__ == "__main__":
    main()
