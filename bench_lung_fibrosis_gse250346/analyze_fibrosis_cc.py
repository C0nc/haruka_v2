#!/usr/bin/env python3
"""Analyze GSE250346 Haruka z-CC/s-CC and prepare paired MOSCOT landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import spearmanr, wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


def balanced_rows(obs: pd.DataFrame, mask: np.ndarray, per_slice: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    slices = obs["slice_id"].astype(str).to_numpy()
    for value in np.unique(slices[mask]):
        rows = np.flatnonzero(mask & (slices == value))
        if len(rows) > per_slice:
            rows = rng.choice(rows, per_slice, replace=False)
        selected.append(np.sort(rows))
    return np.concatenate(selected)


def cluster_score(rep: np.ndarray, obs: pd.DataFrame, rows: np.ndarray, label: str, seed: int) -> dict:
    truth = obs.iloc[rows][label].astype(str).to_numpy()
    n_clusters = len(np.unique(truth))
    scaled = StandardScaler().fit_transform(rep[rows])
    pred = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10, batch_size=8192
    ).fit_predict(scaled)
    return {
        "label": label,
        "n_clusters": n_clusters,
        "ari": adjusted_rand_score(truth, pred),
        "nmi": normalized_mutual_info_score(truth, pred),
    }


def scatter_categorical(ax, xy: np.ndarray, values: np.ndarray, title: str, size: float = 1.2) -> None:
    categories = pd.Categorical(values)
    palette = sns.color_palette("husl", len(categories.categories))
    for code, category in enumerate(categories.categories):
        take = categories.codes == code
        ax.scatter(xy[take, 0], xy[take, 1], s=size, color=palette[code], label=str(category), rasterized=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def stratified_landmarks(obs: pd.DataFrame, rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(rows) <= n:
        return rows
    labels = obs.iloc[rows]["cell_type"].astype(str).to_numpy()
    selected: list[int] = []
    categories, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.floor(n * counts / counts.sum()).astype(int))
    while quota.sum() > n:
        j = np.argmax(quota)
        if quota[j] > 1:
            quota[j] -= 1
        else:
            break
    for category, take_n in zip(categories, quota):
        available = rows[labels == category]
        selected.extend(rng.choice(available, min(take_n, len(available)), replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < n:
        remaining = np.setdiff1d(rows, np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, n - len(selected), replace=False).tolist())
    return np.sort(np.asarray(selected[:n], dtype=np.int64))


def core_centroids(
    rep: np.ndarray, obs: pd.DataFrame, target: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return one equally weighted representation per fibrosis core."""
    records: list[dict] = []
    vectors: list[np.ndarray] = []
    slice_values = obs["slice_id"].astype(str).to_numpy()
    for slice_id in np.unique(slice_values[target]):
        rows = np.flatnonzero(target & (slice_values == slice_id))
        block = obs.iloc[rows]
        records.append(
            {
                "slice_id": slice_id,
                "donor": str(block["donor"].iloc[0]),
                "sample_affect": str(block["sample_affect"].iloc[0]),
                "percent_pathology": float(block["percent_pathology"].iloc[0]),
                "n_cells": len(rows),
            }
        )
        vectors.append(np.asarray(rep[rows], dtype=np.float64).mean(axis=0))
    return pd.DataFrame(records), np.vstack(vectors)


def donor_heldout_severity(
    features: np.ndarray, core_obs: pd.DataFrame, representation: str
) -> tuple[dict, np.ndarray]:
    """Predict Less/More severity with every donor held out in turn."""
    y = (core_obs["sample_affect"].astype(str) == "More Affected").astype(int).to_numpy()
    groups = core_obs["donor"].astype(str).to_numpy()
    predictions = np.full(len(y), np.nan, dtype=float)
    for train, test in LeaveOneGroupOut().split(features, y, groups):
        if len(np.unique(y[train])) < 2:
            continue
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=0
        ).fit(scaler.transform(features[train]), y[train])
        predictions[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    valid = np.isfinite(predictions)
    result = {
        "representation": representation,
        "n_cores": int(valid.sum()),
        "n_donors": int(pd.Series(groups[valid]).nunique()),
        "roc_auc": roc_auc_score(y[valid], predictions[valid]),
        "average_precision": average_precision_score(y[valid], predictions[valid]),
        "balanced_accuracy": balanced_accuracy_score(y[valid], predictions[valid] >= 0.5),
        "validation": "leave-one-donor-out on core centroids",
    }
    return result, predictions


def core_mean_l2(rep: np.ndarray, obs: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    """Mean per-cell representation magnitude for every target core."""
    slice_values = obs["slice_id"].astype(str).to_numpy()
    values: list[float] = []
    for slice_id in np.unique(slice_values[target]):
        rows = np.flatnonzero(target & (slice_values == slice_id))
        block = np.asarray(rep[rows], dtype=np.float64)
        values.append(float(np.linalg.norm(block, axis=1).mean()))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cc-dir", type=Path, required=True)
    parser.add_argument("--haruka-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--plot-per-slice", type=int, default=3000)
    parser.add_argument("--landmarks-per-slice", type=int, default=1500)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    table_dir = args.out_dir / "tables"
    fig_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    data = ad.read_h5ad(args.data, backed="r")
    obs_full = data.obs.copy()
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    source_rows_path = args.cc_dir / "source_rows.npy"
    source_rows = (
        np.load(source_rows_path).astype(np.int64)
        if source_rows_path.exists()
        else np.arange(len(obs_full), dtype=np.int64)
    )
    obs = obs_full.iloc[source_rows].copy()
    if not np.array_equal(names, obs.index.astype(str).to_numpy()):
        raise ValueError("CellCharter rows do not match the AnnData observation order")
    z_cc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    s_cc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    target = obs["condition"].astype(str).to_numpy() == "fibrosis"
    target_rows = balanced_rows(obs, target, args.plot_per_slice, args.seed)
    all_rows = balanced_rows(obs, np.ones(len(obs), dtype=bool), args.plot_per_slice, args.seed)

    scaler_z = StandardScaler().fit(z_cc[target_rows])
    scaler_s = StandardScaler().fit(s_cc[target_rows])
    z_scaled = scaler_z.transform(z_cc[target_rows])
    s_scaled = scaler_s.transform(s_cc[target_rows])
    z_umap = umap.UMAP(n_neighbors=30, min_dist=0.25, random_state=args.seed).fit_transform(z_scaled)
    s_umap = umap.UMAP(n_neighbors=30, min_dist=0.25, random_state=args.seed).fit_transform(s_scaled)

    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters, random_state=args.seed, n_init=20, batch_size=8192
    ).fit(s_scaled)
    s_cluster_all = np.full(len(obs), -1, dtype=np.int16)
    chunk = 100_000
    for start in range(0, len(obs), chunk):
        rows = np.arange(start, min(start + chunk, len(obs)))
        rows = rows[target[rows]]
        if len(rows):
            s_cluster_all[rows] = kmeans.predict(scaler_s.transform(s_cc[rows]))
    s_cluster_plot = s_cluster_all[target_rows]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    scatter_categorical(axes[0, 0], z_umap, obs.iloc[target_rows]["lineage"].astype(str).to_numpy(), "z-CC — published lineage")
    scatter_categorical(axes[0, 1], z_umap, obs.iloc[target_rows]["CNiche"].astype(str).to_numpy(), "z-CC — cellular niche")
    scatter_categorical(axes[0, 2], z_umap, obs.iloc[target_rows]["run"].astype(str).to_numpy(), "z-CC — sequencing run")
    scatter_categorical(axes[1, 0], s_umap, s_cluster_plot.astype(str), "s-CC — unsupervised fibrosis-response cluster")
    scatter_categorical(axes[1, 1], s_umap, obs.iloc[target_rows]["TNiche"].astype(str).to_numpy(), "s-CC — transcriptional niche")
    pathology = obs.iloc[target_rows]["percent_pathology"].astype(float).to_numpy()
    sc = axes[1, 2].scatter(s_umap[:, 0], s_umap[:, 1], c=pathology, cmap="magma", s=1.2, rasterized=True)
    axes[1, 2].set_title("s-CC — percent pathology")
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    fig.colorbar(sc, ax=axes[1, 2], fraction=0.04, label="percent pathology")
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(markerscale=5, fontsize=6, frameon=False, ncol=2, loc="best")
    fig.suptitle("GSE250346 all fibrosis cells: background and salient CellCharter", fontsize=16)
    fig.tight_layout()
    fig.savefig(fig_dir / "condition_only_zcc_scc_umap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    severity_plot = obs.iloc[target_rows]["sample_affect"].astype(str).to_numpy()
    scatter_categorical(
        axes[0], z_umap, severity_plot,
        "Background z-CC — fibrosis severity", size=1.4,
    )
    scatter_categorical(
        axes[1], s_umap, severity_plot,
        "Salient s-CC — fibrosis severity", size=1.4,
    )
    for ax in axes:
        ax.legend(markerscale=5, fontsize=8, frameon=False)
    fig.suptitle("Less versus more affected cores within the Haruka fibrosis arm")
    fig.tight_layout()
    fig.savefig(fig_dir / "less_vs_more_zcc_scc_umap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics: list[dict] = []
    for rep_name, rep, rows in (("z_cc", z_cc, target_rows), ("s_cc", s_cc, target_rows)):
        for label in (
            "cell_type", "lineage", "CNiche", "TNiche", "sample_affect",
            "run", "donor",
        ):
            score = cluster_score(rep, obs, rows, label, args.seed)
            score["representation"] = rep_name
            score["arm"] = "all_fibrosis"
            metrics.append(score)
    pd.DataFrame(metrics).to_csv(table_dir / "zcc_scc_clustering_metrics.csv", index=False)

    assignments = obs[["slice_id", "donor", "run", "sample_affect", "percent_pathology", "cell_type", "lineage", "CNiche", "TNiche"]].copy()
    assignments["s_cc_cluster"] = s_cluster_all
    assignments.loc[~target, "s_cc_cluster"] = -1
    assignments.to_csv(table_dir / "s_cc_cluster_assignments.csv.gz", compression="gzip")

    sample_cluster = (
        assignments[target]
        .groupby(["slice_id", "percent_pathology", "s_cc_cluster"], observed=True)
        .size().rename("n").reset_index()
    )
    sample_cluster["fraction"] = sample_cluster["n"] / sample_cluster.groupby("slice_id")["n"].transform("sum")
    correlations = []
    for cluster_id in range(args.clusters):
        block = sample_cluster[sample_cluster["s_cc_cluster"] == cluster_id]
        rho, pvalue = spearmanr(block["percent_pathology"], block["fraction"])
        correlations.append({"s_cc_cluster": cluster_id, "spearman_rho": rho, "pvalue": pvalue})
    pd.DataFrame(correlations).to_csv(table_dir / "s_cc_cluster_pathology_correlations.csv", index=False)
    sample_cluster.to_csv(table_dir / "s_cc_cluster_fraction_by_core.csv", index=False)

    shift_full = np.load(args.haruka_dir / "salient_composition_shift.npy", mmap_mode="r")
    shift = shift_full[source_rows]
    categories = json.loads((args.haruka_dir / "meta.json").read_text())["cell_type_categories"]
    shift_means = np.zeros((args.clusters, shift.shape[1]), dtype=np.float64)
    shift_counts = np.zeros(args.clusters, dtype=np.int64)
    for cluster_id in range(args.clusters):
        rows = np.flatnonzero(s_cluster_all == cluster_id)
        shift_counts[cluster_id] = len(rows)
        for start in range(0, len(rows), chunk):
            shift_means[cluster_id] += np.asarray(shift[rows[start:start + chunk]], dtype=np.float64).sum(axis=0)
        shift_means[cluster_id] /= max(shift_counts[cluster_id], 1)
    shift_df = pd.DataFrame(shift_means, columns=categories)
    shift_df.insert(0, "s_cc_cluster", np.arange(args.clusters))
    shift_df.to_csv(table_dir / "s_cc_mean_composition_shift.csv", index=False)

    # Severity is not a Haruka training label.  Test it after fitting at the
    # independent core level, with donors held out, to avoid cell-level
    # pseudoreplication and donor leakage.
    core_obs, z_core = core_centroids(z_cc, obs, target)
    core_obs_s, s_core = core_centroids(s_cc, obs, target)
    core_obs_shift, shift_core = core_centroids(shift, obs, target)
    if not (
        core_obs["slice_id"].equals(core_obs_s["slice_id"])
        and core_obs["slice_id"].equals(core_obs_shift["slice_id"])
    ):
        raise RuntimeError("Core ordering differs among z-CC, s-CC, and decoded shift")

    severity_metrics: list[dict] = []
    for name, features in (
        ("z_cc", z_core),
        ("s_cc", s_core),
        ("decoded_composition_shift", shift_core),
    ):
        result, predictions = donor_heldout_severity(features, core_obs, name)
        severity_metrics.append(result)
        core_obs[f"more_affected_probability_{name}"] = predictions
    severity_df = pd.DataFrame(severity_metrics)
    severity_df.to_csv(table_dir / "severity_donor_heldout_metrics.csv", index=False)

    core_obs["z_cc_mean_l2"] = core_mean_l2(z_cc, obs, target)
    core_obs["s_cc_mean_l2"] = core_mean_l2(s_cc, obs, target)
    core_obs["decoded_shift_mean_l2"] = core_mean_l2(shift, obs, target)
    core_obs.to_csv(table_dir / "core_severity_scores.csv", index=False)

    correlation_records: list[dict] = []
    for score in ("z_cc_mean_l2", "s_cc_mean_l2", "decoded_shift_mean_l2"):
        rho, pvalue = spearmanr(core_obs["percent_pathology"], core_obs[score])
        correlation_records.append(
            {"score": score, "spearman_rho": rho, "pvalue": pvalue, "n_cores": len(core_obs)}
        )
    pd.DataFrame(correlation_records).to_csv(
        table_dir / "core_pathology_correlations.csv", index=False
    )

    paired_deltas: list[dict] = []
    paired_summary = core_obs.groupby(
        ["donor", "sample_affect"], observed=True
    )[["z_cc_mean_l2", "s_cc_mean_l2", "decoded_shift_mean_l2"]].mean()
    for score in ("z_cc_mean_l2", "s_cc_mean_l2", "decoded_shift_mean_l2"):
        wide = paired_summary[score].unstack("sample_affect")
        if {"Less Affected", "More Affected"}.issubset(wide.columns):
            delta = (wide["More Affected"] - wide["Less Affected"]).dropna()
        else:
            delta = pd.Series(dtype=float)
        statistic, pvalue = (
            wilcoxon(delta.to_numpy()) if len(delta) and np.any(delta.to_numpy() != 0)
            else (np.nan, np.nan)
        )
        paired_deltas.append(
            {
                "score": score,
                "n_paired_donors": len(delta),
                "median_more_minus_less": float(delta.median()) if len(delta) else np.nan,
                "wilcoxon_statistic": statistic,
                "pvalue": pvalue,
            }
        )
    pd.DataFrame(paired_deltas).to_csv(
        table_dir / "paired_donor_severity_deltas.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].bar(
        severity_df["representation"], severity_df["roc_auc"],
        color=["#4c72b0", "#dd8452", "#55a868"],
    )
    axes[0].axhline(0.5, color="black", ls="--", lw=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Donor-held-out ROC AUC")
    axes[0].set_title("Core-level Less vs More prediction")
    axes[0].tick_params(axis="x", rotation=20)
    palette = {"Less Affected": "#4c72b0", "More Affected": "#c44e52"}
    for ax, score, title in (
        (axes[1], "s_cc_mean_l2", "Salient s-CC response magnitude"),
        (axes[2], "decoded_shift_mean_l2", "Decoded composition-shift magnitude"),
    ):
        sns.scatterplot(
            data=core_obs, x="percent_pathology", y=score,
            hue="sample_affect", palette=palette, s=55, ax=ax,
        )
        rho, pvalue = spearmanr(core_obs["percent_pathology"], core_obs[score])
        ax.set_title(f"{title}\nSpearman rho={rho:.2f}, p={pvalue:.3g}")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Independent fibrosis cores: post-hoc severity readout")
    fig.tight_layout()
    fig.savefig(fig_dir / "severity_core_level_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(np.max(np.abs(shift_means), axis=0))[-20:]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [3, 1]})
    sns.heatmap(shift_means[:, order], cmap="vlag", center=0, xticklabels=np.asarray(categories)[order], yticklabels=[f"S{i}" for i in range(args.clusters)], ax=axes[0])
    axes[0].set_title("Mean decoded salient cell-composition shift")
    corr = pd.DataFrame(correlations).sort_values("spearman_rho")
    axes[1].barh([f"S{x}" for x in corr["s_cc_cluster"]], corr["spearman_rho"], color=np.where(corr["spearman_rho"] >= 0, "#c44e52", "#4c72b0"))
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("Core fraction vs pathology")
    axes[1].set_xlabel("Spearman rho")
    fig.tight_layout()
    fig.savefig(fig_dir / "s_cc_cluster_biology_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    affected = obs["sample_affect"].astype(str).isin(
        ["Less Affected", "More Affected"]
    )
    paired = obs.loc[affected].groupby(
        "donor", observed=True
    )["sample_affect"].nunique()
    paired_donors = paired[paired == 2].index.astype(str).tolist()
    rng = np.random.default_rng(args.seed)
    landmark_rows: list[np.ndarray] = []
    for donor in paired_donors:
        donor_mask = (
            (obs["donor"].astype(str).to_numpy() == donor)
            & affected.to_numpy()
        )
        for slice_id in np.unique(obs.loc[donor_mask, "slice_id"].astype(str)):
            rows = np.flatnonzero(donor_mask & (obs["slice_id"].astype(str).to_numpy() == slice_id))
            landmark_rows.append(stratified_landmarks(obs, rows, args.landmarks_per_slice, rng))
    landmark_rows_array = np.concatenate(landmark_rows)
    landmark_obs = obs.iloc[landmark_rows_array][
        ["slice_id", "donor", "condition", "sample_affect",
         "percent_pathology", "cell_type", "lineage", "CNiche", "TNiche"]
    ].copy()
    # The full Haruka contrast is unaffected vs all fibrosis.  The paired
    # transport object intentionally receives a separate Less -> More arm
    # label so the existing moscot scripts cannot confuse training arms with
    # severity-transition arms.
    landmark_obs["haruka_condition"] = landmark_obs["condition"].astype(str)
    landmark_obs["condition"] = np.where(
        landmark_obs["sample_affect"].astype(str) == "Less Affected",
        "reference",
        "fibrosis",
    )
    landmark_obs["source_index"] = source_rows[landmark_rows_array]
    landmark_obs["s_cc_cluster"] = s_cluster_all[landmark_rows_array].astype(str)
    landmarks = ad.AnnData(X=np.zeros((len(landmark_rows_array), 1), dtype=np.float32), obs=landmark_obs)
    landmarks.obsm["spatial"] = np.asarray(data.obsm["spatial"])[source_rows[landmark_rows_array]].astype(np.float32)
    landmarks.obsm["X_z_cc"] = np.asarray(z_cc[landmark_rows_array], dtype=np.float32)
    landmarks.obsm["X_s_cc"] = np.asarray(s_cc[landmark_rows_array], dtype=np.float32)
    landmarks.uns["alignment_design"] = {
        "joint_feature": "X_z_cc (background CellCharter)",
        "overlay_only": "X_s_cc and s_cc_cluster",
        "pairing": "within donor, Less Affected reference to More Affected target",
        "haruka_training_contrast": "Unaffected reference vs all fibrosis condition",
    }
    landmarks.write_h5ad(args.out_dir / "moscot_landmarks.h5ad", compression="gzip")

    pd.DataFrame({"plot_row": target_rows, "s_cc_cluster": s_cluster_plot, "z_umap1": z_umap[:, 0], "z_umap2": z_umap[:, 1], "s_umap1": s_umap[:, 0], "s_umap2": s_umap[:, 1]}).to_csv(table_dir / "condition_umap_coordinates.csv.gz", index=False, compression="gzip")
    print(json.dumps({"n_cells": len(obs), "n_target": int(target.sum()), "paired_donors": paired_donors, "n_moscot_landmarks": len(landmark_rows_array)}, indent=2))


if __name__ == "__main__":
    main()
