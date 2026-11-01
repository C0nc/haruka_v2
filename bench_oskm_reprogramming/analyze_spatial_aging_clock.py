#!/usr/bin/env python
"""Evaluate Haruka OSKM embeddings with the published SpatialAgingClock.

Haruka is trained only on old control and old OSKM cells.  The aging clock is
an external, frozen, cell-type-specific model and is never used as supervision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--emb", required=True, type=Path)
    parser.add_argument("--clock-repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--max-umap-cells", type=int, default=30000)
    return parser.parse_args()


def load_aligned(data_path: Path, emb_path: Path):
    data = ad.read_h5ad(data_path)
    names = np.load(emb_path / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Index(data.obs_names.astype(str)).get_indexer(names)
    if (order < 0).any() or len(np.unique(order)) != len(order):
        raise ValueError("Embedding observations do not align one-to-one with AnnData")
    data = data[order].copy()
    counts = data.layers["counts"]
    # The published package assumes dense expression during convergence checks.
    data.X = (
        counts.toarray().astype(np.float32, copy=False)
        if sparse.issparse(counts)
        else np.asarray(counts, dtype=np.float32).copy()
    )
    zcc = np.load(emb_path / "spatial.npy").astype(np.float32)
    scc = np.load(emb_path / "salient_spatial.npy").astype(np.float32)
    if zcc.shape[0] != data.n_obs or scc.shape[0] != data.n_obs:
        raise ValueError("Embedding row count does not match AnnData")
    return data, zcc, scc


def apply_official_clock(data, clock_repo: Path) -> None:
    sys.path.insert(0, str(clock_repo))
    from spatialclock.deploy import get_predictions

    get_predictions(
        data,
        fill_missing="mean",
        smooth=True,
        normalize=True,
        standardize=True,
        add_in_place=True,
    )


def add_matched_scores(obs: pd.DataFrame) -> pd.DataFrame:
    frame = obs.copy()
    frame["predicted_age"] = pd.to_numeric(frame["predicted_age"])
    frame["is_oskm"] = frame["condition"].astype(str).eq("oskm")
    frame["clock_rejuvenation"] = np.nan
    frame["baseline_level"] = "unmatched"

    control = frame.loc[~frame["is_oskm"]]
    levels = [
        (["paired_batch", "celltype", "subregion"], "batch_celltype_subregion", 20),
        (["paired_batch", "celltype"], "batch_celltype", 20),
        (["celltype", "subregion"], "celltype_subregion", 20),
        (["celltype"], "celltype", 20),
    ]
    target_index = frame.index[frame["is_oskm"]]
    remaining = pd.Series(True, index=target_index)
    for keys, label, minimum in levels:
        med = control.groupby(keys, observed=True)["predicted_age"].agg(["median", "size"])
        target = frame.loc[target_index[remaining]]
        joined = target[keys].join(med, on=keys)
        valid = joined["size"].fillna(0).ge(minimum)
        chosen = joined.index[valid]
        frame.loc[chosen, "clock_rejuvenation"] = (
            joined.loc[chosen, "median"] - frame.loc[chosen, "predicted_age"]
        )
        frame.loc[chosen, "baseline_level"] = label
        remaining.loc[chosen] = False

    # Remove the mean cell-type/subregion response to isolate responder
    # heterogeneity, without using mouse labels or centering within each mouse.
    target = frame.loc[frame["is_oskm"] & frame["clock_rejuvenation"].notna()]
    group_median = target.groupby(
        ["celltype", "subregion"], observed=True
    )["clock_rejuvenation"].transform("median")
    frame["clock_rejuvenation_heterogeneity"] = np.nan
    frame.loc[target.index, "clock_rejuvenation_heterogeneity"] = (
        target["clock_rejuvenation"] - group_median
    )
    return frame


def paired_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(
        ["paired_batch", "condition", "celltype"], observed=True
    )["predicted_age"].agg(["median", "size"]).reset_index()
    wide = grouped.pivot_table(
        index=["paired_batch", "celltype"], columns="condition", values=["median", "size"]
    )
    rows = []
    for celltype in sorted(frame["celltype"].astype(str).unique()):
        try:
            block = wide.xs(celltype, level="celltype")
            delta = block[("median", "oskm")] - block[("median", "control")]
            ok = (
                block[("size", "oskm")].ge(20)
                & block[("size", "control")].ge(20)
                & delta.notna()
            )
            delta = delta[ok]
        except KeyError:
            delta = pd.Series(dtype=float)
        if len(delta) >= 2:
            test = stats.ttest_1samp(delta, 0.0)
            pvalue = float(test.pvalue)
        else:
            pvalue = np.nan
        rows.append(
            {
                "celltype": celltype,
                "n_pairs": int(len(delta)),
                "median_oskm_minus_control": float(np.median(delta)) if len(delta) else np.nan,
                "mean_oskm_minus_control": float(np.mean(delta)) if len(delta) else np.nan,
                "paired_t_pvalue": pvalue,
                "all_pairs_younger": bool((delta < 0).all()) if len(delta) else False,
                **{f"delta_{idx}": float(value) for idx, value in delta.items()},
            }
        )
    result = pd.DataFrame(rows)
    valid = result["paired_t_pvalue"].notna()
    result["paired_t_fdr"] = np.nan
    if valid.any():
        result.loc[valid, "paired_t_fdr"] = multipletests(
            result.loc[valid, "paired_t_pvalue"], method="fdr_bh"
        )[1]
    return result.sort_values("median_oskm_minus_control")


def grouped_ridge(rep: np.ndarray, response: np.ndarray, groups: np.ndarray):
    valid = np.isfinite(response)
    x, y, g = rep[valid], response[valid], groups[valid]
    unique = np.unique(g)
    folds = GroupKFold(n_splits=len(unique))
    pred = np.full(len(y), np.nan)
    for train, test in folds.split(x, y, groups=g):
        model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 12))
        )
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    return {
        "n": int(len(y)),
        "n_mice": int(len(unique)),
        "r2": float(r2_score(y, pred)),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "mae": float(mean_absolute_error(y, pred)),
    }, y, pred


def make_umap(rep: np.ndarray, rows: np.ndarray, seed: int, maximum: int):
    import umap

    rng = np.random.default_rng(seed)
    if len(rows) > maximum:
        rows = np.sort(rng.choice(rows, maximum, replace=False))
    x = StandardScaler().fit_transform(rep[rows])
    if x.shape[1] > 50:
        x = PCA(50, random_state=seed).fit_transform(x)
    xy = umap.UMAP(
        n_neighbors=30, min_dist=0.25, metric="cosine", random_state=seed
    ).fit_transform(x)
    return rows, xy


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)

    data, zcc, scc = load_aligned(args.data, args.emb)
    score_path = args.out / "spatial_clock_cell_scores.csv.gz"
    if score_path.exists():
        frame = pd.read_csv(score_path)
        if not np.array_equal(frame["cell_id"].astype(str), data.obs_names.astype(str)):
            raise ValueError("Cached clock scores do not align with AnnData")
    else:
        apply_official_clock(data, args.clock_repo)
        frame = add_matched_scores(data.obs)
        frame.insert(0, "cell_id", data.obs_names.astype(str))
        frame.to_csv(score_path, index=False)

    paired = paired_summary(frame)
    paired.to_csv(args.out / "spatial_clock_celltype_paired_summary.csv", index=False)

    target = frame["is_oskm"].to_numpy()
    mice = frame.loc[target, "mouse_id"].astype(str).to_numpy()
    score = frame.loc[target, "clock_rejuvenation"].to_numpy(float)
    heterogeneity = frame.loc[
        target, "clock_rejuvenation_heterogeneity"
    ].to_numpy(float)
    metrics = {}
    predictions = {}
    for name, rep in {"zCC": zcc[target], "sCC": scc[target]}.items():
        metrics[name] = {}
        for endpoint, response in {
            "matched_rejuvenation": score,
            "within_anatomy_heterogeneity": heterogeneity,
        }.items():
            result, observed, predicted = grouped_ridge(rep, response, mice)
            metrics[name][endpoint] = result
            predictions[(name, endpoint)] = (observed, predicted)

    # Unsupervised cluster enrichment for the matched rejuvenation score.
    cluster_tables = []
    for name, rep in {"zCC": zcc[target], "sCC": scc[target]}.items():
        cluster = MiniBatchKMeans(
            args.clusters, n_init=20, batch_size=4096, random_state=args.seed
        ).fit_predict(StandardScaler().fit_transform(rep))
        table = pd.DataFrame(
            {"representation": name, "cluster": cluster, "score": score}
        ).groupby(["representation", "cluster"], observed=True)["score"].agg(
            ["median", "mean", "size"]
        ).reset_index()
        cluster_tables.append(table)
    pd.concat(cluster_tables).to_csv(
        args.out / "spatial_clock_cluster_rejuvenation.csv", index=False
    )

    clocked = frame["predicted_age"].notna()
    metrics["clock_coverage"] = {
        "n": int(clocked.sum()),
        "fraction": float(clocked.mean()),
        "celltypes": int(frame.loc[clocked, "celltype"].nunique()),
    }
    metrics["design"] = {
        "haruka_training": "old_notreatment vs old_treatment only",
        "clock": "published frozen cell-type-specific SpatialAgingClock",
        "clock_supervision_of_haruka": False,
        "spatial_smoothing": "official fixed 20-neighbor, alpha=0.8",
        "evaluation_cv": "leave-one-OSKM-mouse-out",
        "positive_score": "younger than matched old control",
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Cell-type paired age shifts.
    plot = paired.dropna(subset=["median_oskm_minus_control"]).copy()
    plot = plot.sort_values("median_oskm_minus_control")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(plot["median_oskm_minus_control"] < 0, "#2a9d8f", "#e76f51")
    ax.barh(plot["celltype"], plot["median_oskm_minus_control"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OSKM − matched old-control predicted age (months)")
    ax.set_title("Published SpatialAgingClock: OSKM rejuvenation by cell type")
    fig.tight_layout()
    fig.savefig(figures / "spatial_clock_celltype_rejuvenation.png", dpi=220)
    plt.close(fig)

    # Same scalar score on independent z-CC and s-CC geometries.
    target_rows = np.flatnonzero(target & frame["clock_rejuvenation"].notna().to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, (name, rep), offset in zip(
        axes, [("z-CC", zcc), ("s-CC", scc)], [0, 1]
    ):
        rows, xy = make_umap(rep, target_rows, args.seed + offset, args.max_umap_cells)
        values = frame.iloc[rows]["clock_rejuvenation"].to_numpy(float)
        limit = np.nanpercentile(np.abs(values), 98)
        points = axis.scatter(
            xy[:, 0], xy[:, 1], c=values, s=2, cmap="coolwarm",
            vmin=-limit, vmax=limit, alpha=0.75, rasterized=True,
        )
        axis.set_title(f"{name}: clock-derived younger shift")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(points, ax=axis, fraction=0.045, pad=0.02, label="months younger")
    fig.suptitle("Old OSKM cells only; positive values indicate rejuvenation")
    fig.tight_layout()
    fig.savefig(figures / "spatial_clock_zcc_scc_umap.png", dpi=220)
    plt.close(fig)

    # Mouse-held-out representation comparison.
    rows = []
    for representation in ["zCC", "sCC"]:
        for endpoint in ["matched_rejuvenation", "within_anatomy_heterogeneity"]:
            rows.append(
                {"representation": representation, "endpoint": endpoint, **metrics[representation][endpoint]}
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.out / "embedding_clock_prediction_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=comparison, x="endpoint", y="spearman", hue="representation", ax=axes[0])
    sns.barplot(data=comparison, x="endpoint", y="r2", hue="representation", ax=axes[1])
    for axis, title in zip(axes, ["Spearman", "R²"]):
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(f"Leave-one-OSKM-mouse-out {title}")
        axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures / "zcc_scc_clock_prediction_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
