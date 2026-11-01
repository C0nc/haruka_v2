"""Evaluate target-only composition-gated Haruka z-CC and s-CC consistently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def balanced_sample(groups: np.ndarray, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.unique(groups)
    per_group = max(1, limit // max(len(values), 1))
    parts = []
    for value in values:
        rows = np.flatnonzero(groups == value)
        parts.append(rng.choice(rows, min(per_group, len(rows)), replace=False))
    return np.sort(np.concatenate(parts))


def heldout_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, 0
    scores = []
    for train, test in GroupKFold(n_splits=n_splits).split(x, y, groups):
        common = np.intersect1d(np.unique(y[train]), np.unique(y[test]))
        keep_train = np.isin(y[train], common)
        keep_test = np.isin(y[test], common)
        if len(common) < 2 or keep_test.sum() < 20:
            continue
        model = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=10.0, class_weight="balanced"),
        )
        model.fit(x[train][keep_train], y[train][keep_train])
        scores.append(balanced_accuracy_score(y[test][keep_test], model.predict(x[test][keep_test])))
    return (
        float(np.mean(scores)) if scores else np.nan,
        float(np.std(scores)) if scores else np.nan,
        len(scores),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cc-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--study", required=True)
    parser.add_argument("--sample-key", default="graph_id")
    parser.add_argument("--stage-key", default="stage")
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--domain-key", default="tissue_domain")
    parser.add_argument("--max-fit", type=int, default=40000)
    parser.add_argument("--max-plot", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    figdir = args.out_dir / "figures"
    tabdir = args.out_dir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    handle = ad.read_h5ad(args.data, backed="r")
    source_rows = np.load(args.cc_dir / "source_rows.npy").astype(int)
    obs = handle.obs.iloc[source_rows].copy()
    names = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(names, obs.index.astype(str).to_numpy()):
        raise ValueError("Target-only CellCharter rows do not align to the source AnnData")
    zcc = np.load(args.cc_dir / "spatial.npy", mmap_mode="r")
    scc = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")
    sample = obs[args.sample_key].astype(str).to_numpy()
    fit_rows = balanced_sample(sample, args.max_fit, args.seed)

    factors = {
        "stage": args.stage_key,
        "cell_type": args.cell_type_key,
        "tissue_domain": args.domain_key,
        "sample": args.sample_key,
    }
    metrics = []
    cluster_assignments = pd.DataFrame({"cell_id": names, "source_row": source_rows})
    for rep_name, matrix in (("z_cc", zcc), ("s_cc", scc)):
        xfit = np.asarray(matrix[fit_rows], dtype=np.float32)
        xfit_scaled = StandardScaler().fit_transform(xfit)
        for factor, key in factors.items():
            if key not in obs:
                continue
            y = obs[key].astype(str).to_numpy()
            yfit = y[fit_rows]
            n_classes = len(np.unique(yfit))
            if n_classes < 2:
                continue
            clusters = MiniBatchKMeans(
                n_clusters=n_classes, random_state=args.seed, n_init=20, batch_size=4096
            ).fit_predict(xfit_scaled)
            score, sd, folds = heldout_score(xfit, yfit, sample[fit_rows])
            metrics.append({
                "study": args.study,
                "representation": rep_name,
                "factor": factor,
                "n_cells": int(len(fit_rows)),
                "n_classes": int(n_classes),
                "kmeans_ari": float(adjusted_rand_score(yfit, clusters)),
                "kmeans_nmi": float(normalized_mutual_info_score(yfit, clusters)),
                "leave_sample_out_balanced_accuracy": score,
                "fold_sd": sd,
                "n_folds": folds,
            })
        scaler = StandardScaler().fit(xfit)
        k = min(12, max(2, len(np.unique(obs[args.stage_key].astype(str)))))
        model = MiniBatchKMeans(n_clusters=k, random_state=args.seed, n_init=20, batch_size=4096)
        model.fit(scaler.transform(xfit))
        cluster_assignments[f"{rep_name}_cluster"] = model.predict(
            scaler.transform(np.asarray(matrix, dtype=np.float32))
        )

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(tabdir / "target_zcc_scc_metrics.csv", index=False)
    cluster_assignments.to_csv(tabdir / "target_zcc_scc_clusters.csv.gz", index=False, compression="gzip")

    plot_rows = balanced_sample(sample, args.max_plot, args.seed + 1)
    color_factors = [
        ("stage", args.stage_key),
        ("cell type", args.cell_type_key),
        ("tissue domain", args.domain_key),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    for row, (rep_name, matrix) in enumerate((("z-CC background", zcc), ("s-CC salient", scc))):
        xy = UMAP(n_neighbors=30, min_dist=0.25, random_state=args.seed, n_jobs=1).fit_transform(
            StandardScaler().fit_transform(np.asarray(matrix[plot_rows], dtype=np.float32))
        )
        for col, (label, key) in enumerate(color_factors):
            ax = axes[row, col]
            values = obs.iloc[plot_rows][key].astype(str) if key in obs else pd.Series("unavailable", index=plot_rows)
            sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=values.to_numpy(), s=3, linewidth=0,
                            alpha=0.75, ax=ax, legend="brief", rasterized=True)
            ax.set_title(f"{rep_name} — {label}")
            ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
            if ax.legend_ is not None:
                ax.legend(fontsize=6, markerscale=2, frameon=False)
    fig.suptitle(f"{args.study}: target-only composition-gated Haruka", fontsize=16)
    fig.tight_layout()
    fig.savefig(figdir / "target_only_zcc_scc_umap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "study": args.study,
        "population": "condition/target cells only (as encoded by source_rows.npy)",
        "representations": "symmetric slice-bounded 3-hop means with the own-cell block removed",
        "clustering": "deterministic MiniBatchKMeans; cluster count matched to each evaluation label",
        "validation": "leave-one-sample-out ridge classification",
        "n_target": int(len(obs)),
        "seed": args.seed,
    }
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()
