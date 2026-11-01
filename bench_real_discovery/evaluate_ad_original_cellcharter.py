"""Disease-only plaque evaluation of original CellCharter versus Haruka."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP


DISPLAY = {
    "scvi": "scVI latent (before CC)",
    "original_cc": "Original CellCharter (scVI-CC)",
    "haruka_zcc": "Haruka background (z-CC)",
    "haruka_scc": "Haruka salient (s-CC)",
}
PLOT_REPS = ["original_cc", "haruka_zcc", "haruka_scc"]
PLAQUE_COLORS = {
    "No plaque": "#d7d7d7",
    "p-tau only": "#d9485f",
    "Aβ only": "#6f6ad8",
    "Aβ + p-tau": "#f2a43a",
}


def aligned_rep(directory: Path, filename: str, wanted: pd.Index) -> np.ndarray:
    names = np.load(directory / "obs_names.npy", allow_pickle=True).astype(str)
    locations = pd.Index(names).get_indexer(wanted)
    if (locations < 0).any():
        raise ValueError(f"Missing cells in {directory}: {wanted[locations < 0][:5].tolist()}")
    return np.load(directory / filename).astype(np.float32)[locations]


def make_spatial_blocks(xy: np.ndarray, seed: int, n_blocks: int = 20) -> np.ndarray:
    standardized = StandardScaler().fit_transform(xy)
    return KMeans(n_clusters=n_blocks, n_init=20, random_state=seed).fit_predict(standardized)


def spatial_cv(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, seed: int) -> dict:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    fold = np.full(len(y), -1, dtype=np.int16)
    for fold_id, (train, test) in enumerate(splitter.split(x, y, groups=blocks)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=seed,
            ),
        )
        model.fit(x[train], y[train])
        prediction[test] = model.predict_proba(x[test])[:, 1]
        fold[test] = fold_id
    valid = np.isfinite(prediction)
    return {
        "prevalence": float(y.mean()),
        "spatial_block_auroc": float(roc_auc_score(y[valid], prediction[valid])),
        "spatial_block_average_precision": float(
            average_precision_score(y[valid], prediction[valid])
        ),
        "spatial_block_balanced_accuracy": float(
            balanced_accuracy_score(y[valid], prediction[valid] >= 0.5)
        ),
        "prediction": prediction,
        "fold": fold,
    }


def plaque_class(obs: pd.DataFrame) -> np.ndarray:
    tau = obs["plaque_tau"].to_numpy(int) > 0
    abeta = obs["plaque_abeta"].to_numpy(int) > 0
    out = np.full(len(obs), "No plaque", dtype=object)
    out[tau & ~abeta] = "p-tau only"
    out[~tau & abeta] = "Aβ only"
    out[tau & abeta] = "Aβ + p-tau"
    return out.astype(str)


def scatter_binary(ax, xy, y, color, positive_name) -> None:
    neg, pos = y == 0, y == 1
    ax.scatter(xy[neg, 0], xy[neg, 1], s=4, c="#d9d9d9", alpha=0.30,
               linewidths=0, rasterized=True)
    ax.scatter(xy[pos, 0], xy[pos, 1], s=8, c=color, alpha=0.88,
               linewidths=0, rasterized=True)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="", color="#d9d9d9", markersize=5,
               label=f"Negative (n={neg.sum():,})"),
        Line2D([0], [0], marker="o", linestyle="", color=color, markersize=5,
               label=f"{positive_name}+ (n={pos.sum():,})"),
    ], loc="best", frameon=False, fontsize=7.5)


def scatter_classes(ax, xy, labels) -> None:
    for level, color in PLAQUE_COLORS.items():
        selected = labels == level
        ax.scatter(xy[selected, 0], xy[selected, 1],
                   s=4 if level == "No plaque" else 8,
                   c=color, alpha=0.30 if level == "No plaque" else 0.88,
                   linewidths=0, rasterized=True,
                   label=f"{level} (n={selected.sum():,})")
    ax.legend(loc="best", frameon=False, fontsize=7.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--haruka", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(args.data, backed="r")
    disease = a.obs["condition"].astype(str).eq("disease").to_numpy()
    obs = a.obs.iloc[np.flatnonzero(disease)].copy()
    wanted = pd.Index(obs.index.astype(str))
    xy_spatial = np.asarray(a.obsm["spatial"])[disease].astype(np.float32)
    reps = {
        "scvi": aligned_rep(args.original, "scvi.npy", wanted),
        "original_cc": aligned_rep(args.original, "cellcharter.npy", wanted),
        "haruka_zcc": aligned_rep(args.haruka, "spatial.npy", wanted),
        "haruka_scc": aligned_rep(args.haruka, "salient_spatial.npy", wanted),
    }
    labels = {
        key: obs[key].to_numpy(dtype=int)
        for key in ("plaque_tau", "plaque_abeta", "plaque_any")
    }
    combined = plaque_class(obs)
    blocks = make_spatial_blocks(xy_spatial, args.seed)

    score_rows, prediction_columns = [], {}
    score_lookup = {}
    for rep_name, rep in reps.items():
        for plaque_name, y in labels.items():
            result = spatial_cv(rep, y, blocks, args.seed)
            row = {"representation": rep_name, "plaque": plaque_name,
                   **{k: v for k, v in result.items() if k not in ("prediction", "fold")}}
            score_rows.append(row)
            score_lookup[(rep_name, plaque_name)] = row
            prediction_columns[f"{rep_name}_{plaque_name}_probability"] = result["prediction"]
            prediction_columns[f"{rep_name}_{plaque_name}_fold"] = result["fold"]
    scores = pd.DataFrame(score_rows)
    scores.to_csv(args.out / "disease_only_spatial_block_plaque_scores.csv", index=False)

    clustering_rows = []
    cluster_targets = {**labels, "plaque_class": combined}
    for rep_name, rep in reps.items():
        scaled = StandardScaler().fit_transform(rep)
        reduced = PCA(n_components=min(30, scaled.shape[1]), random_state=args.seed).fit_transform(scaled)
        for label_name, y in cluster_targets.items():
            y_text = np.asarray(y).astype(str)
            n_components = len(np.unique(y_text))
            predicted = GaussianMixture(
                n_components=n_components,
                covariance_type="diag",
                n_init=5,
                random_state=args.seed,
            ).fit_predict(reduced)
            clustering_rows.append({
                "representation": rep_name,
                "label": label_name,
                "n_clusters": n_components,
                "ari": float(adjusted_rand_score(y_text, predicted)),
                "nmi": float(normalized_mutual_info_score(y_text, predicted)),
            })
    pd.DataFrame(clustering_rows).to_csv(
        args.out / "disease_only_pca_gmm_plaque_clustering.csv", index=False
    )

    umaps = {}
    coordinate_rows = []
    for rep_name in PLOT_REPS:
        scaled = StandardScaler().fit_transform(reps[rep_name])
        umaps[rep_name] = UMAP(
            n_neighbors=30, min_dist=0.25, metric="euclidean",
            random_state=args.seed, n_jobs=1,
        ).fit_transform(scaled)
        coordinate_rows.append(pd.DataFrame({
            "obs_name": wanted,
            "representation": rep_name,
            "UMAP1": umaps[rep_name][:, 0],
            "UMAP2": umaps[rep_name][:, 1],
            "plaque_tau": labels["plaque_tau"],
            "plaque_abeta": labels["plaque_abeta"],
            "plaque_any": labels["plaque_any"],
            "plaque_class": combined,
            "spatial_block": blocks,
        }))
    pd.concat(coordinate_rows, ignore_index=True).to_csv(
        args.out / "original_cc_vs_haruka_disease_umap_coordinates.csv", index=False
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    for row_index, rep_name in enumerate(PLOT_REPS):
        embedding = umaps[rep_name]
        scatter_binary(axes[row_index, 0], embedding, labels["plaque_tau"], "#d9485f", "p-tau")
        scatter_binary(axes[row_index, 1], embedding, labels["plaque_abeta"], "#6f6ad8", "Aβ")
        scatter_classes(axes[row_index, 2], embedding, combined)
        for col_index, plaque_name in enumerate(("plaque_tau", "plaque_abeta", "plaque_any")):
            metric = score_lookup[(rep_name, plaque_name)]
            heading = ("p-tau", "Aβ", "Plaque class")[col_index]
            axes[row_index, col_index].set_title(
                f"{heading}\nspatial-block AUROC {metric['spatial_block_auroc']:.3f}  |  "
                f"AP {metric['spatial_block_average_precision']:.3f}", fontsize=10.5
            )
            axes[row_index, col_index].set(xticks=[], yticks=[], xlabel="UMAP 1")
            axes[row_index, col_index].set_ylabel("UMAP 2" if col_index == 0 else "")
            axes[row_index, col_index].spines[["top", "right", "left", "bottom"]].set_visible(False)
        axes[row_index, 0].text(
            -0.12, 0.5, DISPLAY[rep_name], transform=axes[row_index, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=12.5, fontweight="bold",
        )
    fig.suptitle(
        "Jill AD disease_r1: original scVI-CellCharter versus Haruka plaque recovery",
        fontsize=16, fontweight="bold",
    )
    fig.text(
        0.5, 0.002,
        "Original scVI-CellCharter was trained jointly on control_r1+disease_r1 with slice_id batch; plaque labels are evaluation-only.",
        ha="center", fontsize=9.5, color="#555555",
    )
    figure = args.out / "original_cellcharter_vs_haruka_disease_plaque_umap.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    prediction_table = pd.DataFrame({
        "obs_name": wanted,
        "x": xy_spatial[:, 0],
        "y": xy_spatial[:, 1],
        "spatial_block": blocks,
        **labels,
        **prediction_columns,
    })
    prediction_table.to_csv(args.out / "disease_only_plaque_predictions.csv", index=False)
    summary = {
        "n_fit_all": int(a.n_obs),
        "n_evaluated_disease": int(len(obs)),
        "fit_arms": ["control_r1", "disease_r1"],
        "evaluation_arm": "disease_r1",
        "scvi_batch_key": "slice_id",
        "evaluation": "5-fold stratified group CV over 20 contiguous spatial KMeans blocks",
        "plaque_labels_used_for_training": False,
        "figure": str(figure),
    }
    (args.out / "evaluation_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(scores.to_string(index=False), flush=True)
    print(figure, flush=True)


if __name__ == "__main__":
    main()
