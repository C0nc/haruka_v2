"""Evaluate Haruka background/lineage recovery on one PEtracer pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def cluster_scores(x: np.ndarray, y: np.ndarray, seed: int, protocol: str) -> dict:
    y = np.asarray(y).astype(str)
    x = np.asarray(x, dtype=np.float64)
    variable = np.var(x, axis=0) > 1e-10
    x = x[:, variable]
    k = len(np.unique(y))
    if not x.shape[1] or k < 2:
        return {"ari": float("nan"), "nmi": float("nan"), "silhouette": float("nan")}
    if protocol == "direct_kmeans":
        pred = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    else:
        raise KeyError(protocol)
    return {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "n_cells": int(len(y)),
        "n_clusters": int(k),
        "n_features": int(x.shape[1]),
        "prediction": pred,
    }


def align(adata: ad.AnnData, emb_dir: Path):
    names = np.load(emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
    positions = pd.Index(adata.obs_names.astype(str)).get_indexer(names)
    if (positions < 0).any():
        raise ValueError("Saved Haruka rows do not align to the paired dataset")
    return positions


def dense_log_cpm(matrix) -> np.ndarray:
    values = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    values = values.astype(np.float32)
    total = values.sum(1, keepdims=True)
    np.divide(values * 1e4, total, out=values, where=total > 0)
    np.log1p(values, out=values)
    return values


def two_dimensional(x: np.ndarray, seed: int) -> np.ndarray:
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float64))
    try:
        import umap

        return umap.UMAP(
            n_neighbors=30, min_dist=0.25, metric="cosine", random_state=seed
        ).fit_transform(x)
    except Exception:
        return PCA(n_components=2, random_state=seed).fit_transform(x)


def strip_prediction(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "prediction"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--emb-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    positions = align(adata, args.emb_dir)
    obs = adata.obs.iloc[positions].copy()
    coords = np.asarray(adata.obsm["spatial"])[positions]
    target = obs["condition"].astype(str).to_numpy() == "disease"
    eligible = target & obs["lineage_effect_eligible"].astype(bool).to_numpy()
    if not eligible.any():
        raise ValueError("No lineage-effect cells in the condition arm")
    representations = {
        "z_latent": np.load(args.emb_dir / "background.npy"),
        "s_latent": np.load(args.emb_dir / "salient.npy"),
        "z": np.load(args.emb_dir / "spatial.npy"),
        "s": np.load(args.emb_dir / "salient_spatial.npy"),
    }

    counts = adata.layers["counts"][positions]
    target_rows = np.flatnonzero(target)
    control_rows = np.flatnonzero(~target)
    if len(target_rows) != len(control_rows):
        raise ValueError("Expected equal control and condition arms")
    log_target = dense_log_cpm(counts[target_rows])
    log_control = dense_log_cpm(counts[control_rows])
    raw_pca = PCA(
        n_components=min(50, log_target.shape[1], len(log_target) - 1),
        random_state=args.seed,
    ).fit_transform(StandardScaler().fit_transform(log_target))
    delta = log_target - log_control
    delta_pca = PCA(
        n_components=min(50, delta.shape[1], len(delta) - 1),
        random_state=args.seed,
    ).fit_transform(StandardScaler().fit_transform(delta))

    truths = {
        "background_niche": obs["gt_background_niche"].astype(str).to_numpy(),
        "background_celltype": obs["gt_background_celltype"].astype(str).to_numpy(),
        "background_boundary": obs["gt_background_boundary_bin"].astype(str).to_numpy(),
        "lineage_family": obs["gt_lineage_family"].astype(str).to_numpy(),
        "lineage_clade": obs["gt_lineage_clade"].astype(str).to_numpy(),
    }
    masks = {
        "background_niche": target,
        "background_celltype": target,
        "background_boundary": target & (truths["background_boundary"] != "unknown"),
        "lineage_family": eligible,
        "lineage_clade": eligible,
    }
    all_scores = {}
    predictions = {}
    for protocol in ("direct_kmeans",):
        all_scores[protocol] = {}
        predictions[protocol] = {}
        for rep_name, representation in representations.items():
            all_scores[protocol][rep_name] = {}
            predictions[protocol][rep_name] = {}
            for truth_name, truth in truths.items():
                mask = masks[truth_name]
                score = cluster_scores(
                    representation[mask], truth[mask], args.seed, protocol
                )
                predictions[protocol][rep_name][truth_name] = score.pop("prediction", None)
                all_scores[protocol][rep_name][truth_name] = score

        condition_eligible = obs.iloc[target_rows]["lineage_effect_eligible"].astype(bool).to_numpy()
        for baseline_name, values in (
            ("raw_expression_pca", raw_pca),
            ("paired_delta_oracle", delta_pca),
        ):
            all_scores[protocol][baseline_name] = {}
            for truth_name in truths:
                if truth_name.startswith("background"):
                    local_mask = np.ones(len(target_rows), dtype=bool)
                    truth = obs.iloc[target_rows][
                        {
                            "background_niche": "gt_background_niche",
                            "background_celltype": "gt_background_celltype",
                            "background_boundary": "gt_background_boundary_bin",
                        }[truth_name]
                    ].astype(str).to_numpy()
                    if truth_name == "background_boundary":
                        local_mask &= truth != "unknown"
                else:
                    local_mask = condition_eligible
                    truth = obs.iloc[target_rows][
                        "gt_lineage_family" if truth_name == "lineage_family" else "gt_lineage_clade"
                    ].astype(str).to_numpy()
                score = cluster_scores(values[local_mask], truth[local_mask], args.seed, protocol)
                all_scores[protocol][baseline_name][truth_name] = strip_prediction(score)

    z_control = representations["z_latent"][~target]
    z_target = representations["z_latent"][target]
    paired_cosine = np.sum(z_control * z_target, axis=1) / np.maximum(
        np.linalg.norm(z_control, axis=1) * np.linalg.norm(z_target, axis=1), 1e-8
    )
    section = str(obs["section_id"].iloc[0])
    result = {
        "section": section,
        "n_condition_cells": int(target.sum()),
        "n_lineage_cells": int(eligible.sum()),
        "n_clades": int(len(np.unique(truths["lineage_clade"][eligible]))),
        "n_families": int(len(np.unique(truths["lineage_family"][eligible]))),
        "paired_background_cosine_mean": float(np.mean(paired_cosine)),
        "scores": all_scores,
        "primary": {
            "background_niche_ari_z": all_scores["direct_kmeans"]["z"]["background_niche"]["ari"],
            "lineage_family_ari_s": all_scores["direct_kmeans"]["s"]["lineage_family"]["ari"],
            "lineage_clade_ari_s": all_scores["direct_kmeans"]["s"]["lineage_clade"]["ari"],
            "z_to_lineage_family_leakage_ari": all_scores["direct_kmeans"]["z"]["lineage_family"]["ari"],
            "s_to_background_niche_leakage_ari": all_scores["direct_kmeans"]["s"]["background_niche"]["ari"],
        },
    }
    out_dir = args.out_dir or args.emb_dir / "petracer_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(result, handle, indent=2)

    # Condition-only UMAP/PCA geometry, with the same coordinates reused for labels.
    z2 = two_dimensional(representations["z"][target], args.seed)
    s2 = two_dimensional(representations["s"][target], args.seed + 1)
    target_obs = obs.iloc[target_rows]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    panels = (
        (axes[0, 0], z2, target_obs["gt_background_niche"].astype(str), "z-CC — background niche (condition only)"),
        (axes[0, 1], z2, target_obs["gt_lineage_family"].astype(str), "same z-CC — lineage family"),
        (axes[0, 2], coords[target], target_obs["gt_background_niche"].astype(str), "physical tissue — background niche"),
        (axes[1, 0], s2, target_obs["gt_lineage_family"].astype(str), "s-CC — lineage family (condition only)"),
        (axes[1, 1], s2, target_obs["gt_background_niche"].astype(str), "same s-CC — background niche"),
        (axes[1, 2], coords[target], target_obs["gt_lineage_family"].astype(str), "physical tissue — lineage family"),
    )
    for axis, xy, labels, title in panels:
        labels = labels.to_numpy() if hasattr(labels, "to_numpy") else np.asarray(labels)
        for value in np.unique(labels):
            take = labels == value
            if str(value) == "not_lineage_target":
                axis.scatter(xy[take, 0], xy[take, 1], s=1, c="lightgray", alpha=0.25, rasterized=True)
            else:
                axis.scatter(xy[take, 0], xy[take, 1], s=2.5, alpha=0.65, label=str(value), rasterized=True)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if len(np.unique(labels)) <= 20:
            axis.legend(markerscale=3, fontsize=6, frameon=False, ncol=2)
    fig.suptitle(f"PEtracer M1-T1 {section}: exact paired lineage simulation", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "background_lineage_umap_spatial.png", dpi=220)
    plt.close(fig)

    flat = []
    for protocol, by_rep in all_scores.items():
        for rep_name, by_truth in by_rep.items():
            for truth_name, score in by_truth.items():
                flat.append({"section": section, "protocol": protocol, "representation": rep_name, "truth": truth_name, **score})
    pd.DataFrame(flat).to_csv(out_dir / "all_clustering_scores.csv", index=False)
    print(json.dumps(result["primary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
