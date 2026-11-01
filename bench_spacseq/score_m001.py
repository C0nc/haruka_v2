"""Score Haruka's four slots on the matched SPAC-seq M001 contrast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def binary_auc(embedding, labels, groups=None, seed=0):
    labels = np.asarray(labels, dtype=int)
    if groups is None:
        splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(
            embedding, labels
        )
    else:
        splits = StratifiedGroupKFold(
            5, shuffle=True, random_state=seed
        ).split(embedding, labels, groups)
    scores = []
    for train, test in splits:
        if np.unique(labels[test]).size != 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        model.fit(embedding[train], labels[train])
        scores.append(roc_auc_score(labels[test], model.predict_proba(embedding[test])[:, 1]))
    return float(np.mean(scores)) if scores else float("nan")


def multiclass_balanced_accuracy(embedding, labels, seed=0):
    labels = LabelEncoder().fit_transform(np.asarray(labels).astype(str))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = []
    for train, test in cv.split(embedding, labels):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
        )
        model.fit(embedding[train], labels[train])
        scores.append(balanced_accuracy_score(labels[test], model.predict(embedding[test])))
    return float(np.mean(scores))


def gmm_ari(embedding, labels, n_components, seed=0):
    labels = LabelEncoder().fit_transform(np.asarray(labels).astype(str))
    x = StandardScaler().fit_transform(embedding)
    if x.shape[1] > 50:
        x = PCA(50, random_state=seed).fit_transform(x)
    pred = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=seed,
        reg_covar=1e-5, n_init=3
    ).fit_predict(x)
    return float(adjusted_rand_score(labels, pred))


def cross_guide_auc(embedding, obs, target, seed=0):
    anchor = obs["is_anchor"].to_numpy(dtype=bool)
    guide = obs["guide_name"].astype(str).to_numpy()
    target_guides = sorted(set(g for g in guide if g.startswith(f"sg{target}_")))
    nt_guides = sorted(set(g for g in guide if g.startswith("sgnon-targeting_")))
    if len(target_guides) != 2:
        return None, []
    nt_halves = [set(nt_guides[::2]), set(nt_guides[1::2])]
    scores = []
    for fold in range(2):
        train_target = target_guides[fold]
        test_target = target_guides[1 - fold]
        train = anchor & ((guide == train_target) | np.isin(guide, list(nt_halves[fold])))
        test = anchor & ((guide == test_target) | np.isin(guide, list(nt_halves[1 - fold])))
        y_train = (guide[train] == train_target).astype(int)
        y_test = (guide[test] == test_target).astype(int)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        model.fit(embedding[train], y_train)
        scores.append(roc_auc_score(y_test, model.predict_proba(embedding[test])[:, 1]))
    return float(np.mean(scores)), [float(x) for x in scores]


def guide_direction_cosine(embedding, obs, target):
    anchor = obs["is_anchor"].to_numpy(dtype=bool)
    guide = obs["guide_name"].astype(str).to_numpy()
    control = anchor & np.char.startswith(guide.astype(str), "sgnon-targeting_")
    center = embedding[control].mean(axis=0)
    directions = []
    target_guides = sorted(set(g for g in guide if g.startswith(f"sg{target}_")))
    if len(target_guides) != 2:
        return None
    for target_guide in target_guides:
        directions.append(
            embedding[anchor & (guide == target_guide)].mean(axis=0) - center
        )
    a, b = directions
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data)
    emb_dir = Path(args.emb_dir)
    saved_names = np.load(emb_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(saved_names, adata.obs_names.to_numpy().astype(str)):
        raise ValueError("embedding and dataset barcode order differ")
    reps = {
        name: np.load(emb_dir / f"{name}.npy")
        for name in ("background", "spatial", "salient", "salient_spatial")
    }
    if not all(np.isfinite(value).all() for value in reps.values()):
        raise ValueError("non-finite embedding detected")

    obs = adata.obs
    target = str(adata.uns["spacseq_design"]["target"])
    condition = (obs["condition"].astype(str).to_numpy() == "disease").astype(int)
    anchor = obs["is_anchor"].to_numpy(dtype=bool)
    context = ~anchor
    known_niche = obs["niche"].astype(str).to_numpy() != "Unknown"
    spatial = np.asarray(adata.obsm["spatial"])
    spatial_blocks = (
        (spatial[:, 0] // 64).astype(int).astype(str)
        + "_"
        + (spatial[:, 1] // 64).astype(int).astype(str)
    )

    result = {
        "data": str(args.data), "emb_dir": str(emb_dir),
        "n_cells": int(adata.n_obs), "n_anchor": int(anchor.sum()),
        "rep_widths": {k: int(v.shape[1]) for k, v in reps.items()},
    }
    # Raw-expression PCA is the non-disentangled information ceiling.
    count_matrix = adata.layers["counts"]
    count_dense = (
        count_matrix.toarray() if sparse.issparse(count_matrix)
        else np.asarray(count_matrix)
    ).astype(np.float32)
    library = count_dense.sum(axis=1, keepdims=True)
    log_expression = np.log1p(1e4 * count_dense / np.maximum(library, 1))
    reps["pca_expression"] = PCA(
        min(50, log_expression.shape[1]), random_state=args.seed
    ).fit_transform(StandardScaler().fit_transform(log_expression))
    result["rep_widths"]["pca_expression"] = int(reps["pca_expression"].shape[1])
    for name, embedding in reps.items():
        result[f"{name}__condition_AUROC"] = binary_auc(
            embedding, condition, seed=args.seed
        )
        block_auc = binary_auc(
            embedding, condition, groups=spatial_blocks, seed=args.seed
        )
        result[f"{name}__condition_spatial_block_AUROC"] = block_auc
        result[f"{name}__condition_spatial_block_symmetric_AUROC"] = float(
            max(block_auc, 1.0 - block_auc)
        )
        result[f"{name}__anchor_condition_AUROC"] = binary_auc(
            embedding[anchor], condition[anchor], seed=args.seed
        )
        result[f"{name}__context_condition_AUROC"] = binary_auc(
            embedding[context], condition[context], seed=args.seed
        )
        result[f"{name}__cell_type_balanced_accuracy"] = multiclass_balanced_accuracy(
            embedding, obs["cell_type"], seed=args.seed
        )
        result[f"{name}__niche_balanced_accuracy"] = multiclass_balanced_accuracy(
            embedding[known_niche], obs.loc[known_niche, "niche"], seed=args.seed
        )
        if name != "pca_expression":
            mean_cross, folds = cross_guide_auc(embedding, obs, target, seed=args.seed)
            result[f"{name}__cross_guide_AUROC"] = mean_cross
            result[f"{name}__cross_guide_folds"] = folds
            result[f"{name}__guide_direction_cosine"] = guide_direction_cosine(
                embedding, obs, target
            )

    n_cell_types = obs["cell_type"].nunique()
    result["background__cell_type_ARI"] = gmm_ari(
        reps["background"], obs["cell_type"], n_cell_types, args.seed
    )
    result["spatial__niche_ARI"] = gmm_ari(
        reps["spatial"][known_niche], obs.loc[known_niche, "niche"], 2, args.seed
    )
    result["salient__anchor_condition_ARI"] = gmm_ari(
        reps["salient"][anchor], condition[anchor], 2, args.seed
    )
    result["salient_spatial__context_condition_ARI"] = gmm_ari(
        reps["salient_spatial"][context], condition[context], 2, args.seed
    )
    result["pca_expression__cell_type_ARI"] = gmm_ari(
        reps["pca_expression"], obs["cell_type"], n_cell_types, args.seed
    )
    result["pca_expression__niche_ARI"] = gmm_ari(
        reps["pca_expression"][known_niche], obs.loc[known_niche, "niche"], 2,
        args.seed
    )
    result["pca_expression__anchor_condition_ARI"] = gmm_ari(
        reps["pca_expression"][anchor], condition[anchor], 2, args.seed
    )

    # Direct perturbation check on the target transcript, normalized per cell.
    gene_index = int(np.flatnonzero(adata.var_names.to_numpy() == target)[0])
    counts = adata.layers["counts"]
    target_count = counts[:, gene_index]
    target_count = target_count.toarray().ravel() if sparse.issparse(target_count) else np.asarray(target_count).ravel()
    library = np.asarray(counts.sum(axis=1)).ravel()
    expression = np.log1p(1e4 * target_count / np.maximum(library, 1))
    guide = obs["guide_name"].astype(str).to_numpy()
    target_guides = sorted(set(g for g in guide if g.startswith(f"sg{target}_")))
    groups = {"NT": anchor & np.char.startswith(guide.astype(str), "sgnon-targeting_")}
    groups.update({target_guide: anchor & (guide == target_guide)
                   for target_guide in target_guides})
    expression_key = f"{target}_expression"
    result[expression_key] = {
        key: {"n": int(mask.sum()), "mean_log1p_cptt": float(expression[mask].mean()),
              "detection_rate": float((target_count[mask] > 0).mean())}
        for key, mask in groups.items()
    }
    for target_guide in target_guides:
        stat = mannwhitneyu(
            expression[groups[target_guide]], expression[groups["NT"]],
            alternative="two-sided"
        )
        result[expression_key][target_guide]["vs_NT_mannwhitney_p"] = float(stat.pvalue)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
