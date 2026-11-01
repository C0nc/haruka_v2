"""Annotate the biological content of Haruka background intrinsic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


MODULES = {
    "epithelial": ["Krt8", "Krt18", "Krt19", "Epcam", "Lcn2", "Slc34a2"],
    "ecm_mesenchymal": ["Col1a1", "Col1a2", "Col3a1", "Col6a3", "Fn1", "Mmp2", "Dpt", "Tnc", "Thbs2"],
    "antigen_presentation": ["Cd74", "H2-Aa", "H2-Ab1", "H2-Eb1", "B2m"],
    "myeloid": ["C1qa", "C1qb", "C1qc", "Lyz2", "Apoe", "Tyrobp"],
    "lymphoid": ["Cd3d", "Cd3e", "Trbc1", "Nkg7", "Cd79a", "Ms4a1"],
    "proliferation": ["Mki67", "Pcna", "Top2a", "Cdk1", "Ccnb1", "Mcm5"],
    "hypoxia_glycolysis": ["Vegfa", "Slc2a1", "Ldha", "Pdk1", "Bnip3", "Aldoa"],
    "interferon": ["Isg15", "Ifit1", "Ifit2", "Ifit3", "Irf7", "Stat1"],
}


def leave_slice_out_r2(x, y, slices):
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    prediction = np.zeros_like(y)
    folds = {}
    for train, test in LeaveOneGroupOut().split(x, y, slices):
        prediction[test] = Ridge(alpha=10).fit(x[train], y[train]).predict(x[test])
        folds[str(np.unique(slices[test])[0])] = float(
            1 - np.square(y[test] - prediction[test]).sum()
            / np.square(y[test] - y[test].mean()).sum()
        )
    total = 1 - np.square(y - prediction).sum() / np.square(y - y.mean()).sum()
    return float(total), folds


def ranked(names, values, n=18):
    order = np.argsort(np.abs(values))[::-1][:n]
    return [{"gene": str(names[i]), "correlation": float(values[i])} for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.data)
    root = Path(args.emb_dir)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    background = np.load(root / "background.npy")
    background_z = StandardScaler().fit_transform(background)

    counts = adata.layers["counts"]
    totals = np.asarray(counts.sum(1)).ravel().clip(min=1)
    expression = counts.multiply(1e4 / totals[:, None])
    if sparse.issparse(expression):
        expression.data = np.log1p(expression.data)
        expression = expression.toarray()
    else:
        expression = np.log1p(expression)
    variable = expression.var(0) > 1e-8
    expression_z = StandardScaler().fit_transform(expression[:, variable])
    names = np.asarray(adata.var_names.astype(str))[variable]
    gene_index = {gene: i for i, gene in enumerate(names)}
    module_scores, present = {}, {}
    for module, genes in MODULES.items():
        keep = [gene for gene in genes if gene in gene_index]
        present[module] = keep
        if len(keep) >= 2:
            module_scores[module] = expression_z[:, [gene_index[g] for g in keep]].mean(1)

    pca = PCA(6, random_state=0).fit(background_z)
    axes = pca.transform(background_z)
    axis_result = {}
    for axis in range(axes.shape[1]):
        axis_z = StandardScaler().fit_transform(axes[:, [axis]]).ravel()
        correlation = expression_z.T @ axis_z / len(axis_z)
        axis_result[f"PC{axis + 1}"] = {
            "variance_ratio": float(pca.explained_variance_ratio_[axis]),
            "top_genes": ranked(names, correlation),
            "module_correlations": {
                module: float(np.corrcoef(axis_z, score)[0, 1])
                for module, score in module_scores.items()
            },
        }

    candidates = []
    for k in range(2, 11):
        label = GaussianMixture(
            k, covariance_type="diag", n_init=10, reg_covar=1e-5, random_state=0
        ).fit_predict(background_z)
        candidates.append((silhouette_score(background_z, label), k, label))
    best_silhouette, best_k, label = max(candidates, key=lambda item: item[0])
    obs = adata.obs
    cell_type = obs.cell_type.astype(str).to_numpy()
    niche = obs.niche.astype(str).to_numpy()
    slices = obs.slice_id.astype(str).to_numpy()
    condition = obs.condition.astype(str).to_numpy()
    cluster_summary = {}
    for cluster in np.unique(label):
        take = label == cluster
        types, type_count = np.unique(cell_type[take], return_counts=True)
        niches, niche_count = np.unique(niche[take], return_counts=True)
        cluster_summary[str(cluster)] = {
            "n": int(take.sum()),
            "top_cell_types": {
                str(types[i]): int(type_count[i])
                for i in np.argsort(type_count)[::-1][:4]
            },
            "niche_counts": {
                str(niches[i]): int(niche_count[i])
                for i in np.argsort(niche_count)[::-1]
            },
            "module_means_z": {
                module: float(StandardScaler().fit_transform(score.reshape(-1, 1))[take].mean())
                for module, score in module_scores.items()
            },
        }

    result = {
        "n_cells": int(adata.n_obs),
        "cell_type_counts": obs.cell_type.astype(str).value_counts().astype(int).to_dict(),
        "module_genes_present": present,
        "module_leave_slice_out_R2": {
            module: {
                "R2": leave_slice_out_r2(background, score, slices)[0],
                "fold_R2": leave_slice_out_r2(background, score, slices)[1],
            }
            for module, score in module_scores.items()
        },
        "background_axes": axis_result,
        "clusters": {
            "selected_k": int(best_k),
            "silhouette": float(best_silhouette),
            "ARI_cell_type": float(adjusted_rand_score(cell_type, label)),
            "ARI_niche": float(adjusted_rand_score(niche, label)),
            "ARI_slice": float(adjusted_rand_score(slices, label)),
            "ARI_condition": float(adjusted_rand_score(condition, label)),
            "summary": cluster_summary,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
