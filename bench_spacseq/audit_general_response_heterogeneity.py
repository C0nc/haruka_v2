"""Test guide-independent general response heterogeneity in salient intrinsic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


MODULES = {
    "immediate_early_stress": ["Fos", "Fosb", "Jun", "Junb", "Jund", "Atf3", "Ddit3", "Hspa1a", "Hspa1b"],
    "interferon": ["Isg15", "Ifit1", "Ifit2", "Ifit3", "Irf7", "Stat1", "Oas1a", "Oas2"],
    "proliferation": ["Mki67", "Pcna", "Top2a", "Cdk1", "Ccnb1", "Ccnb2", "Mcm5"],
    "hypoxia_glycolysis": ["Hif1a", "Vegfa", "Slc2a1", "Ldha", "Pdk1", "Bnip3", "Aldoa"],
    "apoptosis": ["Bax", "Bbc3", "Pmaip1", "Casp3", "Casp8", "Bcl2l11"],
    "inflammatory_nfkb": ["Nfkbia", "Nfkb1", "Rel", "Tnfaip3", "Cxcl1", "Cxcl2", "Il6"],
    "emt_invasion": ["Vim", "Zeb1", "Zeb2", "Snai1", "Snai2", "Fn1", "Mmp2", "Mmp9", "Cdh2"],
}


def cross_fitted_residual(target, nuisance, seed=0):
    target = StandardScaler().fit_transform(target)
    nuisance = StandardScaler().fit_transform(nuisance)
    prediction = np.zeros_like(target)
    for train, test in KFold(5, shuffle=True, random_state=seed).split(target):
        model = Ridge(alpha=10.0).fit(nuisance[train], target[train])
        prediction[test] = model.predict(nuisance[test])
    return target - prediction


def leave_function_out_r2(x, y, functions):
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    prediction = np.zeros_like(y)
    folds = {}
    for train, test in LeaveOneGroupOut().split(x, y, functions):
        model = Ridge(alpha=10.0).fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
        folds[str(np.unique(functions[test])[0])] = float(r2_score(y[test], prediction[test]))
    return float(r2_score(y, prediction)), folds


def correlations(x, y):
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    return np.asarray([np.corrcoef(x[:, j], y)[0, 1] for j in range(x.shape[1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.data)
    root = Path(args.emb_dir)
    order = np.load(root / "obs_names.npy", allow_pickle=True).astype(str)
    adata = adata[order].copy()
    function = adata.obs.function.astype(str).to_numpy()
    guide = adata.obs.guide_identity.astype(str).to_numpy()
    slices = adata.obs.slice_id.astype(str).to_numpy()
    anchor = adata.obs.is_anchor.to_numpy(bool) & (function != "NT")
    function, guide, slices = function[anchor], guide[anchor], slices[anchor]

    reps = {
        name: np.load(root / f"{name}.npy")[anchor]
        for name in ("background", "spatial", "salient", "salient_spatial")
    }
    nuisance_categories = pd.get_dummies(
        pd.DataFrame({"guide": guide, "slice": slices}), drop_first=False,
        dtype=float,
    ).to_numpy()
    nuisance = np.column_stack([reps["salient_spatial"], nuisance_categories])
    reps["salient_intrinsic_general_residual"] = cross_fitted_residual(
        reps["salient"], nuisance, args.seed
    )

    counts = adata.layers["counts"][anchor]
    totals = np.asarray(counts.sum(1)).ravel().clip(min=1)
    expression = counts.multiply(1e4 / totals[:, None])
    if sparse.issparse(expression):
        expression.data = np.log1p(expression.data)
        expression = expression.toarray()
    else:
        expression = np.log1p(expression)
    gene_z = StandardScaler().fit_transform(expression)
    gene_index = {str(gene): i for i, gene in enumerate(adata.var_names)}
    module_scores, module_genes = {}, {}
    for module, genes in MODULES.items():
        present = [gene for gene in genes if gene in gene_index]
        module_genes[module] = present
        if len(present) >= 2:
            module_scores[module] = gene_z[:, [gene_index[g] for g in present]].mean(1)

    result = {
        "n_target_anchors": int(anchor.sum()),
        "n_functions": int(len(np.unique(function))),
        "n_guides": int(len(np.unique(guide))),
        "residualized_from": ["salient_spatial", "guide_identity", "slice"],
        "module_genes_present": module_genes,
        "module_prediction_leave_function_out": {},
        "residual_intrinsic_axes": {},
        "residual_intrinsic_clusters": {},
    }
    for module, score in module_scores.items():
        result["module_prediction_leave_function_out"][module] = {}
        for name, x in reps.items():
            overall, folds = leave_function_out_r2(x, score, function)
            corr = correlations(x, score)
            result["module_prediction_leave_function_out"][module][name] = {
                "R2": overall,
                "fold_R2": folds,
                "max_abs_dimension_correlation": float(np.max(np.abs(corr))),
            }

    residual = reps["salient_intrinsic_general_residual"]
    pca = PCA(min(4, residual.shape[1]), random_state=args.seed).fit(residual)
    axes = pca.transform(residual)
    for axis in range(axes.shape[1]):
        gene_corr = np.asarray([
            np.corrcoef(axes[:, axis], gene_z[:, j])[0, 1]
            for j in range(gene_z.shape[1])
        ])
        order_gene = np.argsort(np.abs(gene_corr))[::-1][:15]
        module_corr = {
            module: float(np.corrcoef(axes[:, axis], score)[0, 1])
            for module, score in module_scores.items()
        }
        result["residual_intrinsic_axes"][f"PC{axis + 1}"] = {
            "variance_ratio": float(pca.explained_variance_ratio_[axis]),
            "top_expression_genes": [
                {"gene": str(adata.var_names[j]), "correlation": float(gene_corr[j])}
                for j in order_gene
            ],
            "module_correlations": module_corr,
        }

    z = StandardScaler().fit_transform(residual)
    candidates = []
    for k in range(2, 9):
        labels = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=10,
            reg_covar=1e-5, random_state=args.seed,
        ).fit_predict(z)
        candidates.append((silhouette_score(z, labels), k, labels))
    best_silhouette, best_k, labels = max(candidates, key=lambda item: item[0])
    result["residual_intrinsic_clusters"] = {
        "selected_k": int(best_k),
        "silhouette": float(best_silhouette),
        "ARI_function": float(adjusted_rand_score(function, labels)),
        "ARI_guide": float(adjusted_rand_score(guide, labels)),
        "ARI_slice": float(adjusted_rand_score(slices, labels)),
        "cluster_sizes": {
            str(cluster): int((labels == cluster).sum())
            for cluster in np.unique(labels)
        },
        "module_means_z": {
            str(cluster): {
                module: float(StandardScaler().fit_transform(score.reshape(-1, 1))[labels == cluster].mean())
                for module, score in module_scores.items()
            }
            for cluster in np.unique(labels)
        },
    }
    output = Path(args.out)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
