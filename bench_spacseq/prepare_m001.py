"""Prepare one SPAC-seq Visium HD section for the Haruka 2x2 model.

The contrast is a clean non-targeting (NT) microenvironment against a real
perturbation microenvironment.  Anchor bins must be
annotated tumor bins with exactly one detected guide.  An anchor contributes
itself and its eight nearest annotated bins, subject to these rules:

* NT context contains only no-guide or NT-only bins.
* Target context contains no-guide, NT, or the selected target, but excludes
  every other targeting guide.
* Cells claimed by both arms are removed rather than assigned arbitrarily.

The two arms remain disjoint graph libraries.  ``timepoint`` is deliberately
constant, so the decoder batch covariate cannot act as a condition proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def _guide_mask(names: np.ndarray, prefix: str) -> np.ndarray:
    return np.asarray(pd.Index(names).str.lower().str.startswith(prefix.lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir", default="bench_spacseq/raw/Lung_Metastasis_M001"
    )
    parser.add_argument(
        "--out", default="bench_spacseq/data/spac_m001_cttn_matched_k8_hvg2000.h5ad"
    )
    parser.add_argument("--target", default="Cttn")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n_hvg", type=int, default=2000)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    guide_path = raw_dir / "filtered_guide_bc_matrix.h5"
    rna_path = raw_dir / "filtered_gene_bc_matrix.h5"
    cell_path = raw_dir / "lm992_cell_annotated.csv"
    niche_path = raw_dir / "lm992_niche_annotated.csv"

    guide = ad.read_h5ad(guide_path)
    cell = pd.read_csv(cell_path, index_col=0)
    niche = pd.read_csv(niche_path, index_col=0)

    # Work only with bins that the authors annotated as cells.  The original
    # H5AD barcode order is preserved so selected names index the RNA file.
    guide = guide[guide.obs_names.isin(cell.index)].copy()
    cell = cell.reindex(guide.obs_names)
    counts = guide.X.tocsr()
    coords = np.column_stack(
        [guide.obs["array_col"].to_numpy(), guide.obs["array_row"].to_numpy()]
    ).astype(np.float32)

    names = np.asarray(guide.var_names.astype(str), dtype=str)
    is_nt = _guide_mask(names, "sgnon-targeting")
    is_target = _guide_mask(names, f"sg{args.target}_")
    if is_nt.sum() == 0:
        raise ValueError("no non-targeting guides found")
    if is_target.sum() < 2:
        raise ValueError(
            f"target {args.target!r} must have at least two independent guides; "
            f"found {int(is_target.sum())}"
        )

    guide_nnz = np.diff(counts.indptr)
    single = guide_nnz == 1
    single_feature = np.full(guide.n_obs, -1, dtype=np.int32)
    single_feature[single] = counts.indices[counts.indptr[:-1][single]]
    safe_feature = np.maximum(single_feature, 0)
    is_tumor = cell["cell_type"].to_numpy() == "Non-immune cell"

    nt_anchor = single & is_nt[safe_feature] & is_tumor
    target_anchor = single & is_target[safe_feature] & is_tumor
    nt_centers = np.flatnonzero(nt_anchor)
    target_candidates = np.flatnonzero(target_anchor)
    if len(nt_centers) == 0 or len(target_candidates) == 0:
        raise ValueError(
            f"empty anchor arm: NT={len(nt_centers)}, target={len(target_candidates)}"
        )

    # Minimum-cost one-to-one matching uses spatial position only.  It prevents
    # the selected perturbation clone's broad location from trivially defining
    # condition while leaving expression completely untouched by selection.
    match_cost = cdist(coords[nt_centers], coords[target_candidates])
    nt_match, target_match = linear_sum_assignment(match_cost)
    nt_centers = nt_centers[nt_match]
    target_centers = target_candidates[target_match]

    any_targeting = np.asarray(counts[:, ~is_nt].getnnz(axis=1)).ravel() > 0
    any_nonselected_guide = np.asarray(
        counts[:, ~is_target].getnnz(axis=1)
    ).ravel() > 0

    all_centers = np.concatenate([nt_centers, target_centers])
    neighbors = NearestNeighbors(
        n_neighbors=args.k + 1, algorithm="kd_tree", n_jobs=-1
    ).fit(coords).kneighbors(coords[all_centers], return_distance=False)
    center_row = {cell_index: row for row, cell_index in enumerate(all_centers)}

    control_claims = np.zeros(guide.n_obs, dtype=np.int32)
    target_claims = np.zeros(guide.n_obs, dtype=np.int32)
    for center in nt_centers:
        local = neighbors[center_row[center]]
        local = local[~any_targeting[local]]
        np.add.at(control_claims, local, 1)
    for center in target_centers:
        local = neighbors[center_row[center]]
        local = local[~any_nonselected_guide[local]]
        np.add.at(target_claims, local, 1)

    ambiguous = (control_claims > 0) & (target_claims > 0)
    control_index = np.flatnonzero((control_claims > 0) & ~ambiguous)
    target_index = np.flatnonzero((target_claims > 0) & ~ambiguous)
    if np.intersect1d(control_index, target_index).size:
        raise RuntimeError("arm overlap remained after ambiguity removal")
    if not np.all(np.isin(nt_centers, control_index)):
        raise RuntimeError("an NT anchor was lost from the control arm")
    if not np.all(np.isin(target_centers, target_index)):
        raise RuntimeError("a target anchor was lost from the condition arm")

    selected_index = np.concatenate([control_index, target_index])
    selected_names = guide.obs_names[selected_index]
    rna_backed = ad.read_h5ad(rna_path, backed="r")
    if not selected_names.isin(rna_backed.obs_names).all():
        raise RuntimeError("selected guide barcodes are missing from RNA data")
    adata = rna_backed[selected_names].to_memory()
    rna_backed.file.close()
    adata.var_names_make_unique()

    n_control = len(control_index)
    condition = np.where(
        np.arange(adata.n_obs) < n_control, "control", "disease"
    )
    selected_single = single_feature[selected_index]
    selected_guide = np.full(adata.n_obs, "", dtype=object)
    has_single = selected_single >= 0
    selected_guide[has_single] = names[selected_single[has_single]]

    adata.obs["condition"] = pd.Categorical(
        condition, categories=["control", "disease"]
    )
    adata.obs["slice_id"] = pd.Categorical(
        np.where(condition == "control", "M001_NT", f"M001_{args.target}")
    )
    adata.obs["timepoint"] = pd.Categorical(np.repeat("M001", adata.n_obs))
    adata.obs["cell_type"] = pd.Categorical(
        cell.reindex(selected_names)["cell_type"].fillna("Unknown").astype(str)
    )
    adata.obs["niche"] = pd.Categorical(
        niche.reindex(selected_names)["niche"].fillna("Unknown").astype(str)
    )
    adata.obs["guide_name"] = pd.Categorical(selected_guide.astype(str))
    adata.obs["is_anchor"] = np.concatenate(
        [np.isin(control_index, nt_centers), np.isin(target_index, target_centers)]
    )
    adata.obs["context_claims"] = np.concatenate(
        [control_claims[control_index], target_claims[target_index]]
    )
    adata.obs["guide_nnz"] = guide_nnz[selected_index]
    adata.obs["guide_counts"] = np.asarray(counts[selected_index].sum(axis=1)).ravel()
    adata.obsm["spatial"] = coords[selected_index]

    # Preserve the raw Poisson count contract through HVG selection.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(args.n_hvg, adata.n_vars),
        flavor="seurat_v3", layer="counts"
    )
    # Always retain the perturbed transcript as a direct cell-autonomous readout
    # while keeping the feature budget fixed.
    target_gene = np.asarray(
        [name == args.target or name.startswith(f"{args.target}-")
         for name in adata.var_names.astype(str)]
    )
    if not target_gene.any():
        raise ValueError(f"target transcript {args.target!r} is absent from RNA data")
    hvg = adata.var["highly_variable"].to_numpy().copy()
    additions = target_gene & ~hvg
    n_add = int(additions.sum())
    if n_add:
        candidates = np.flatnonzero(hvg & ~target_gene)
        rank = adata.var["highly_variable_rank"].to_numpy()
        drop = candidates[np.argsort(np.nan_to_num(rank[candidates], nan=np.inf))[-n_add:]]
        hvg[drop] = False
        hvg[target_gene] = True
        adata.var["highly_variable"] = hvg
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.layers["counts"].copy()

    target_anchor_guides = pd.Series(
        names[single_feature[target_centers]]
    ).value_counts().sort_index().to_dict()
    design = {
        "source": "SPAC-seq Lung_Metastasis_M001 (library 992)",
        "target": args.target,
        "k": int(args.k),
        "control_rule": "single NT tumor anchor + kNN without any targeting guide",
        "condition_rule": (
            f"spatially matched single sg{args.target} tumor anchor + kNN "
            "without any foreign guide"
        ),
        "anchor_matching": "minimum-cost one-to-one Euclidean coordinate matching",
        "ambiguous_overlap_removed": int(ambiguous.sum()),
        "nt_anchor_count": int(len(nt_centers)),
        "target_anchor_count": int(len(target_centers)),
        "target_anchor_candidates": int(len(target_candidates)),
        "target_anchor_guides": json.dumps(target_anchor_guides, sort_keys=True),
        "control_cells": int(len(control_index)),
        "condition_cells": int(len(target_index)),
        "annotated_source_cells": int(guide.n_obs),
        "training_layer": "counts",
        "count_contract_validated": True,
    }
    adata.uns["spacseq_design"] = design

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")
    print(json.dumps({"out": str(out), "shape": adata.shape, **design}, indent=2))


if __name__ == "__main__":
    main()
