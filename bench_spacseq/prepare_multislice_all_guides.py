"""Build a balanced multi-slice, multi-perturbation SPAC-seq dataset.

The three lung-metastasis sections are co-trained. Non-targeting guides define
the background arm; every sufficiently powered targeting gene defines a
separate condition class. Graphs are isolated by ``slice x perturbation`` so a
cell never receives a foreign-sgRNA neighbor.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


SLICES = {
    "M001": ("Lung_Metastasis_M001", "lm992"),
    "M002": ("Lung_Metastasis_M002", "lm994"),
    "M003": ("Lung_Metastasis_M003", "lm993"),
}


def guide_target(name: str) -> str:
    if name.lower().startswith("sgnon-targeting"):
        return "NT"
    match = re.match(r"^sg(.+)_([0-9]+)$", name)
    return match.group(1) if match else name.removeprefix("sg")


def load_slice(raw_root: Path, slice_id: str):
    directory, prefix = SLICES[slice_id]
    root = raw_root / directory
    guide = ad.read_h5ad(root / "filtered_guide_bc_matrix.h5")
    cells = pd.read_csv(root / f"{prefix}_cell_annotated.csv", index_col=0)
    niche = pd.read_csv(root / f"{prefix}_niche_annotated.csv", index_col=0)
    keep = guide.obs_names.isin(cells.index)
    guide = guide[keep].copy()
    cells = cells.reindex(guide.obs_names)
    counts = guide.X.tocsr()
    coords = np.column_stack(
        [guide.obs["array_col"].to_numpy(), guide.obs["array_row"].to_numpy()]
    ).astype(np.float32)
    nnz = np.diff(counts.indptr)
    single_feature = np.full(guide.n_obs, -1, dtype=np.int32)
    single = nnz == 1
    single_feature[single] = counts.indices[counts.indptr[:-1][single]]
    names = np.asarray(guide.var_names.astype(str), dtype=str)
    feature_target = np.asarray([guide_target(x) for x in names], dtype=object)
    tumor = cells["cell_type"].astype(str).to_numpy() == "Non-immune cell"
    return {
        "root": root,
        "guide": guide,
        "cells": cells,
        "niche": niche,
        "counts": counts,
        "coords": coords,
        "nnz": nnz,
        "single_feature": single_feature,
        "single": single,
        "names": names,
        "feature_target": feature_target,
        "tumor": tumor,
    }


def select_supported_targets(slice_data, min_anchors: int, require_guides: int):
    tables = []
    for slice_id, data in slice_data.items():
        idx = np.flatnonzero(data["single"] & data["tumor"])
        targets = data["feature_target"][data["single_feature"][idx]]
        guides = data["names"][data["single_feature"][idx]]
        frame = pd.DataFrame({"slice_id": slice_id, "target": targets, "guide": guides})
        tables.append(frame[frame.target != "NT"])
    anchors = pd.concat(tables, ignore_index=True)
    counts = anchors.groupby(["target", "slice_id"]).size().unstack(fill_value=0)
    guide_counts = anchors.groupby("target").guide.nunique()
    supported = counts.index[(counts.min(axis=1) >= min_anchors) &
                             (guide_counts.reindex(counts.index) >= require_guides)]
    return sorted(supported), counts, guide_counts


def balanced_anchor_indices(data, targets, n_per_group, seed):
    rng = np.random.default_rng(seed)
    feature = data["single_feature"]
    target = np.full(len(feature), "", dtype=object)
    valid = feature >= 0
    target[valid] = data["feature_target"][feature[valid]]
    eligible = data["single"] & data["tumor"]
    result = {}
    for label in ["NT", *targets]:
        idx = np.flatnonzero(eligible & (target == label))
        if len(idx) < n_per_group:
            raise ValueError(f"{label}: only {len(idx)} anchors, need {n_per_group}")
        result[label] = np.sort(rng.choice(idx, n_per_group, replace=False))
    return result


def assign_contexts(data, anchors, k):
    """Assign each annotated cell to its nearest eligible perturbation context."""
    coords = data["coords"]
    all_anchor = np.concatenate(list(anchors.values()))
    all_label = np.concatenate(
        [np.repeat(label, len(idx)) for label, idx in anchors.items()]
    )
    nbr = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=-1)
    nbr.fit(coords)
    distances, indices = nbr.kneighbors(coords[all_anchor])

    claims = defaultdict(list)
    feature = data["single_feature"]
    cell_target = np.full(len(feature), "", dtype=object)
    valid = feature >= 0
    cell_target[valid] = data["feature_target"][feature[valid]]
    no_guide = data["nnz"] == 0
    for anchor, label, dist, local in zip(all_anchor, all_label, distances, indices):
        allowed = no_guide[local] | (cell_target[local] == label)
        for cell, distance in zip(local[allowed], dist[allowed]):
            claims[int(cell)].append((float(distance), str(label), int(anchor)))

    assigned = {}
    for cell, candidates in claims.items():
        # The nearest eligible anchor owns overlaps. Lexical tie-breaking is stable.
        assigned[cell] = min(candidates, key=lambda x: (x[0], x[1], x[2]))[1]
    # An anchor always owns itself even if exact duplicate coordinates create a tie.
    for label, idx in anchors.items():
        for cell in idx:
            assigned[int(cell)] = label
    groups = {
        label: np.asarray(sorted(cell for cell, owner in assigned.items() if owner == label))
        for label in anchors
    }
    return groups


def build(args):
    raw_root = Path(args.raw_root)
    slice_data = {sid: load_slice(raw_root, sid) for sid in SLICES}
    targets, support, guide_counts = select_supported_targets(
        slice_data, args.min_anchors_per_slice, args.min_guides_per_target
    )
    if not targets:
        raise RuntimeError("no targeting genes satisfy the cross-slice support thresholds")

    support_min = int(support.loc[targets].min(axis=1).min())
    n_anchor = min(args.max_anchors_per_group, support_min)
    pieces = []
    audit = {
        "targets": targets,
        "n_targets": len(targets),
        "anchors_per_target_per_slice": n_anchor,
        "support_counts": support.loc[targets].astype(int).to_dict(orient="index"),
        "independent_guides": guide_counts.loc[targets].astype(int).to_dict(),
        "slices": {},
    }
    for slice_number, (slice_id, data) in enumerate(slice_data.items()):
        anchors = balanced_anchor_indices(
            data, targets, n_anchor, args.seed + 1009 * slice_number
        )
        groups = assign_contexts(data, anchors, args.k)
        selected = np.concatenate([groups[label] for label in ["NT", *targets]])
        labels = np.concatenate(
            [np.repeat(label, len(groups[label])) for label in ["NT", *targets]]
        )
        names = data["guide"].obs_names[selected]
        rna = ad.read_h5ad(data["root"] / "filtered_gene_bc_matrix.h5", backed="r")
        piece = rna[names].to_memory()
        rna.file.close()
        piece.var_names_make_unique()
        feature = data["single_feature"][selected]
        guide_name = np.full(len(selected), "", dtype=object)
        has_guide = feature >= 0
        guide_name[has_guide] = data["names"][feature[has_guide]]
        anchor_set = np.concatenate(list(anchors.values()))
        piece.obs_names = pd.Index([f"{slice_id}:{x}" for x in names])
        piece.obs["slice_id"] = slice_id
        piece.obs["graph_id"] = [f"{slice_id}__{x}" for x in labels]
        piece.obs["condition"] = np.where(labels == "NT", "control", "disease")
        piece.obs["perturbation"] = labels
        piece.obs["guide_name"] = guide_name
        piece.obs["is_anchor"] = np.isin(selected, anchor_set)
        piece.obs["cell_type"] = data["cells"].reindex(names)["cell_type"].fillna("Unknown").astype(str).to_numpy()
        piece.obs["niche"] = data["niche"].reindex(names)["niche"].fillna("Unknown").astype(str).to_numpy()
        piece.obsm["spatial"] = data["coords"][selected]
        piece.layers["counts"] = piece.X.copy()
        pieces.append(piece)
        audit["slices"][slice_id] = {
            "cells": int(piece.n_obs),
            "groups": pd.Series(labels).value_counts().astype(int).to_dict(),
            "anchors": {label: int(len(idx)) for label, idx in anchors.items()},
        }

    adata = ad.concat(pieces, join="inner", merge="same", uns_merge="same")
    adata.layers["counts"] = adata.X.copy()
    for key in ["slice_id", "graph_id", "condition", "perturbation", "guide_name",
                "cell_type", "niche"]:
        adata.obs[key] = adata.obs[key].astype("category")
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=args.n_hvg, flavor="seurat_v3", layer="counts",
        batch_key="slice_id", subset=False
    )
    forced = [x for x in targets if x in adata.var_names]
    adata.var.loc[forced, "highly_variable"] = True
    rank = adata.var["highly_variable_rank"].fillna(np.inf)
    chosen = adata.var.index[adata.var.highly_variable].tolist()
    if len(chosen) > args.n_hvg:
        forced_set = set(forced)
        removable = sorted((x for x in chosen if x not in forced_set), key=lambda x: rank[x], reverse=True)
        for gene in removable[:len(chosen) - args.n_hvg]:
            adata.var.loc[gene, "highly_variable"] = False
    adata = adata[:, adata.var.highly_variable].copy()
    adata.X = adata.layers["counts"].copy()
    adata.uns["spac_multislice_audit"] = audit
    audit["n_cells"] = int(adata.n_obs)
    audit["n_genes"] = int(adata.n_vars)
    audit["condition_counts"] = adata.obs.condition.value_counts().astype(int).to_dict()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output, compression="gzip")
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"[save] {output}: {adata.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="bench_spacseq/raw")
    parser.add_argument("--out", default="bench_spacseq/data/spac_lung_all_slices_all_guides_k8_hvg3000.h5ad")
    parser.add_argument("--min_anchors_per_slice", type=int, default=30)
    parser.add_argument("--min_guides_per_target", type=int, default=2)
    parser.add_argument("--max_anchors_per_group", type=int, default=150)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n_hvg", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
