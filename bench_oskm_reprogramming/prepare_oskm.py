#!/usr/bin/env python
"""Prepare matched old-control/OSKM and young-reference MERFISH subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


CONTROL = "old_notreatment"
TARGET = "old_treatment"
YOUNG = "young_notreatment"


def proportional_sample(
    frame: pd.DataFrame,
    candidates: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exactly size rows, preserving celltype x subregion proportions."""
    if size <= 0:
        return candidates.copy()
    if len(candidates) <= size:
        return candidates.copy()
    labels = (
        frame.iloc[candidates]["celltype"].astype(str)
        + "::"
        + frame.iloc[candidates]["subregion"].astype(str)
    ).to_numpy()
    values, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    expected = size * counts / counts.sum()
    quota = np.floor(expected).astype(int)
    remainder = size - quota.sum()
    if remainder:
        order = np.argsort(-(expected - quota), kind="stable")
        quota[order[:remainder]] += 1
    chosen = []
    for group_i, _ in enumerate(values):
        pool = candidates[inverse == group_i]
        if quota[group_i]:
            chosen.append(rng.choice(pool, quota[group_i], replace=False))
    return np.sort(np.concatenate(chosen).astype(np.int64))


def materialize(source: ad.AnnData, selected: np.ndarray) -> ad.AnnData:
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    counts = np.asarray(source.X[selected, :], dtype=np.float32)
    counts = sparse.csr_matrix(counts)
    obs = source.obs.iloc[selected].copy()
    var = source.var.copy()
    data = ad.AnnData(X=counts.copy(), obs=obs, var=var)
    data.layers["counts"] = counts.copy()
    data.obsm["spatial"] = np.asarray(source.obsm["spatial"])[selected].astype(
        np.float32
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "data" / "reprogramming.h5ad",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path(__file__).parent / "data" / "oskm_old_matched_160k.h5ad",
    )
    parser.add_argument(
        "--young-output",
        type=Path,
        default=Path(__file__).parent / "data" / "young_reference_20k.h5ad",
    )
    parser.add_argument(
        "--per-old-mouse", type=int, default=20_000,
        help="Maximum cells per old mouse; use 0 to retain every cell.",
    )
    parser.add_argument(
        "--per-young-mouse", type=int, default=5_000,
        help="Maximum cells per young mouse; use 0 to retain every cell.",
    )
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    source = ad.read_h5ad(args.input, backed="r")
    obs = source.obs
    rng = np.random.default_rng(args.seed)
    selected_old = []
    selected_young = []
    sampling = {}
    for mouse in sorted(obs["mouse_id"].astype(str).unique()):
        candidates = np.flatnonzero(obs["mouse_id"].astype(str).to_numpy() == mouse)
        cohort = str(obs.iloc[candidates[0]]["cohort"])
        if cohort in {CONTROL, TARGET}:
            take = proportional_sample(obs, candidates, args.per_old_mouse, rng)
            selected_old.append(take)
        elif cohort == YOUNG:
            take = proportional_sample(obs, candidates, args.per_young_mouse, rng)
            selected_young.append(take)
        else:
            continue
        sampling[mouse] = {
            "cohort": cohort,
            "available": int(len(candidates)),
            "selected": int(len(take)),
        }

    train = materialize(source, np.concatenate(selected_old))
    train.obs["condition"] = pd.Categorical(
        np.where(train.obs["cohort"].astype(str) == TARGET, "oskm", "control"),
        categories=["control", "oskm"],
    )
    train.obs["graph_id"] = train.obs["mouse_id"].astype(str).astype("category")
    train.obs["slice_id"] = train.obs["mouse_id"].astype(str).astype("category")
    train.obs["paired_batch"] = train.obs["slide_id"].astype(str).astype("category")
    train.obs["cell_type"] = train.obs["celltype"].astype(str).astype("category")
    train.obs["region_label"] = train.obs["subregion"].astype(str).astype("category")
    train.obs["perturbation"] = train.obs["condition"].copy()
    train.obs_names = pd.Index(
        [
            f"{mouse}:{name}"
            for mouse, name in zip(
                train.obs["mouse_id"].astype(str), train.obs_names.astype(str)
            )
        ]
    )

    young = materialize(source, np.concatenate(selected_young))
    young.obs["condition"] = pd.Categorical(["young_reference"] * young.n_obs)
    young.obs["graph_id"] = young.obs["mouse_id"].astype(str).astype("category")
    young.obs["slice_id"] = young.obs["mouse_id"].astype(str).astype("category")
    young.obs["cell_type"] = young.obs["celltype"].astype(str).astype("category")
    young.obs["region_label"] = young.obs["subregion"].astype(str).astype("category")
    young.obs_names = pd.Index(
        [
            f"{mouse}:{name}"
            for mouse, name in zip(
                young.obs["mouse_id"].astype(str), young.obs_names.astype(str)
            )
        ]
    )

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    train.write_h5ad(args.train_output, compression="gzip")
    young.write_h5ad(args.young_output, compression="gzip")

    batch_table = pd.crosstab(train.obs["paired_batch"], train.obs["condition"])
    report = {
        "source": str(args.input.resolve()),
        "training_output": str(args.train_output.resolve()),
        "young_reference_output": str(args.young_output.resolve()),
        "training_shape": [int(train.n_obs), int(train.n_vars)],
        "young_reference_shape": [int(young.n_obs), int(young.n_vars)],
        "condition_counts": train.obs["condition"].value_counts().astype(int).to_dict(),
        "paired_batch_by_condition": batch_table.astype(int).to_dict(),
        "sampling": sampling,
        "contract": {
            "training_contrast": "old_notreatment control vs old_treatment OSKM",
            "young_notreatment_role": "external evaluation only; absent from training",
            "all_old_mice_included": True,
            "sampling": (
                "none; every source cell retained"
                if args.per_old_mouse <= 0 and args.per_young_mouse <= 0
                else "proportional within mouse by celltype x subregion"
            ),
            "spatial_graph_boundary": "mouse_id",
            "decoder_batch_key": "slide_id (paired across both arms)",
            "training_supervision": "binary arm membership only",
        },
    }
    args.train_output.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
