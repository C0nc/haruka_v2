"""Original CellCharter baseline for the Jill AD control_r1/disease_r1 pair.

The scVI encoder is trained jointly on both arms with ``slice_id`` as its batch
key. The spatial graph is slice-bounded, and CellCharter concatenates the focal
scVI latent with three mean-aggregated graph-hop layers. Evaluation is performed
separately by ``evaluate_ad_original_cellcharter.py`` on disease cells only.

The available Jill matrix is log1p(library-normalized) and has no raw-count
layer. We reconstruct the closest deterministic integer pseudo-count vector at
the observed library size before applying scVI's negative-binomial likelihood.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def closest_integer_counts(log_matrix: np.ndarray) -> tuple[sp.csr_matrix, dict]:
    abundance = np.expm1(np.asarray(log_matrix, dtype=np.float64))
    abundance[abundance < 0] = 0
    target_depth = np.rint(abundance.sum(axis=1)).astype(np.int64)
    counts = np.floor(abundance).astype(np.int32)
    remainder = target_depth - counts.sum(axis=1)
    if (remainder < 0).any():
        raise ValueError("Largest-remainder allocation received a negative remainder")
    # Process many cells per argpartition call. This is mathematically identical
    # to selecting each row's largest fractional components, but avoids tens of
    # thousands of small NumPy calls on dense STARmap matrices.
    chunk_size = 1024
    for start in range(0, counts.shape[0], chunk_size):
        stop = min(start + chunk_size, counts.shape[0])
        add = remainder[start:stop]
        max_add = int(add.max(initial=0))
        if max_add == 0:
            continue
        fraction = abundance[start:stop] - counts[start:stop]
        candidate = np.argpartition(fraction, -max_add, axis=1)[:, -max_add:]
        score = np.take_along_axis(fraction, candidate, axis=1)
        order = np.argsort(score, axis=1)[:, ::-1]
        chosen = np.take_along_axis(candidate, order, axis=1)
        mask = np.arange(max_add)[None, :] < add[:, None]
        row = np.repeat(np.arange(stop - start), add)
        counts[start + row, chosen[mask]] += 1
    reconstructed_depth = counts.sum(axis=1)
    audit = {
        "source_transform": "log1p library-normalized expression",
        "raw_integer_counts_available": False,
        "reconstruction": "closest deterministic integer vector after expm1",
        "source_expm1_depth_min": float(abundance.sum(axis=1).min()),
        "source_expm1_depth_median": float(np.median(abundance.sum(axis=1))),
        "source_expm1_depth_max": float(abundance.sum(axis=1).max()),
        "integer_depth_min": int(reconstructed_depth.min()),
        "integer_depth_median": float(np.median(reconstructed_depth)),
        "integer_depth_max": int(reconstructed_depth.max()),
        "mean_absolute_reconstruction_error": float(np.mean(np.abs(counts - abundance))),
    }
    return sp.csr_matrix(counts, dtype=np.float32), audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-key", default="slice_id")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import cellcharter as cc
    import scvi
    import squidpy as sq

    started = time.time()
    source = ad.read_h5ad(args.data)
    for key in (args.batch_key, "condition"):
        if key not in source.obs:
            raise KeyError(f"Missing obs[{key!r}]")
    if "spatial" not in source.obsm:
        raise KeyError("Missing obsm['spatial']")

    dense = source.layers["counts"] if "counts" in source.layers else source.X
    if sp.issparse(dense):
        dense = dense.toarray()
    pseudo_counts, reconstruction_audit = closest_integer_counts(dense)

    train = source.copy()
    train.layers["scvi_counts"] = pseudo_counts
    scvi.settings.seed = args.seed
    scvi.model.SCVI.setup_anndata(
        train,
        layer="scvi_counts",
        batch_key=args.batch_key,
    )
    model = scvi.model.SCVI(
        train,
        n_hidden=args.hidden,
        n_latent=args.latent_dim,
        n_layers=args.layers,
        gene_likelihood="nb",
        dispersion="gene",
    )
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=0.9,
        early_stopping=True,
        early_stopping_monitor="elbo_validation",
        accelerator="gpu",
        devices=1,
    )
    scvi_latent = model.get_latent_representation().astype(np.float32)

    train.obsm["X_scVI"] = scvi_latent
    sq.gr.spatial_neighbors(
        train,
        library_key=args.batch_key,
        coord_type="generic",
        delaunay=True,
        spatial_key="spatial",
    )
    cc.gr.remove_long_links(train)
    cellcharter = cc.gr.aggregate_neighbors(
        train,
        n_layers=args.hops,
        aggregations="mean",
        connectivity_key="spatial",
        use_rep="X_scVI",
        sample_key=args.batch_key,
        copy=True,
    ).astype(np.float32)

    adjacency = train.obsp["spatial_connectivities"].tocsr()
    rows, cols = adjacency.nonzero()
    batches = train.obs[args.batch_key].astype(str).to_numpy()
    condition_batch = pd.crosstab(
        train.obs[args.batch_key].astype(str), train.obs["condition"].astype(str)
    )
    confounded = bool((condition_batch.gt(0).sum(axis=1) == 1).all())

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "scvi.npy", scvi_latent)
    np.save(args.out / "cellcharter.npy", cellcharter)
    np.save(args.out / "obs_names.npy", train.obs_names.astype(str).to_numpy())
    model.save(args.out / "scvi_model", overwrite=True)
    meta = {
        "method": "original_cellcharter_scvi",
        "data": str(args.data.resolve()),
        "fit_cells": "control_r1 and disease_r1 jointly",
        "evaluation_cells": "disease_r1 only (performed downstream)",
        "scvi_batch_key": args.batch_key,
        "batch_condition_table": condition_batch.astype(int).to_dict(),
        "batch_is_fully_condition_confounded": confounded,
        "likelihood": "negative_binomial",
        "latent_dim": args.latent_dim,
        "hidden": args.hidden,
        "n_layers": args.layers,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "cellcharter_hops": args.hops,
        "cellcharter_blocks": ["self", "hop1_mean", "hop2_mean", "hop3_mean"],
        "graph_boundary": args.batch_key,
        "graph_edges": int(adjacency.nnz),
        "cross_batch_edges": int(np.sum(batches[rows] != batches[cols])),
        "runtime_seconds": float(time.time() - started),
        "count_reconstruction": reconstruction_audit,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
