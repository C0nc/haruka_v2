#!/usr/bin/env python3
"""Exact, parallel deployment of Sun et al.'s frozen Spatial Aging Clock.

Parallelism is only across celltype x mouse_id groups.  This is algebraically
identical to spatialclock.deploy.get_predictions because the published
implementation first subsets a cell type and SpatialSmooth then loops over
mouse_id without sharing edges or state across groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


CLOCK_NAME = "lasso_cv5_nalphas20_spatialsmooth_alpha08_neigh20"
G_COUNTS = None
G_SPATIAL = None
G_VARS = None
G_CLOCK_REPO = None
G_PARTIALS = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _worker(task: tuple[str, str, np.ndarray]) -> dict:
    """Run the official transforms for one independent celltype x mouse."""
    from threadpoolctl import threadpool_limits

    celltype, mouse_id, rows = task
    safe = hashlib.sha1(f"{celltype}\t{mouse_id}".encode()).hexdigest()[:16]
    partial = Path(G_PARTIALS) / f"{safe}.npz"
    if partial.exists():
        x = np.load(partial)
        return {"celltype": celltype, "mouse_id": mouse_id, "n": len(rows),
                "rows": x["rows"], "pred": x["pred"], "seconds": 0.0,
                "resumed": True}

    started = time.time()
    with threadpool_limits(limits=1):
        sys.path.insert(0, str(G_CLOCK_REPO))
        from spatialclock.utils import normalize_adata, spatial_smoothing_expression

        # Official deploy.py is run on a dense data.X; propagate() relies on
        # ndarray broadcasting in its convergence check.
        x = G_COUNTS[rows].toarray().astype(np.float32, copy=False)
        obs = pd.DataFrame({"mouse_id": np.repeat(mouse_id, len(rows))},
                           index=pd.Index([str(v) for v in rows]))
        group = ad.AnnData(X=x, obs=obs,
                           var=pd.DataFrame(index=pd.Index(G_VARS)))
        group.obsm["spatial"] = np.asarray(G_SPATIAL[rows], dtype=np.float64)

        # These are the exact defaults explicitly selected by deploy.get_predictions.
        normalize_adata(group, zscore=False)
        spatial_smoothing_expression(
            group, graph_method="fixed", adjacency_method="binary",
            group_obs="mouse_id", n_neighbors=20, alpha=0.8,
            max_iter=30, tol=1e-2,
        )

        clock_dir = Path(G_CLOCK_REPO) / "spatialclock" / "clocks"
        stem = f"{CLOCK_NAME}_{celltype}"
        impdf = pd.read_csv(clock_dir / "mean_impute_files" / f"{stem}.csv")
        clock_genes = impdf.columns.astype(str).to_numpy()
        var_index = pd.Index(G_VARS)
        gene_pos = var_index.get_indexer(clock_genes)
        missing = gene_pos < 0
        pred_x = np.empty((len(rows), len(clock_genes)), dtype=np.float64)
        if sparse.issparse(group.X):
            pred_x[:, ~missing] = group.X[:, gene_pos[~missing]].toarray()
        else:
            pred_x[:, ~missing] = np.asarray(group.X)[:, gene_pos[~missing]]
        if missing.any():
            pred_x[:, missing] = impdf.iloc[0, np.flatnonzero(missing)].to_numpy(float)
        with (clock_dir / f"{stem}.pkl").open("rb") as fh:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline = pickle.load(fh)
        pred = np.asarray(pipeline.predict(pred_x), dtype=np.float64)

    np.savez_compressed(partial, rows=rows, pred=pred)
    return {"celltype": celltype, "mouse_id": mouse_id, "n": len(rows),
            "rows": rows, "pred": pred, "seconds": time.time() - started,
            "resumed": False}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--clock-repo", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--restrict-cell-ids", type=Path)
    p.add_argument("--expected-cache", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partials = args.out / "partials"
    partials.mkdir(exist_ok=True)

    print("Loading full source AnnData...", flush=True)
    data = ad.read_h5ad(args.data)
    source_n = data.n_obs
    if args.restrict_cell_ids is not None:
        keep_df = pd.read_csv(args.restrict_cell_ids, usecols=["cell_id"])
        pos = pd.Index(data.obs_names.astype(str)).get_indexer(keep_df["cell_id"].astype(str))
        if (pos < 0).any() or len(np.unique(pos)) != len(pos):
            raise ValueError("Restricted cell IDs do not map one-to-one into source data")
        data = data[pos].copy()
    counts = data.layers["counts"]
    counts = counts.tocsr().astype(np.float32) if sparse.issparse(counts) else sparse.csr_matrix(np.asarray(counts, dtype=np.float32))
    spatial = np.asarray(data.obsm["spatial"], dtype=np.float64)
    celltypes = data.obs["celltype"].astype(str).to_numpy()
    mice = data.obs["mouse_id"].astype(str).to_numpy()
    clock_dir = args.clock_repo / "spatialclock" / "clocks"
    tasks = []
    unsupported = []
    for ct in sorted(np.unique(celltypes)):
        model = clock_dir / f"{CLOCK_NAME}_{ct}.pkl"
        if not model.exists():
            unsupported.append(ct)
            continue
        for mouse in sorted(np.unique(mice[celltypes == ct])):
            rows = np.flatnonzero((celltypes == ct) & (mice == mouse)).astype(np.int64)
            tasks.append((ct, mouse, rows))

    global G_COUNTS, G_SPATIAL, G_VARS, G_CLOCK_REPO, G_PARTIALS
    G_COUNTS = counts
    G_SPATIAL = spatial
    G_VARS = data.var_names.astype(str).to_numpy()
    G_CLOCK_REPO = str(args.clock_repo)
    G_PARTIALS = str(partials)

    pred = np.full(data.n_obs, np.nan, dtype=np.float64)
    logs = []
    ctx = mp.get_context("fork")
    print(f"Running {len(tasks)} independent official clock groups with {args.workers} workers", flush=True)
    with ctx.Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, tasks), start=1):
            pred[result.pop("rows")] = result.pop("pred")
            logs.append(result)
            print(f"[{i}/{len(tasks)}] {result['celltype']} {result['mouse_id']} n={result['n']} seconds={result['seconds']:.1f} resumed={result['resumed']}", flush=True)

    frame = data.obs[["condition", "slice_id", "paired_batch", "mouse_id", "celltype", "cell_type", "subregion", "age", "cohort"]].copy()
    frame.insert(0, "cell_id", data.obs_names.astype(str))
    frame["predicted_age"] = pred
    frame.to_csv(args.out / "spatial_clock_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(logs).sort_values(["celltype", "mouse_id"]).to_csv(args.out / "task_log.csv", index=False)

    overlap_rows = []
    for ct in sorted(np.unique(celltypes)):
        imp = clock_dir / "mean_impute_files" / f"{CLOCK_NAME}_{ct}.csv"
        if imp.exists():
            genes = pd.read_csv(imp, nrows=0).columns.astype(str)
            overlap_rows.append({"celltype": ct, "clock_genes": len(genes),
                                 "overlap_genes": int(pd.Index(genes).isin(data.var_names).sum()),
                                 "missing_genes": int((~pd.Index(genes).isin(data.var_names)).sum())})
    pd.DataFrame(overlap_rows).to_csv(args.out / "feature_overlap.csv", index=False)

    validation = None
    if args.expected_cache is not None:
        expected = pd.read_csv(args.expected_cache, usecols=["cell_id", "predicted_age"])
        joined = frame[["cell_id", "predicted_age"]].merge(expected, on="cell_id", suffixes=("_new", "_expected"), validate="one_to_one")
        diff = joined["predicted_age_new"] - joined["predicted_age_expected"]
        validation = {"n": int(len(joined)), "max_abs_difference": float(diff.abs().max()),
                      "mean_abs_difference": float(diff.abs().mean()),
                      "pearson": float(joined[["predicted_age_new", "predicted_age_expected"]].corr().iloc[0, 1])}
        pd.DataFrame([validation]).to_csv(args.out / "cache_validation.csv", index=False)

    repo_files = [args.clock_repo / "spatialclock" / "deploy.py", args.clock_repo / "spatialclock" / "utils.py"]
    metadata = {
        "source_data": str(args.data), "source_n": int(source_n), "analysis_n": int(data.n_obs),
        "clock_repo": str(args.clock_repo), "clock_name": CLOCK_NAME,
        "official_parameters": {"normalize_total": 250, "log1p": True, "smooth": True,
                                "graph_method": "fixed", "adjacency_method": "binary",
                                "group_obs": "mouse_id", "n_neighbors": 20, "alpha": 0.8,
                                "max_iter": 30, "tol": 0.01, "standardize": True},
        "parallelization_equivalence": "Only independent celltype x mouse_id groups are parallelized; no edges or smoothing state cross these groups in official deploy.py/utils.py.",
        "supported_celltypes": sorted(set(celltypes) - set(unsupported)),
        "unsupported_celltypes": unsupported,
        "coverage_n": int(np.isfinite(pred).sum()), "coverage_fraction": float(np.isfinite(pred).mean()),
        "workers": args.workers, "source_hashes": {str(x): sha256(x) for x in repo_files},
        "validation": validation,
    }
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
