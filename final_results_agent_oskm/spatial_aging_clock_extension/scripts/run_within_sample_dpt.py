#!/usr/bin/env python3
"""Deterministic s-CC diffusion pseudotime, independently within each OSKM slice."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


G_REP = None
G_SAMPLE = None
G_IDS = None
G_PARTIALS = None


def worker(sample: str) -> dict:
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from threadpoolctl import threadpool_limits
    import scanpy as sc

    partial = Path(G_PARTIALS) / f"{sample}.npz"
    rows = np.flatnonzero(G_SAMPLE == sample).astype(np.int64)
    if partial.exists():
        x = np.load(partial)
        return {"sample": sample, "rows": x["rows"], "cell_ids": x["cell_ids"],
                "dpt_min_pc1": x["dpt_min_pc1"], "dpt_max_pc1": x["dpt_max_pc1"],
                "local_pc1": x["local_pc1"], "root_min_cell_id": str(x["root_min_cell_id"]),
                "root_max_cell_id": str(x["root_max_cell_id"]), "n": len(rows),
                "seconds": 0.0, "resumed": True}
    started = time.time()
    with threadpool_limits(limits=1):
        x = np.asarray(G_REP[rows], dtype=np.float32)
        z = StandardScaler().fit_transform(x)
        pcs = PCA(n_components=30, svd_solver="randomized", random_state=0).fit_transform(z).astype(np.float32)
        a = ad.AnnData(X=pcs)
        sc.pp.neighbors(a, n_neighbors=30, n_pcs=30, random_state=0, method="umap", metric="euclidean")
        sc.tl.diffmap(a, n_comps=15, random_state=0)
        root_min = int(np.argmin(pcs[:, 0]))
        root_max = int(np.argmax(pcs[:, 0]))
        a.uns["iroot"] = root_min
        sc.tl.dpt(a, n_dcs=10)
        dpt_min = np.asarray(a.obs["dpt_pseudotime"], dtype=np.float32).copy()
        a.uns["iroot"] = root_max
        sc.tl.dpt(a, n_dcs=10)
        dpt_max = np.asarray(a.obs["dpt_pseudotime"], dtype=np.float32).copy()
    cell_ids = G_IDS[rows].astype(str)
    np.savez_compressed(partial, rows=rows, cell_ids=cell_ids, dpt_min_pc1=dpt_min,
                        dpt_max_pc1=dpt_max, local_pc1=pcs[:, 0],
                        root_min_cell_id=cell_ids[root_min], root_max_cell_id=cell_ids[root_max])
    return {"sample": sample, "rows": rows, "cell_ids": cell_ids,
            "dpt_min_pc1": dpt_min, "dpt_max_pc1": dpt_max,
            "local_pc1": pcs[:, 0], "root_min_cell_id": cell_ids[root_min],
            "root_max_cell_id": cell_ids[root_max], "n": len(rows),
            "seconds": time.time() - started, "resumed": False}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cc-dir", required=True, type=Path)
    p.add_argument("--umap-table", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partials = args.out / "partials"
    partials.mkdir(exist_ok=True)

    ids = np.load(args.cc_dir / "obs_names.npy", allow_pickle=True).astype(str)
    umap = pd.read_csv(args.umap_table, usecols=["cell_id", "slice_id"])
    if not np.array_equal(ids, umap["cell_id"].astype(str).to_numpy()):
        raise ValueError("CC rows and fixed-seed condition UMAP rows do not align")
    sample = umap["slice_id"].astype(str).to_numpy()
    rep = np.load(args.cc_dir / "salient_spatial.npy", mmap_mode="r")

    global G_REP, G_SAMPLE, G_IDS, G_PARTIALS
    G_REP, G_SAMPLE, G_IDS, G_PARTIALS = rep, sample, ids, str(partials)
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(worker, sorted(np.unique(sample))):
            print(f"{result['sample']} n={result['n']} seconds={result['seconds']:.1f} resumed={result['resumed']}", flush=True)
            results.append(result)

    dpt_min = np.full(len(ids), np.nan, dtype=np.float32)
    dpt_max = np.full(len(ids), np.nan, dtype=np.float32)
    pc1 = np.full(len(ids), np.nan, dtype=np.float32)
    logs = []
    for r in results:
        rows = r.pop("rows")
        r.pop("cell_ids")
        dpt_min[rows] = r.pop("dpt_min_pc1")
        dpt_max[rows] = r.pop("dpt_max_pc1")
        pc1[rows] = r.pop("local_pc1")
        logs.append(r)
    pd.DataFrame({"cell_id": ids, "slice_id": sample, "dpt_min_pc1": dpt_min,
                  "dpt_max_pc1": dpt_max, "within_slice_scc_pc1": pc1}).to_csv(
                      args.out / "within_sample_scc_dpt.csv.gz", index=False, compression="gzip")
    pd.DataFrame(logs).sort_values("sample").to_csv(args.out / "dpt_run_log.csv", index=False)
    meta = {"representation": "condition-only s-CC, 96 dimensions", "fit_scope": "separately within each OSKM slice",
            "preprocessing": "within-slice StandardScaler then randomized PCA(30), seed 0",
            "graph": "scanpy neighbors k=30, Euclidean, UMAP method, seed 0",
            "diffusion": "scanpy diffmap 15 components seed 0; DPT uses 10 components",
            "primary_root": "cell with minimum within-slice s-CC PC1",
            "sensitivity_root": "cell with maximum within-slice s-CC PC1",
            "orientation": "never flipped using predicted age or response; static DPT is descriptive",
            "n_cells": int(len(ids)), "samples": sorted(np.unique(sample).tolist())}
    (args.out / "dpt_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
