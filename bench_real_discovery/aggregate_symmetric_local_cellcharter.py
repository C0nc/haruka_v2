"""Symmetric one-ring CellCharter readout for Haruka z and s latents.

Both slots use the same exact spatial k-nearest-neighbor graph.  The focal
cell is excluded, and the output is only the mean of the first neighbor ring.
Graphs are bounded by ``sample_key`` so cells from different slices/arms can
never contribute to one another.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench_merfish"))
from common import Timer, save_embeddings  # noqa: E402


def exact_local_graph(
    coordinates: np.ndarray, samples: np.ndarray, n_neighbors: int
) -> tuple[sp.csr_matrix, dict]:
    """Return a directed graph with exactly k nearby cells per focal cell."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    samples = np.asarray(samples).astype(str)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    neighbor_distances: list[np.ndarray] = []
    degrees = np.zeros(len(samples), dtype=np.int32)

    for sample in pd.unique(samples):
        index = np.flatnonzero(samples == sample)
        if len(index) < 2:
            continue
        k = min(n_neighbors, len(index) - 1)
        model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        model.fit(coordinates[index])
        distances, local_neighbors = model.kneighbors(coordinates[index])
        # sklearn returns the query itself first for this fitted/query matrix.
        local_neighbors = local_neighbors[:, 1 : k + 1]
        distances = distances[:, 1 : k + 1]
        rows.append(np.repeat(index, k))
        cols.append(index[local_neighbors.reshape(-1)])
        neighbor_distances.append(distances.reshape(-1))
        degrees[index] = k

    if not rows:
        raise ValueError("No sample has enough cells to construct a local graph")
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    distance = np.concatenate(neighbor_distances)
    graph = sp.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(len(samples), len(samples)),
    )
    audit = {
        "requested_neighbors": int(n_neighbors),
        "degree_min": int(degrees.min()),
        "degree_median": float(np.median(degrees)),
        "degree_max": int(degrees.max()),
        "neighbor_distance_median": float(np.median(distance)),
        "neighbor_distance_q95": float(np.quantile(distance, 0.95)),
        "cross_sample_edges": int(np.sum(samples[row] != samples[col])),
        "self_edges": int(np.sum(row == col)),
    }
    return graph, audit


def neighbor_mean(a: ad.AnnData, rep: np.ndarray, key: str, sample_key: str) -> np.ndarray:
    import cellcharter as cc

    a.obsm[key] = np.asarray(rep, dtype=np.float32)
    # A list containing only layer 1 tells CellCharter to return the immediate
    # neighbor mean without concatenating the focal-cell representation.
    output = cc.gr.aggregate_neighbors(
        a,
        n_layers=[1],
        aggregations="mean",
        connectivity_key="spatial",
        use_rep=key,
        sample_key=sample_key,
        copy=True,
    )
    return np.asarray(output, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--input-emb", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--target-label", default="disease")
    parser.add_argument("--n-neighbors", type=int, default=15)
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    for key in (args.sample_key, args.condition_key):
        if key not in a.obs:
            raise KeyError(f"Missing obs[{key!r}]")
    if "spatial" not in a.obsm:
        raise KeyError("Missing obsm['spatial']")

    names = np.load(args.input_emb / "obs_names.npy", allow_pickle=True).astype(str)
    positions = pd.Index(a.obs_names.astype(str)).get_indexer(names)
    if (positions < 0).any():
        raise ValueError("Embedding observations do not align to AnnData")
    a = a[positions].copy()
    z = np.load(args.input_emb / "background.npy").astype(np.float32)
    s = np.load(args.input_emb / "salient.npy").astype(np.float32)
    if z.shape[0] != a.n_obs or s.shape[0] != a.n_obs:
        raise ValueError("Embedding and AnnData row counts differ")

    samples = a.obs[args.sample_key].astype(str).to_numpy()
    graph, graph_audit = exact_local_graph(
        np.asarray(a.obsm["spatial"]), samples, args.n_neighbors
    )
    a.obsp["spatial_connectivities"] = graph

    target = a.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    s_for_neighbors = s.copy()
    s_for_neighbors[~target] = 0.0
    with Timer() as timer:
        # The identical graph and identical readout are used for both slots.
        z_cc = neighbor_mean(a, z, "haruka_z", args.sample_key)
        s_cc = neighbor_mean(a, s_for_neighbors, "haruka_s", args.sample_key)

    source_meta = {}
    if (args.input_emb / "meta.json").exists():
        source_meta = json.loads((args.input_emb / "meta.json").read_text())
    save_embeddings(
        str(args.out),
        background=z,
        salient=s,
        spatial=z_cc,
        salient_spatial=s_cc,
        obs_names=a.obs_names.to_numpy(),
        method="haruka_symmetric_local_cellcharter",
        source_result=str(args.input_emb.resolve()),
        source_method=source_meta.get("method"),
        zs_independence_penalty=source_meta.get("zs_independence_penalty"),
        batch_injection_mode=source_meta.get("batch_injection_mode"),
        sample_key=args.sample_key,
        n_neighbors=args.n_neighbors,
        n_layers=[1],
        aggregation="mean",
        own_block_included=False,
        z_cell_scope="all_cells",
        s_cell_scope="all_cells",
        identical_graph_for_z_and_s=True,
        control_salient_zero_before_aggregation=True,
        graph_audit=graph_audit,
        runtime_cellcharter_s=timer.seconds,
    )
    print(json.dumps(graph_audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
