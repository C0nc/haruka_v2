"""Symmetric configurable-hop CellCharter readout within matched graphs.

For both Haruka z and s, CellCharter concatenates the focal representation and
the mean representation at each requested graph distance. The spatial graph
is constructed separately for every sample-by-condition group.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench_merfish"))
from common import Timer, save_embeddings  # noqa: E402


def aggregate(
    a: ad.AnnData, rep: np.ndarray, key: str, group_key: str, n_layers: int
) -> np.ndarray:
    import cellcharter as cc

    a.obsm[key] = np.asarray(rep, dtype=np.float32)
    output = cc.gr.aggregate_neighbors(
        a, n_layers=n_layers, aggregations="mean", connectivity_key="spatial",
        use_rep=key, sample_key=group_key, copy=True,
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
    parser.add_argument("--n-layers", type=int, default=3)
    args = parser.parse_args()

    import cellcharter as cc
    import squidpy as sq

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

    group_key = "_cc_sample_condition"
    a.obs[group_key] = (
        a.obs[args.sample_key].astype(str) + "::"
        + a.obs[args.condition_key].astype(str)
    ).astype("category")
    sq.gr.spatial_neighbors(
        a, library_key=group_key, coord_type="generic", delaunay=True,
        spatial_key="spatial",
    )
    cc.gr.remove_long_links(a)
    adjacency = a.obsp["spatial_connectivities"].tocsr()
    rows, cols = adjacency.nonzero()
    groups = a.obs[group_key].astype(str).to_numpy()
    degree = np.diff(adjacency.indptr)
    graph_audit = {
        "n_edges": int(adjacency.nnz),
        "degree_min": int(degree.min()),
        "degree_median": float(np.median(degree)),
        "degree_max": int(degree.max()),
        "cross_sample_condition_edges": int(np.sum(groups[rows] != groups[cols])),
        "self_edges": int(np.sum(rows == cols)),
    }

    target = a.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    s_for_neighbors = s.copy()
    s_for_neighbors[~target] = 0.0
    with Timer() as timer:
        z_cc = aggregate(a, z, "haruka_z", group_key, args.n_layers)
        s_cc = aggregate(a, s_for_neighbors, "haruka_s", group_key, args.n_layers)

    source_meta = {}
    if (args.input_emb / "meta.json").exists():
        source_meta = json.loads((args.input_emb / "meta.json").read_text())
    save_embeddings(
        str(args.out), background=z, salient=s, spatial=z_cc,
        salient_spatial=s_cc, obs_names=a.obs_names.to_numpy(),
        method=f"haruka_symmetric_condition_cellcharter_{args.n_layers}hop",
        source_result=str(args.input_emb.resolve()),
        source_method=source_meta.get("method"),
        zs_independence_penalty=source_meta.get("zs_independence_penalty"),
        batch_injection_mode=source_meta.get("batch_injection_mode"),
        base_sample_key=args.sample_key,
        graph_grouping=f"{args.sample_key} x {args.condition_key}",
        graph_type="Delaunay with CellCharter long-link removal",
        n_layers=args.n_layers,
        included_blocks=["self"] + [
            f"hop{layer}_mean" for layer in range(1, args.n_layers + 1)
        ],
        identical_graph_for_z_and_s=True,
        identical_readout_for_z_and_s=True,
        control_salient_zero_before_aggregation=True,
        graph_audit=graph_audit,
        runtime_cellcharter_s=timer.seconds,
    )
    print(json.dumps(graph_audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
