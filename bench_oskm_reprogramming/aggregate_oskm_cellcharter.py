#!/usr/bin/env python
"""Create the default asymmetric z-CC/s-CC readout for the OSKM study.

Background z-CC uses all cells, three spatial hops, and drops the focal-cell
block. Salient s-CC uses one hop plus the focal block within annotated cell
lineages, because OSKM response should be compared within lineage rather than
rediscovering background cell identity. Control salient latents are zeroed.
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
    data: ad.AnnData,
    representation: np.ndarray,
    key: str,
    n_hops: int,
    library_key: str,
    keep_own: bool,
) -> np.ndarray:
    import cellcharter as cc
    import squidpy as sq

    data.obsm[key] = np.asarray(representation, dtype=np.float32)
    sq.gr.spatial_neighbors(
        data,
        library_key=library_key,
        coord_type="generic",
        delaunay=True,
        spatial_key="spatial",
    )
    cc.gr.remove_long_links(data)
    cc.gr.aggregate_neighbors(
        data,
        n_layers=n_hops,
        use_rep=key,
        out_key=f"{key}_cc",
        sample_key=library_key,
    )
    result = np.asarray(data.obsm[f"{key}_cc"], dtype=np.float32)
    return result if keep_own else result[:, representation.shape[1] :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--input-emb", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-key", default="graph_id")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--target-label", default="oskm")
    parser.add_argument("--celltype-key", default="cell_type")
    parser.add_argument("--background-hops", type=int, default=3)
    parser.add_argument("--salient-hops", type=int, default=1)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    for key in (args.sample_key, args.condition_key, args.celltype_key):
        if key not in data.obs:
            raise KeyError(f"Missing obs[{key!r}]")
    names = np.load(args.input_emb / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Index(data.obs_names.astype(str)).get_indexer(names)
    if (order < 0).any():
        raise ValueError("Embedding observations do not align to AnnData")
    data = data[order].copy()
    z = np.load(args.input_emb / "background.npy").astype(np.float32)
    s = np.load(args.input_emb / "salient.npy").astype(np.float32)
    if z.shape[0] != data.n_obs or s.shape[0] != data.n_obs:
        raise ValueError("Embedding and AnnData row counts differ")

    target = data.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    celltype = np.asarray(
        data.obs[args.celltype_key].astype(str).to_numpy(), dtype=str
    )
    mouse = np.asarray(data.obs[args.sample_key].astype(str).to_numpy(), dtype=str)
    lineage_graph = np.char.add(np.char.add(mouse, "::"), celltype)
    counts = pd.Series(lineage_graph).value_counts()
    valid_graphs = set(counts.index[counts >= 4])
    valid = np.asarray([value in valid_graphs for value in lineage_graph])

    with Timer() as timer:
        z_cc = aggregate(
            data.copy(), z, "haruka_z", args.background_hops,
            args.sample_key, keep_own=False,
        )

        s_zeroed = s.copy()
        s_zeroed[~target] = 0.0
        s_cc = np.concatenate([s_zeroed, np.zeros_like(s_zeroed)], axis=1)
        if valid.any():
            salient_data = data[valid].copy()
            salient_data.obs["lineage_graph"] = pd.Categorical(lineage_graph[valid])
            aggregated = aggregate(
                salient_data,
                s_zeroed[valid],
                "haruka_s",
                args.salient_hops,
                "lineage_graph",
                keep_own=True,
            )
            s_cc[valid] = aggregated
        s_cc[~target] = 0.0

    source_meta = {}
    if (args.input_emb / "meta.json").exists():
        source_meta = json.loads((args.input_emb / "meta.json").read_text())
    save_embeddings(
        str(args.out),
        background=z,
        salient=s,
        spatial=z_cc,
        salient_spatial=s_cc,
        obs_names=data.obs_names.to_numpy(),
        method="haruka_asymmetric_cellcharter_oskm",
        source_result=str(args.input_emb.resolve()),
        zs_independence_penalty=source_meta.get("zs_independence_penalty"),
        batch_key=source_meta.get("batch_key"),
        sample_key=args.sample_key,
        background_cell_scope="all",
        background_hops=args.background_hops,
        background_own_block=False,
        salient_cell_scope=f"within {args.celltype_key} for every lineage",
        salient_hops=args.salient_hops,
        salient_own_block=True,
        salient_zero_control=True,
        annotation_guided_graph=True,
        small_lineage_graphs_own_only=int((~valid).sum()),
        runtime_cellcharter_s=timer.seconds,
    )


if __name__ == "__main__":
    main()
