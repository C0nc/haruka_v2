"""Asymmetric CellCharter readout for PEtracer background and lineage response.

z-CC uses all cells, three neighbor hops, and excludes the focal-cell block.
s-CC uses malignant cells only, one neighbor hop, retains the focal-cell block,
and is zero for control and non-malignant cells.
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
    a: ad.AnnData,
    rep: np.ndarray,
    key: str,
    n_hops: int,
    sample_key: str,
    keep_own: bool,
) -> np.ndarray:
    import cellcharter as cc
    import squidpy as sq

    a.obsm[key] = np.asarray(rep, dtype=np.float32)
    sq.gr.spatial_neighbors(
        a,
        library_key=sample_key,
        coord_type="generic",
        delaunay=True,
        spatial_key="spatial",
    )
    cc.gr.remove_long_links(a)
    cc.gr.aggregate_neighbors(
        a,
        n_layers=n_hops,
        use_rep=key,
        out_key=f"{key}_cc",
        sample_key=sample_key,
    )
    full = np.asarray(a.obsm[f"{key}_cc"], dtype=np.float32)
    return full if keep_own else full[:, rep.shape[1] :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--input-emb", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-key", default="graph_id")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--target-label", default="disease")
    parser.add_argument("--celltype-key", default="gt_background_celltype")
    parser.add_argument("--salient-celltype", default="Malignant")
    parser.add_argument("--background-hops", type=int, default=3)
    parser.add_argument("--salient-hops", type=int, default=1)
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    for key in (args.sample_key, args.condition_key, args.celltype_key):
        if key not in a.obs:
            raise KeyError(f"Missing obs[{key!r}]")
    if "spatial" not in a.obsm:
        raise KeyError("Missing obsm['spatial']")

    names = np.load(args.input_emb / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Index(a.obs_names.astype(str)).get_indexer(names)
    if (order < 0).any():
        raise ValueError("Embedding observations do not align to AnnData")
    a = a[order].copy()
    z = np.load(args.input_emb / "background.npy").astype(np.float32)
    s = np.load(args.input_emb / "salient.npy").astype(np.float32)
    if z.shape[0] != a.n_obs or s.shape[0] != a.n_obs:
        raise ValueError("Embedding and AnnData row counts differ")

    target = a.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    salient_cells = a.obs[args.celltype_key].astype(str).to_numpy() == args.salient_celltype
    with Timer() as timer:
        z_cc = aggregate(
            a.copy(), z, "haruka_z", args.background_hops,
            args.sample_key, keep_own=False,
        )

        malignant = a[salient_cells].copy()
        s_malignant = s[salient_cells].copy()
        s_malignant[~target[salient_cells]] = 0.0
        s_malignant_cc = aggregate(
            malignant, s_malignant, "haruka_s", args.salient_hops,
            args.sample_key, keep_own=True,
        )
        s_cc = np.zeros((a.n_obs, s_malignant_cc.shape[1]), dtype=np.float32)
        s_cc[salient_cells] = s_malignant_cc

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
        method="haruka_asymmetric_cellcharter_lineage",
        source_result=str(args.input_emb.resolve()),
        zs_independence_penalty=source_meta.get("zs_independence_penalty"),
        sample_key=args.sample_key,
        background_cell_scope="all",
        background_hops=args.background_hops,
        background_own_block=False,
        salient_cell_scope=f"{args.celltype_key}={args.salient_celltype}",
        salient_hops=args.salient_hops,
        salient_own_block=True,
        salient_zero_control=True,
        salient_zero_nonmatching_celltype=True,
        supervised_labels=False,
        annotation_guided_graph=True,
        runtime_cellcharter_s=timer.seconds,
    )


if __name__ == "__main__":
    main()
