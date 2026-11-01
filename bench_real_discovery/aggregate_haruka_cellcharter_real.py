"""Build slice-bounded CellCharter readouts of Haruka z and s latents."""

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
    a: ad.AnnData, rep: np.ndarray, key: str, n_hops: int, sample_key: str
) -> np.ndarray:
    import cellcharter as cc
    import squidpy as sq

    a.obsm[key] = np.asarray(rep, dtype=np.float32)
    if "spatial_connectivities" not in a.obsp:
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
    return full[:, rep.shape[1] :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input-emb", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-hops", type=int, default=3)
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--target-label", default="disease")
    parser.add_argument("--sample-key", default="slice_id")
    args = parser.parse_args()

    a = ad.read_h5ad(args.data)
    if args.sample_key not in a.obs:
        raise KeyError(f"CellCharter aggregation requires obs[{args.sample_key!r}]")
    if "spatial" not in a.obsm:
        raise KeyError("CellCharter aggregation requires obsm['spatial']")

    source = Path(args.input_emb)
    names = np.load(source / "obs_names.npy", allow_pickle=True).astype(str)
    missing = pd.Index(names).difference(a.obs_names.astype(str))
    if len(missing):
        raise KeyError(f"Embedding cells absent from data: {missing[:5].tolist()}")
    order = pd.Series(np.arange(a.n_obs), index=a.obs_names.astype(str)).loc[names].to_numpy()
    a = a[order].copy()
    z = np.load(source / "background.npy").astype(np.float32)
    s = np.load(source / "salient.npy").astype(np.float32)
    if z.shape[0] != a.n_obs or s.shape[0] != a.n_obs:
        raise ValueError("Embedding and observation row counts differ")

    target = a.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    s_for_neighbors = s.copy()
    s_for_neighbors[~target] = 0.0
    with Timer() as timer:
        z_cc = aggregate(a, z, "haruka_z", args.n_hops, args.sample_key)
        s_cc = aggregate(a, s_for_neighbors, "haruka_s", args.n_hops, args.sample_key)

    source_meta = {}
    if (source / "meta.json").exists():
        source_meta = json.loads((source / "meta.json").read_text())
    penalty = source_meta.get("zs_independence_penalty")
    method = (
        f"haruka_lambda{float(penalty):g}_cellcharter_real"
        if penalty is not None
        else "haruka_cellcharter_real"
    )
    save_embeddings(
        args.out,
        background=z,
        salient=s,
        spatial=z_cc,
        salient_spatial=s_cc,
        obs_names=a.obs_names.to_numpy(),
        method=method,
        source_result=str(source.resolve()),
        source_method=source_meta.get("method"),
        zs_independence_penalty=source_meta.get("zs_independence_penalty"),
        batch_injection_mode=source_meta.get("batch_injection_mode"),
        runtime_cellcharter_s=timer.seconds,
        sample_key=args.sample_key,
        n_hops=args.n_hops,
        own_block_removed=True,
        control_salient_zero_before_aggregation=True,
        all_samples_joint=True,
        holdout=None,
    )


if __name__ == "__main__":
    main()
