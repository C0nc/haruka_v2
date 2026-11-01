#!/usr/bin/env python3
"""Build symmetric three-hop z-CC and s-CC using condition cells only."""

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
from common import load_embeddings, save_embeddings  # noqa: E402
from run_baselines_2x2 import _cc_aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-key", default="condition_label")
    parser.add_argument("--condition-value", default="KRASi_treated")
    parser.add_argument("--slice-key", default="slice_id")
    args = parser.parse_args()

    source = ad.read_h5ad(args.data, backed="r")
    obs = source.obs[[args.condition_key, args.slice_key]].copy()
    obs.index = obs.index.astype(str)
    spatial = np.asarray(source.obsm["spatial"], dtype=np.float32)
    source.file.close()

    loaded = load_embeddings(args.input_dir)
    saved_names = pd.Index(loaded["obs_names"].astype(str))
    source_rows = obs.index.get_indexer(saved_names)
    if (source_rows < 0).any():
        raise ValueError("Haruka latent rows do not align with the source data")
    keep = obs.iloc[source_rows][args.condition_key].astype(str).eq(args.condition_value).to_numpy()
    kept_rows = source_rows[keep]
    kept_names = saved_names.to_numpy()[keep]

    graph = ad.AnnData(
        X=np.zeros((keep.sum(), 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"slice_id": obs.iloc[kept_rows][args.slice_key].astype(str).to_numpy()},
            index=kept_names,
        ),
    )
    graph.obs["slice_id"] = pd.Categorical(graph.obs["slice_id"])
    graph.obsm["spatial"] = spatial[kept_rows]
    z = np.asarray(loaded["background"], dtype=np.float32)[keep]
    s = np.asarray(loaded["salient"], dtype=np.float32)[keep]
    z_cc = _cc_aggregate(graph, z, "haruka_z_condition")
    s_cc = _cc_aggregate(graph, s, "haruka_s_condition")

    save_embeddings(
        args.output_dir,
        background=z,
        salient=s,
        spatial=z_cc,
        salient_spatial=s_cc,
        obs_names=kept_names,
        **{
            **loaded["meta"],
            "method": "haruka_z_s_cellcharter_condition_only",
            "source_result": str(args.input_dir.resolve()),
            "condition_key": args.condition_key,
            "condition_value": args.condition_value,
            "sample_key": args.slice_key,
            "n_hops": 3,
            "spatial_readout": "symmetric three-hop CellCharter aggregation on condition cells only; own block removed",
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "source_rows.npy", kept_rows.astype(np.int64))
    diagnostics = {
        "n_source": int(len(obs)),
        "n_latent": int(len(saved_names)),
        "n_condition": int(keep.sum()),
        "condition_key": args.condition_key,
        "condition_value": args.condition_value,
        "slice_counts": graph.obs["slice_id"].value_counts().sort_index().to_dict(),
    }
    (args.output_dir / "condition_only_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
