"""Rebuild MENDER spatial embeddings from saved decoder compositions."""

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
from mender_composition import aggregate_mender_compositions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--emb", required=True)
    parser.add_argument("--sample-key", default="graph_id")
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--scale", type=int, default=6)
    args = parser.parse_args()

    emb = Path(args.emb)
    data = ad.read_h5ad(args.data, backed="r")
    names = np.load(emb / "obs_names.npy", allow_pickle=True).astype(str)
    order = pd.Series(
        np.arange(data.n_obs), index=data.obs_names.astype(str)
    ).loc[names].to_numpy()
    coords = np.asarray(data.obsm["spatial"][order], dtype=np.float32)
    samples = data.obs.iloc[order][args.sample_key].astype(str).to_numpy()
    background = np.load(emb / "background_composition.npy")
    salient = np.load(emb / "salient_composition.npy")
    if not (len(coords) == len(background) == len(salient)):
        raise ValueError("Coordinate and saved-composition row counts differ")

    spatial, salient_spatial, diagnostics = aggregate_mender_compositions(
        coords,
        samples,
        background,
        salient,
        scale=args.scale,
        radius_neighbors=args.neighbors,
        return_diagnostics=True,
    )
    np.save(emb / "spatial.npy", spatial)
    np.save(emb / "salient_spatial.npy", salient_spatial)
    (emb / "mender_radius_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )

    meta_path = emb / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "mender_scale": args.scale,
            "mender_neighbors": args.neighbors,
            "mender_radius_mode": "cell_local_kth_neighbor",
            "spatial_readout": (
                "MENDER soft pseudo-state composition in non-overlapping rings; "
                "cell-local radius calibrated by kth-neighbor distance"
            ),
        }
    )
    meta.pop("mender_radius_quantile", None)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[save] {emb}: spatial{spatial.shape}, "
        f"salient_spatial{salient_spatial.shape}"
    )


if __name__ == "__main__":
    main()
