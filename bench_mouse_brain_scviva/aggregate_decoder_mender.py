"""MENDER-style multi-range aggregation of Haruka decoder compositions."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench_merfish"))
from common import load_embeddings, save_embeddings  # noqa: E402


def valid_composition(values: np.ndarray) -> np.ndarray:
    """Map unconstrained MSE predictions to nonnegative soft state memberships."""
    out = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    total = out.sum(1, keepdims=True)
    np.divide(out, total, out=out, where=total > 0)
    return out


def mender_soft_counts(coords, membership, scale, radius):
    """Concatenate raw soft state counts in non-overlapping MENDER rings."""
    tree = cKDTree(np.asarray(coords, dtype=np.float64))
    distances = tree.sparse_distance_matrix(
        tree, max_distance=scale * radius, output_type="coo_matrix"
    )
    result = []
    for ring in range(scale):
        left, right = ring * radius, (ring + 1) * radius
        take = (distances.data >= left) & (distances.data < right)
        adjacency = sp.csr_matrix(
            (np.ones(int(take.sum()), dtype=np.float32),
             (distances.row[take], distances.col[take])),
            shape=(len(coords), len(coords)),
        )
        result.append(np.asarray(adjacency @ membership, dtype=np.float32))
    return np.concatenate(result, axis=1)


def normalize_log_counts(out: np.ndarray) -> np.ndarray:
    """Apply MENDER's normalize-total and log1p transform to one slot."""
    out = np.asarray(out, dtype=np.float32).copy()
    total = out.sum(1, keepdims=True)
    np.divide(out * 1e4, total, out=out, where=total > 0)
    np.log1p(out, out=out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--radius", type=float, default=15.0)
    args = parser.parse_args()

    data = ad.read_h5ad(args.data)
    loaded = load_embeddings(args.input_dir)
    saved = pd.Index(loaded["obs_names"].astype(str))
    pos = pd.Index(data.obs_names.astype(str)).get_indexer(saved)
    if (pos < 0).any():
        raise ValueError("Decoder rows do not align to the paired dataset")
    a = data[pos]
    n_bg = int(loaded["meta"]["n_bg_states"])
    n_sal = int(loaded["meta"]["n_sal_states"])
    bg = valid_composition(loaded["spatial"][:, :n_bg])
    sal = valid_composition(loaded["salient_spatial"][:, :n_sal])
    bg_out = np.zeros((a.n_obs, args.scale * n_bg), dtype=np.float32)
    sal_out = np.zeros((a.n_obs, args.scale * n_sal), dtype=np.float32)
    slices = a.obs["slice_id"].astype(str).to_numpy()
    coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
    for value in np.unique(slices):
        rows = np.flatnonzero(slices == value)
        print(f"[MENDER] {value}: {len(rows)} cells", flush=True)
        # Build the expensive spatial graph once. Split and normalize the two
        # composition slots independently so one slot cannot set the other's scale.
        joint = mender_soft_counts(
            coords[rows], np.concatenate([bg[rows], sal[rows]], axis=1),
            args.scale, args.radius,
        ).reshape(len(rows), args.scale, n_bg + n_sal)
        bg_counts = joint[:, :, :n_bg].reshape(len(rows), args.scale * n_bg)
        sal_counts = joint[:, :, n_bg:].reshape(len(rows), args.scale * n_sal)
        bg_out[rows] = normalize_log_counts(bg_counts)
        sal_out[rows] = normalize_log_counts(sal_counts)
    save_embeddings(
        args.output_dir,
        background=loaded["background"], salient=loaded["salient"],
        spatial=bg_out, salient_spatial=sal_out, obs_names=loaded["obs_names"],
        **{
            **loaded["meta"],
            "method": "haruka_decoder_composition_mender",
            "source_result": args.input_dir,
            "spatial_readout": "MENDER-style soft cell-state counts in non-overlapping rings",
            "mender_scale": args.scale,
            "mender_radius": args.radius,
            "decoder_block": "composition only",
            "composition_transform": "clip>=0 then row-normalize",
        },
    )


if __name__ == "__main__":
    main()
