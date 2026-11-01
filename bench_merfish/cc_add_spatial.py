"""Add CellCharter spatial slots to a result dir that only has intrinsic ones.

Exists because CINEMA-OT lives in env `vio` and CellCharter in env `cellcharter`,
and the two stacks cannot be installed together (cinemaot pins an older
scanpy/anndata than cellcharter 0.3.5 requires). So `run_cinemaot_2x2.py` writes
`background.npy` / `salient.npy` in `vio`, and this script -- run in
`cellcharter` -- reads them back and writes the two spatial slots.

The aggregation is byte-for-byte the one in `run_baselines_2x2._cc_aggregate`:
3-hop `aggregate_neighbors` on a Delaunay graph with long links removed, keeping
the hop blocks and dropping the own block. Importing it rather than copying it is
what guarantees CINEMA-OT's spatial slots are built the same way contrastiveVI's
are -- otherwise a difference in the aggregation would show up in the matrix as
if it were a difference between the methods.

    python cc_add_spatial.py --data data/simv3_d1_s0.h5ad \
        --result_dir results_v3/simv3_d1_s0/cinemaot_2x2_seed0
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np

from run_baselines_2x2 import _cc_aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--result_dir", required=True)
    args = ap.parse_args()

    adata = ad.read_h5ad(args.data)
    obs_names = np.load(os.path.join(args.result_dir, "obs_names.npy"), allow_pickle=True)
    bg = np.load(os.path.join(args.result_dir, "background.npy"))
    sal = np.load(os.path.join(args.result_dir, "salient.npy"))

    # The runner may emit cells in a different order than the h5ad (CINEMA-OT is
    # run per timepoint and the pieces are concatenated). Reindex the AnnData to
    # the saved order rather than assuming they agree -- a silent misalignment
    # here would look like a method that cannot find spatial structure.
    a = adata[np.asarray(obs_names, dtype=str)].copy()
    assert a.n_obs == len(bg) == len(sal), (a.n_obs, len(bg), len(sal))

    bg_sp = _cc_aggregate(a, bg, "co_bg")
    sal_sp = _cc_aggregate(a, sal, "co_sal")
    np.save(os.path.join(args.result_dir, "spatial.npy"), bg_sp)
    np.save(os.path.join(args.result_dir, "salient_spatial.npy"), sal_sp)

    meta_path = os.path.join(args.result_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    # n_layer MUST stay 0. It tells the scorer "these reps are the cellcharter
    # [own | hop1 | hop2 | hop3] concatenation, slice out the leading own block".
    # That is true of the Haruka runs, but NOT here: the intrinsic slots are raw
    # latents and `_cc_aggregate` already dropped the own block from the spatial
    # ones. Setting it to 3 made the scorer try to divide a 19-wide confounder
    # matrix into 4 blocks and drop CINEMA-OT from the table entirely.
    meta.update(n_layer=0, cc_spatial_added=True,
                note=str(meta.get("note", "")) + " | spatial slots = CellCharter "
                     "3-hop aggregation of each intrinsic latent (hop blocks only)")
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[cc_add_spatial] {args.result_dir}  spatial{bg_sp.shape} "
          f"salient_spatial{sal_sp.shape}")


if __name__ == "__main__":
    main()
