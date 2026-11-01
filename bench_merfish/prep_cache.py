"""One-time extraction of the benchmark's source slices from the full atlas.

`process_all_brain.h5ad` stores a DENSE float64 matrix of 2.47M x 300. Backed-mode
fancy indexing over ~70k scattered rows of that is pathologically slow (h5py reads
a chunk per row), so this reads the matrix once, keeps only the mice named in
`build_sim.PAIRS`, recovers counts, and writes a ~900 MB cache that every
subsequent (delta, seed) build reads in seconds.

Counts are recovered as ``expm1(X) * transcript_count / 1e4``: the atlas keeps
only the log-normalised matrix, but it also keeps the library size that
normalisation divided by, so the inverse is exact -- recovered row sums match
`transcript_count` to machine precision, which is asserted here rather than
assumed.

    python prep_cache.py --out data/atlas_pairs.h5ad
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_sim import ATLAS, PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=ATLAS)
    ap.add_argument("--out", default="data/atlas_pairs.h5ad")
    args = ap.parse_args()

    mice = sorted({m for _, cm, _, dm, _ in PAIRS for m in (cm, dm)})
    print(f"[prep] keeping {len(mice)} mice: {mice}")

    handle = ad.read_h5ad(args.atlas, backed="r")
    obs = handle.obs.copy()
    var = handle.var.copy()
    var_names = handle.var_names.copy()
    handle.file.close()

    keep = ((obs["cohort"].values == "aging_coronal")
            & obs["mouse_id"].astype(str).isin(mice).values)
    rows = np.where(keep)[0]
    print(f"[prep] {len(rows)} cells; reading X by contiguous blocks")

    # Contiguous-block reads rather than one fancy-index call: the selected mice
    # occupy a handful of runs in the file, so this is ~2 orders of magnitude
    # faster than h5py's per-row path and uses the same memory.
    with h5py.File(args.atlas, "r") as fh:
        dset = fh["X"]
        chunks = []
        start = 0
        block = 200_000
        while start < dset.shape[0]:
            stop = min(start + block, dset.shape[0])
            sel = rows[(rows >= start) & (rows < stop)]
            if len(sel):
                chunks.append(np.asarray(dset[start:stop])[sel - start])
            start = stop
    X = np.concatenate(chunks, axis=0).astype(np.float64)
    del chunks

    sub_obs = obs.iloc[rows].copy()
    tc = sub_obs["transcript_count"].to_numpy(dtype=np.float64)[:, None]
    counts = np.clip(np.expm1(X) * (tc / 1e4), 0.0, None)
    rel = np.abs(counts.sum(1) - tc.ravel()) / np.maximum(tc.ravel(), 1.0)
    assert rel.max() < 1e-6, f"count recovery is not exact: max rel err {rel.max():.2e}"
    print(f"[prep] count recovery ok (max rel err {rel.max():.2e}); "
          f"mean library {counts.sum(1).mean():.1f}")

    out = ad.AnnData(
        X=sp.csr_matrix(counts.astype(np.float32)),
        obs=sub_obs.reset_index(drop=False).set_index("index"),
        var=var,
    )
    out.var_names = var_names
    with h5py.File(args.atlas, "r") as fh:
        out.obsm["spatial"] = np.asarray(fh["obsm/spatial"][:], dtype=np.float32)[rows]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(f"[prep] wrote {args.out}  {out.shape}")
    print(pd.crosstab(out.obs["mouse_id"].astype(str), out.obs["subregion"].astype(str)))


if __name__ == "__main__":
    main()
