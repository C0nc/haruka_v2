"""Is the salient_spatial quadrant actually hard, or free to anyone who pools?

`sal_domain` sets P(sal_state=1). Pooling k neighbours estimates that rate, so
the quadrant's difficulty is set entirely by how far apart the rates are
relative to the binomial noise of the pool -- NOT by how good a method is.

With the original rates (0.1, 0.5, 0.9) the gap is 0.4 against a pooled SD of
about sqrt(0.25/40) = 0.08, i.e. ~5 sigma, and every method scored 0.873-0.990
there regardless of its per-cell salient signal. contrastiveVI went from 0.293
per-cell to 0.873 pooled: pooling launders a weak signal into a strong one.

This computes the POOLED ORACLE CEILING: take the TRUE per-cell salient state,
one-hot it, apply the same CellCharter 3-hop aggregation the methods get, and
ask how well that recovers the domain. That is the best any method could do, so
it says what the quadrant is worth as a test.

    ceiling ~1.0  quadrant is free, discriminates nothing
    ceiling ~0.6-0.8  quadrant rewards a genuinely good salient representation

Also reports the per-cell oracle, so the gap between them is the pure pooling
amplification.

    python validate_salsp.py --data data/simv4_g0.8_r0.3_d1_s0.h5ad
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import dependence  # noqa: E402
from run_baselines_2x2 import _cc_aggregate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    adata = ad.read_h5ad(args.data)
    rates = adata.uns.get("sim2x2", {}).get("sal_rates", "?")
    treated = (adata.obs["condition"].astype(str) == "disease").to_numpy()

    state = adata.obs["gt_sal_intrinsic"].astype(str).to_numpy()
    onehot = pd.get_dummies(state).to_numpy(dtype=np.float32)

    a = adata.copy()
    pooled = _cc_aggregate(a, onehot, "oracle_state")

    dom = adata.obs["gt_sal_domain"].astype(str).to_numpy()[treated]
    grp = adata.obs["slice_id"].astype(str).to_numpy()[treated]

    per_cell = dependence(onehot[treated], dom, grp, discrete=True, seed=0)
    pooled_d = dependence(pooled[treated], dom, grp, discrete=True, seed=0)

    print(f"data          {os.path.basename(args.data)}")
    print(f"sal_rates     {rates}")
    print(f"\nORACLE per-cell state  -> sal_domain : {per_cell:.3f}")
    print(f"ORACLE pooled (3-hop)  -> sal_domain : {pooled_d:.3f}   <-- quadrant ceiling")
    print(f"pooling amplification                : {pooled_d - per_cell:+.3f}")
    print("\nceiling ~1.0 => quadrant is free and discriminates nothing;"
          "\nceiling 0.6-0.8 => it rewards a genuinely good salient representation")


if __name__ == "__main__":
    main()
