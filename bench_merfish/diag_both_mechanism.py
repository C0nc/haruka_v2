"""Why does `spatial_attach="both"` put anatomy in the SALIENT latents?

HYPOTHESIS. Under `both`, `s_sp` is attention-pooled from the NEIGHBOURS' salient
encoder outputs. A neighbourhood average of anything spatially smooth is a very
good reconstruction feature, and anatomy is the smoothest, highest-variance
structure in the data. So if `s` starts encoding anatomy, `s_sp` immediately
becomes a powerful smoothed-anatomy predictor, reconstruction loss drops, and the
gradient reinforces `s` encoding anatomy. A positive feedback loop.

Nothing in the loss opposes it:

  * `_independence_penalty` constrains ONLY `z _||_ z_sp`. The salient arm is
    behind `penalize_salient_spatial`, which is OFF by default and deliberately
    so (the docstring argues that forcing `s _||_ z_sp` deletes the lesion).
  * The Wasserstein term is `||qs_m[background]||^2` -- it drives `s` to zero on
    CONTROL cells only. On treated cells `s` is free to encode whatever lowers
    reconstruction loss, including anatomy.

Under `spatial_attach="background"` the same smoothing pathway exists but it hangs
off `z`, so the benefit accrues to the background latent, which is where anatomy
belongs. That asymmetry is the proposed mechanism.

FOUR PREDICTIONS, each falsifiable on saved embeddings:

  P1  `s` on TREATED cells encodes anatomy under `both` but not under `background`
      or `vanilla`.
  P2  `s` on CONTROL cells is near zero for all variants -- i.e. the Wasserstein
      term IS working, and the leak is not a failure of the control-arm zeroing
      but of the treated arm being unconstrained.
  P3  `s_sp` is largely a neighbourhood-smoothed copy of `s` (high R^2 from the
      neighbour mean of `s`), which is what makes it a cheap reconstruction win.
  P4  Under `background`, `z_sp` is the smoothed copy instead, and anatomy sits in
      `z`.

If P1-P4 all hold, the fix is to constrain the salient pair -- or to stop the
salient branch from having its own pooling path -- not to touch the loss terms
SIMVI lacks (which the ablations already showed do not fix it).

    python diag_both_mechanism.py --data data/simcnt_d1_s0.h5ad --results results_counts
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import load_embeddings

# latent widths (SHARED in run_haruka.py): latent_dim=32, n_spatial_latent=16.
# `cc.gr.aggregate_neighbors(n_layers=3)` concatenates [own | hop1 | hop2 | hop3],
# so the cell's OWN latent is the leading block of each saved representation.
D_INTRINSIC, D_SPATIAL = 32, 16


def own_block(arr, d):
    return np.asarray(arr[:, :d], dtype=np.float64)


def nbr_mean(X, xy, sl, k=15):
    out = np.zeros_like(X)
    for s in np.unique(sl):
        m = np.where(sl == s)[0]
        _, idx = cKDTree(xy[m]).query(xy[m], k=min(k, len(m) - 1) + 1)
        out[m] = X[m][idx[:, 1:]].mean(1)
    return out


def r2(X, y):
    """Ridge R^2, features standardised. Multi-output y is averaged."""
    X = StandardScaler().fit_transform(X)
    return float(RidgeCV(alphas=np.logspace(-2, 3, 8)).fit(X, y).score(X, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/simcnt_d1_s0.h5ad")
    ap.add_argument("--results", default="results_counts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+",
                    default=["haruka_vanilla", "haruka_background", "haruka_both",
                             "haruka_both_simviloss"])
    args = ap.parse_args()

    adata = ad.read_h5ad(args.data)
    obs = adata.obs
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    sl = obs["slice_id"].astype(str).to_numpy()
    arch = OneHotEncoder(sparse_output=False).fit_transform(
        obs["gt_arch"].astype(str).to_numpy().reshape(-1, 1))

    tag = os.path.basename(args.data).replace("sim_", "").replace(".h5ad", "")
    print(f"{'variant':<24}{'|s| ctrl':>10}{'|s| treat':>11}"
          f"{'R2(s->arch)':>13}{'R2(z->arch)':>13}"
          f"{'R2(nbrmean s -> s_sp)':>23}{'R2(nbrmean z -> z_sp)':>23}")
    for v in args.variants:
        d = os.path.join(args.results, tag, f"{v}_seed{args.seed}")
        if not os.path.exists(os.path.join(d, "meta.json")):
            print(f"{v:<24}  (missing)")
            continue
        e = load_embeddings(d)
        z = own_block(e["background"], D_INTRINSIC)
        s = own_block(e["salient"], D_INTRINSIC)

        # P2: is the Wasserstein zeroing working on controls?
        ns_ctrl = float(np.linalg.norm(s[~treated], axis=1).mean())
        ns_treat = float(np.linalg.norm(s[treated], axis=1).mean())

        # P1 / P4: which latent carries anatomy, on the TREATED arm
        r2_s_arch = r2(s[treated], arch[treated])
        r2_z_arch = r2(z[treated], arch[treated])

        # P3: is the spatial block a smoothed copy of its own intrinsic partner?
        r2_ssp = float("nan")
        if "salient_spatial" in e:
            s_sp = own_block(e["salient_spatial"], D_SPATIAL)
            r2_ssp = r2(nbr_mean(s, xy, sl)[treated], s_sp[treated])
        r2_zsp = float("nan")
        if "spatial" in e:
            z_sp = own_block(e["spatial"], D_SPATIAL)
            r2_zsp = r2(nbr_mean(z, xy, sl)[treated], z_sp[treated])

        print(f"{v:<24}{ns_ctrl:>10.3f}{ns_treat:>11.3f}"
              f"{r2_s_arch:>13.3f}{r2_z_arch:>13.3f}{r2_ssp:>23.3f}{r2_zsp:>23.3f}")

    print("\nP1 s->arch high only for `both`      P2 |s| ctrl ~0 everywhere")
    print("P3 nbrmean(s)->s_sp high for `both`  P4 z->arch high for `background`")


if __name__ == "__main__":
    main()
