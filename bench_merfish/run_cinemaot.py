"""CINEMA-OT baseline (conda env `vio`, which has cinemaot 0.0.5).

CINEMA-OT is the causal-matching answer to the same question Haruka asks: it
factors the data into a CONFOUNDER embedding (independent components whose
Chatterjee coefficient against the treatment label is below `thres`) and a
per-cell TREATMENT EFFECT obtained by subtracting each cell's entropically
regularised optimal-transport match in the other arm. Confounder <-> background,
treatment effect <-> salient. It has no spatial term at all, which is the point
of including it.

TWO CHOICES THAT MATTER, both made so the comparison is fair rather than
convenient:

* DIRECTION IS REVERSED. `cinemaot_unweighted(..., ref_label, expr_label)` returns
  `te` indexed by the cells of `ref_label`. Called the natural way (ref=control)
  the effect matrix would live on CONTROL cells, where the implanted domain does
  not exist, and CINEMA-OT would score zero for a reason that has nothing to do
  with its ability. So `ref_label="disease"`: each treated cell is matched to its
  control counterparts and the effect is that cell's residual, which is exactly
  the quantity the benchmark asks about.

* IT IS RUN PER TIMEPOINT. CINEMA-OT has no batch term and its transport map is
  dense (n_ref x n_expr). Run across all four timepoints at once it would both
  need a 36k x 36k map and be free to match a young control cell to an old
  treated one. Per-timepoint keeps the matching within an age-matched pair, which
  is how the method is meant to be used, and keeps each map at 9k x 9k.

Cells not covered (none, under this scheme) would be dropped; the scorer aligns
on `obs_names`, so a partial covering is handled rather than silently misaligned.

    python run_cinemaot.py --data data/sim_d0.6_s0.h5ad --out_root results
"""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Timer, save_embeddings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_root", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=20, help="number of independent components")
    ap.add_argument("--thres", type=float, default=0.15)
    args = ap.parse_args()

    import cinemaot as co

    tag = os.path.basename(args.data).replace("sim_", "").replace(".h5ad", "")
    out_dir = os.path.join(args.out_root, tag, f"cinemaot_seed{args.seed}")

    adata = ad.read_h5ad(args.data)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    np.random.seed(args.seed)
    obs_names, cf_all, te_all = [], [], []

    with Timer() as t:
        for tp in adata.obs["timepoint"].astype(str).unique():
            sub = adata[adata.obs["timepoint"].astype(str) == tp].copy()
            sub.obs["cond"] = sub.obs["condition"].astype(str)
            # CINEMA-OT reads `obsm['X_pca'][:, :dim]` for the ICA that produces the
            # confounder components. Computed within the timepoint, matching the
            # per-timepoint scope of the transport map.
            sc.pp.pca(sub, n_comps=max(args.dim, 50), svd_solver="arpack")
            print(f"[cinemaot] {tp}: {sub.n_obs} cells "
                  f"({(sub.obs['cond'] == 'disease').sum()} treated)", flush=True)

            cf, ot, TE = co.cinemaot.cinemaot_unweighted(
                sub, obs_label="cond", ref_label="disease", expr_label="control",
                dim=args.dim, thres=args.thres, smoothness=1e-4, eps=1e-3,
                mode="parametric",
            )
            # `cf` covers every cell of the pair, in `sub` order; `TE` covers the
            # ref (= treated) cells only.
            ref_mask = (sub.obs["cond"] == "disease").to_numpy()
            te = np.asarray(TE.X if not sp.issparse(TE.X) else TE.X.toarray())

            cf = np.asarray(cf)
            # Salient is defined only where an effect is defined; background is
            # padded to the same rows so both arrays share one index. Treated rows
            # get their own effect, control rows get zeros -- which is the same
            # convention Haruka uses (salient is zeroed on controls by design).
            te_full = np.zeros((sub.n_obs, te.shape[1]), dtype=np.float32)
            te_full[ref_mask] = te.astype(np.float32)

            obs_names.append(sub.obs_names.to_numpy())
            cf_all.append(cf.astype(np.float32))
            te_all.append(te_full)

    # CINEMA-OT keeps a VARIABLE number of confounder components per run -- it drops
    # any independent component whose Chatterjee coefficient against the treatment
    # label exceeds `thres`, and that count differs between timepoints (19 vs 20
    # here). Since each timepoint gets its own independent ICA, the components are
    # not comparable across timepoints anyway; zero-padding to a common width is
    # what lets them share one matrix, and the scorer standardises and re-PCAs
    # before clustering, so the padding columns carry no weight.
    width = max(c.shape[1] for c in cf_all)
    cf_all = [np.pad(c, ((0, 0), (0, width - c.shape[1]))) for c in cf_all]

    save_embeddings(
        out_dir,
        background=np.concatenate(cf_all, 0),
        salient=np.concatenate(te_all, 0),
        obs_names=np.concatenate(obs_names, 0),
        method="cinemaot", single_rep=False, runtime_s=t.seconds,
        dim=args.dim, thres=args.thres,
        note="cinemaot_unweighted per timepoint, ref=disease so the effect matrix "
             "is indexed by treated cells; confounder ICs = background, "
             "per-cell effect = salient",
    )


if __name__ == "__main__":
    main()
