"""Measure and store the attainable ceiling of each 2x2 quadrant.

Writes `uns['sim2x2']['quadrant_ceilings']` into the dataset so the scorer can
report `diag_frac_ceiling` -- each quadrant as a fraction of what is actually
achievable rather than of 1.0.

WHY THIS IS NOT COSMETIC. The v5 process gives the four quadrants very different
attainable maxima (~1.00 cell type, ~1.00 background domain, 0.775 salient state,
0.640 salient domain). Averaging raw scores across them rewards a method for the
easy quadrants and penalises one that does well on the hard ones. Every headline
number reported so far used the raw mean, so a method strong on the subtle
salient factors was scored below its true standing.

CEILING = what a strong non-parametric probe gets FROM RAW EXPRESSION, with the
same neighbourhood pooling the methods receive:

  * intrinsic factors (cell type, salient state) -- probe on raw per-cell expression
  * spatial factors (both domains) -- probe on the CellCharter 3-hop aggregation of
    raw expression, because these are neighbourhood properties and no method is
    expected to read them off one cell

This is an upper bound on any method that starts from the same counts, so a score
above it means the ceiling is measured wrong -- which is exactly how the two-route
`sal_domain` bug was found (CINEMA-OT scored 0.990 against a 0.957 estimate that
only accounted for the state-pooling route and ignored the direct s_sp term).

    python measure_ceilings.py --data data/simv4_g0.8_r0.3_p0.15_d1_s0.h5ad
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from common import dependence, embed  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from run_baselines_2x2 import _cc_aggregate  # noqa: E402


def nonlinear_dependence(X, y, groups, seed=0):
    """Same scale as `dependence`, but with a NON-LINEAR probe.

    `dependence` uses LogisticRegression, so its ceiling is what a LINEAR readout
    can extract. That is the wrong denominator for a factor deliberately built to
    be invisible to linear readouts: v4's gate uses per-cell-type weights centred
    to zero mean across types (effect-direction cosine -0.035), so the pooled
    linear signal cancels by construction. Measured on data_easy, the linear and
    non-linear ceilings agree to within 0.02 on three of the four factors and
    differ by 0.272 on gt_sal_intrinsic alone -- exactly the engineered one.

    Reporting both matters in two directions: the non-linear value is the real
    information ceiling and the honest denominator for `diag_frac_ceiling`, while
    a method scoring ABOVE the linear value has itself done non-linear work that
    raw expression does not hand a linear probe for free.
    """
    Z = embed(X, n_pcs=50, seed=seed)
    y = np.asarray(y)
    out = []
    for c in np.unique(y):
        yy = (y == c).astype(int)
        if yy.sum() < 50 or (1 - yy).sum() < 50:
            continue
        pred = np.zeros(len(yy), dtype=float)
        for tr, te in GroupKFold(n_splits=min(4, len(np.unique(groups)))).split(Z, yy, groups):
            sc_ = StandardScaler().fit(Z[tr])
            m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
            m.fit(sc_.transform(Z[tr]), yy[tr])
            pred[te] = m.predict_proba(sc_.transform(Z[te]))[:, 1]
        out.append(roc_auc_score(yy, pred))
    return max(0.0, 2.0 * (float(np.mean(out)) - 0.5)) if out else float("nan")

#: factor -> whether it is a neighbourhood property (needs pooling to be readable)
SPATIAL = {"gt_arch": False, "gt_sal_intrinsic": False,
           "gt_niche": True, "gt_sal_domain": True}
TREATED_ONLY = {"gt_sal_intrinsic", "gt_sal_domain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--write", action="store_true",
                    help="store the ceilings back into the h5ad")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.data)
    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    X = work.X.toarray() if sp.issparse(work.X) else np.asarray(work.X)
    pooled = _cc_aggregate(work.copy(), X, "ceil_expr")

    treated = (adata.obs["condition"].astype(str) == "disease").to_numpy()
    slide = adata.obs["slice_id"].astype(str).to_numpy()

    ceilings, nl = {}, {}
    for factor, needs_pool in SPATIAL.items():
        src = pooled if needs_pool else X
        m = treated if factor in TREATED_ONLY else np.ones(len(X), bool)
        y = adata.obs[factor].astype(str).to_numpy()[m]
        ceilings[factor] = float(dependence(src[m], y, slide[m], discrete=True, seed=0))
        nl[factor] = float(nonlinear_dependence(src[m], y, slide[m], seed=0))
        print(f"{factor:20s} linear {ceilings[factor]:.3f}   nonlinear {nl[factor]:.3f}"
              f"   gap {nl[factor]-ceilings[factor]:+.3f}"
              f"   ({'pooled' if needs_pool else 'per-cell'} raw expression)")

    if args.write:
        u = dict(adata.uns.get("sim2x2", {}) or {})
        # `quadrant_ceilings` is the NON-LINEAR (true information) ceiling, the
        # honest denominator for diag_frac_ceiling. The linear one is kept beside
        # it because exceeding it is a meaningful result in its own right.
        u["quadrant_ceilings"] = nl
        u["quadrant_ceilings_linear"] = ceilings
        adata.uns["sim2x2"] = u
        adata.write_h5ad(args.data, compression="gzip")
        print(f"\nwrote quadrant_ceilings into {args.data}")


if __name__ == "__main__":
    main()
