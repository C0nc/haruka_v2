"""Validate the synthetic-counts build.

The assertions here are NOT the same as `self_check_2x2.py`'s, and the difference
is the point. On real tissue `gt_niche` was constructed to be invisible in a cell's
own row (`haruka_sim.py`'s rho = 0). Here the counts come from the model's own
generative process, whose decoder reads `z_sp` -- so the niche IS in the cell's own
expression, by construction and correctly. Asserting otherwise would be asserting
that the generative process is not the generative process.

What must hold instead:

  A. EVERY factor is recoverable from expression. If not, delta is below the noise
     floor and any method scoring zero is being blamed for the simulator.
  B. `gt_sal_intrinsic` is NOT recoverable from the NEIGHBOURHOOD mean. It is
     assigned i.i.d. in space, so pooling must destroy it -- this is the factor
     that stops "smooth everything" from winning, and it only does that job if
     neighbour-recovery sits at chance.
  C. `gt_niche` IS recoverable from the neighbourhood mean, and more so than
     `gt_sal_intrinsic` is. It is a neighbourhood quantity; pooling should help.
  D. The two salient factors are ABSENT on control cells. Their counterfactual
     labels must be unpredictable there.
  E. The four factors are MUTUALLY INDEPENDENT (pairwise NMI ~ 0). This is what
     makes an off-diagonal entry in the scoring matrix attributable to the model
     mixing factors rather than to the data confounding them.

    python self_check_counts.py --data data/simcnt_d1_s0.h5ad
"""

from __future__ import annotations

import argparse
import itertools
import json

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.metrics import normalized_mutual_info_score

from common import probe_auroc_multiclass

TOL_NULL = 0.58
TOL_SIGNAL = 0.62
TOL_NMI = 0.02

FACTORS = ["gt_arch", "gt_niche", "gt_sal_intrinsic", "gt_sal_domain"]
TREATED_ONLY = {"gt_sal_intrinsic", "gt_sal_domain"}


def _lognorm(adata):
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1.0) * 1e4)


def neighbour_mean(X, xy, slice_id, k=30):
    out = np.zeros_like(X)
    for s in np.unique(slice_id):
        m = np.where(slice_id == s)[0]
        _, idx = cKDTree(xy[m]).query(xy[m], k=min(k, len(m) - 1) + 1)
        out[m] = X[m][idx[:, 1:]].mean(1)
    return out


def check(path: str, k: int = 30, signal_delta: float = 1.0) -> dict:
    adata = ad.read_h5ad(path)
    delta = float(adata.uns["sim2x2"]["delta"])
    obs = adata.obs
    treated = (obs["condition"].astype(str) == "disease").to_numpy()
    sl = obs["slice_id"].astype(str).to_numpy()
    xy = np.asarray(adata.obsm["spatial"], dtype=float)

    X = _lognorm(adata)
    N = neighbour_mean(X, xy, sl, k=k)

    res = dict(path=path, delta=delta)
    fails = []

    for f in FACTORS:
        y = obs[f].astype(str).to_numpy()
        m = treated if f in TREATED_ONLY else np.ones(len(y), bool)
        res[f"{f}__own"] = probe_auroc_multiclass(X[m], y[m], sl[m])
        res[f"{f}__nbr"] = probe_auroc_multiclass(N[m], y[m], sl[m])
        if f in TREATED_ONLY:
            res[f"{f}__control"] = probe_auroc_multiclass(
                X[~treated], y[~treated], sl[~treated])

    # --- E: factors mutually independent -----------------------------------
    res["pairwise_nmi"] = {}
    for a, b in itertools.combinations(FACTORS, 2):
        v = float(normalized_mutual_info_score(obs[a].astype(str), obs[b].astype(str)))
        res["pairwise_nmi"][f"{a}|{b}"] = v
        if v > TOL_NMI:
            fails.append(f"E: {a} and {b} are not independent (NMI {v:.3f} > {TOL_NMI}); "
                         f"an off-diagonal score would be the DATA confounding them, "
                         f"not the model mixing them")

    if delta == 0:
        # Nothing but architecture is switched on at delta 0.
        for f in ("gt_niche", "gt_sal_intrinsic", "gt_sal_domain"):
            if res[f"{f}__own"] >= TOL_NULL:
                fails.append(f"null arm: {f} own={res[f'{f}__own']:.3f} is not null")
        res["fails"] = fails
        res["ok"] = not fails
        return res

    # --- A: every factor recoverable from expression -----------------------
    for f in FACTORS:
        v = res[f"{f}__own"]
        if v <= TOL_SIGNAL:
            if delta >= signal_delta:
                fails.append(f"A: {f} own={v:.3f} not recoverable at delta={delta}")
            else:
                res.setdefault("below_floor", []).append(f)

    # --- B: the i.i.d. factor must die under pooling ------------------------
    if res["gt_sal_intrinsic__nbr"] >= TOL_NULL:
        fails.append(f"B: gt_sal_intrinsic nbr={res['gt_sal_intrinsic__nbr']:.3f} "
                     f"survives neighbourhood pooling; it is meant to be i.i.d. in "
                     f"space and is the only factor smoothing cannot help with")

    # --- C: the niche must benefit from pooling more than the i.i.d. factor -
    if res["gt_niche__nbr"] <= res["gt_sal_intrinsic__nbr"]:
        fails.append(f"C: gt_niche nbr={res['gt_niche__nbr']:.3f} is not above "
                     f"gt_sal_intrinsic nbr={res['gt_sal_intrinsic__nbr']:.3f}; "
                     f"the niche is supposed to be a neighbourhood quantity")

    # --- D: salient factors absent on controls ------------------------------
    for f in TREATED_ONLY:
        v = res[f"{f}__control"]
        if v >= TOL_NULL:
            fails.append(f"D: {f} is predictable on CONTROL cells (AUROC {v:.3f}); "
                         f"the factor was supposed to be switched off there")

    res["fails"] = fails
    res["ok"] = not fails
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--signal_delta", type=float, default=1.0)
    args = ap.parse_args()
    bad = 0
    for p in args.data:
        r = check(p, k=args.k, signal_delta=args.signal_delta)
        print(json.dumps(r, indent=2))
        if not r["ok"]:
            bad += 1
            for f in r["fails"]:
                print(f"  !! {f}")
    print(f"\n[self_check_counts] {len(args.data) - bad}/{len(args.data)} builds passed")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
