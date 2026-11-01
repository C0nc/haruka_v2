"""Sweep the gate strength and check ALL FOUR factors survive it.

Two separate questions, both of which have burned this benchmark before:

  1. Does a stronger gate buy more non-linearity? Tracked by the gap between a
     linear and a non-linear probe for the salient state, and by the cosine
     between per-cell-type effect directions. A bigger gap means more of the
     signal is only reachable by conditioning on cell type.

  2. Does the gate WRECK the other three quadrants? The gate multiplies h_bg, so
     pushing it too hard drowns the background factors in salient modulation --
     the v1 failure mode, where a factor was left with no recoverable signal and
     every method scored ~0 for a reason that had nothing to do with method
     quality. Each factor therefore gets a non-linear probe from raw expression,
     which upper-bounds what ANY method can achieve. All four must stay high.

Prints one row per gate strength; pick the largest gap subject to all four
factors staying recoverable.

    python validate_v4.py --gates 0.8 1.2 1.6
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import itertools  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from build_sim_v4 import build  # noqa: E402

FACTORS = ["gt_arch", "gt_niche", "gt_sal_intrinsic", "gt_sal_domain"]
TREATED_ONLY = {"gt_sal_intrinsic", "gt_sal_domain"}


def ovr_auroc(X, y, groups, model, seed=0):
    """One-vs-rest macro AUROC, grouped by slice so nothing leaks across folds."""
    classes = np.unique(y)
    scores = []
    for c in classes:
        yy = (y == c).astype(int)
        if yy.sum() < 50 or (1 - yy).sum() < 50:
            continue
        pred = np.zeros(len(yy), dtype=float)
        for tr, te in GroupKFold(n_splits=3).split(X, yy, groups):
            s = StandardScaler().fit(X[tr])
            m = model()
            m.fit(s.transform(X[tr]), yy[tr])
            pred[te] = m.predict_proba(s.transform(X[te]))[:, 1]
        scores.append(roc_auc_score(yy, pred))
    return float(np.mean(scores)) if scores else float("nan")


def assess(path, n_sub, seed=0):
    adata = ad.read_h5ad(path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    treated = (adata.obs["condition"].astype(str) == "disease").to_numpy()

    out = {}
    rng = np.random.default_rng(seed)
    for f in FACTORS:
        a = adata[treated] if f in TREATED_ONLY else adata
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        y = a.obs[f].astype(str).to_numpy()
        g = a.obs["slice_id"].astype(str).to_numpy()
        idx = rng.choice(len(X), min(n_sub, len(X)), replace=False)
        out[f] = ovr_auroc(X[idx], y[idx], g[idx],
                           lambda: HistGradientBoostingClassifier(max_iter=150))

    # salient state: linear vs non-linear, and cross-cell-type effect geometry
    a = adata[treated]
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    st = a.obs["gt_sal_intrinsic"].astype(str).to_numpy().astype(int)
    ct = a.obs["gt_arch"].astype(str).to_numpy()
    g = a.obs["slice_id"].astype(str).to_numpy()
    idx = rng.choice(len(X), min(n_sub, len(X)), replace=False)
    out["lin"] = ovr_auroc(X[idx], st[idx], g[idx],
                           lambda: LogisticRegression(max_iter=2000))
    out["nonlin"] = out["gt_sal_intrinsic"]

    dirs = {}
    for c in np.unique(ct):
        m = ct == c
        if (st[m] == 1).sum() < 30 or (st[m] == 0).sum() < 30:
            continue
        dirs[c] = X[m & (st == 1)].mean(0) - X[m & (st == 0)].mean(0)
    cos = [float(dirs[a_] @ dirs[b_] /
                 (np.linalg.norm(dirs[a_]) * np.linalg.norm(dirs[b_]) + 1e-12))
           for a_, b_ in itertools.combinations(sorted(dirs), 2)]
    out["cos"] = float(np.mean(cos))
    out["cos_min"] = float(np.min(cos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", type=float, nargs="+", default=[0.8, 1.2, 1.6])
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_side", type=int, default=95)
    ap.add_argument("--n_genes", type=int, default=300)
    ap.add_argument("--n_sub", type=int, default=15000)
    args = ap.parse_args()

    print(f"{'gate':>5} {'ctype':>7} {'bgDOM':>7} {'state':>7} {'salDOM':>7} "
          f"{'lin':>7} {'gap':>7} {'cos':>7} {'cosmin':>7}")
    for gth in args.gates:
        path = f"data/simv4_g{gth:g}_d{args.delta:g}_s{args.seed}.h5ad"
        if not os.path.exists(path):
            built = build(args.delta, args.seed, args.n_side, args.n_genes,
                          "data", gate_strength=gth)
            os.rename(built, path)
        r = assess(path, args.n_sub, args.seed)
        print(f"{gth:5.1f} {r['gt_arch']:7.3f} {r['gt_niche']:7.3f} "
              f"{r['gt_sal_intrinsic']:7.3f} {r['gt_sal_domain']:7.3f} "
              f"{r['lin']:7.3f} {r['nonlin'] - r['lin']:7.3f} "
              f"{r['cos']:7.3f} {r['cos_min']:7.3f}")
    print("\nall four factor columns must stay high (a low one = unreachable "
          "factor, the v1 failure); pick the largest `gap`")


if __name__ == "__main__":
    main()
