"""Is the salient effect the SAME direction in every cell type?

This decides whether the benchmark is testing anything non-trivial about the
salient axis. Two things are measured on the treated arm:

  1. EFFECT-DIRECTION AGREEMENT. Within each cell type, take
     mean(expr | state=1) - mean(expr | state=0) -- that type's treatment-effect
     direction -- and compute pairwise cosine similarity across types.

     ~1.0  the perturbation is one shared vector. Additive and confounder-free,
           which is exactly the model CINEMA-OT assumes, so a high CINEMA-OT
           score says nothing about harder data.
     ~0.0  the response is cell-type-specific, and recovering the state requires
           conditioning on cell type.

  2. HOW EASY IS THE STATE TO READ OFF RAW EXPRESSION. A linear probe and a
     non-linear one (gradient boosting), both grouped by slice. If the linear
     probe already scores ~1.0 then "recovering the salient state" is not
     evidence of disentanglement -- every method that keeps expression around
     gets it for free, which is the SIMVI/PCA result.

    python diag_salient_geometry.py --data data/simv3_d1_s0.h5ad
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


def grouped_auroc(X, y, groups, model):
    preds = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=4).split(X, y, groups):
        sc_ = StandardScaler().fit(X[tr])
        m = model()
        m.fit(sc_.transform(X[tr]), y[tr])
        preds[te] = m.predict_proba(sc_.transform(X[te]))[:, 1]
    return roc_auc_score(y, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_sub", type=int, default=20000)
    args = ap.parse_args()

    adata = ad.read_h5ad(args.data)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    treated = (adata.obs["condition"].astype(str) == "disease").to_numpy()
    a = adata[treated]
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    state = a.obs["gt_sal_intrinsic"].astype(str).to_numpy().astype(int)
    ctype = a.obs["gt_arch"].astype(str).to_numpy()
    slide = a.obs["slice_id"].astype(str).to_numpy()

    print(f"treated cells: {len(X)}  state=1 fraction: {state.mean():.3f}")

    # --- 1. per-cell-type effect directions --------------------------------
    dirs = {}
    for c in np.unique(ctype):
        m = ctype == c
        if (state[m] == 1).sum() < 30 or (state[m] == 0).sum() < 30:
            continue
        dirs[c] = X[m & (state == 1)].mean(0) - X[m & (state == 0)].mean(0)

    print(f"\ncell types with both states: {len(dirs)}")
    cos = []
    for c1, c2 in itertools.combinations(sorted(dirs), 2):
        v1, v2 = dirs[c1], dirs[c2]
        cos.append(float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)))
    cos = np.asarray(cos)
    print(f"pairwise cosine of per-cell-type effect directions:")
    print(f"    mean {cos.mean():.3f}   min {cos.min():.3f}   max {cos.max():.3f}")
    print("    -> ~1.0 means ONE shared additive perturbation (CINEMA-OT's "
          "assumption holds exactly)")

    # --- 2. decodability of the state from raw expression -------------------
    idx = np.random.default_rng(0).choice(len(X), min(args.n_sub, len(X)), replace=False)
    Xs, ys, gs = X[idx], state[idx], slide[idx]
    lin = grouped_auroc(Xs, ys, gs, lambda: LogisticRegression(max_iter=2000, C=1.0))
    nl = grouped_auroc(Xs, ys, gs, lambda: HistGradientBoostingClassifier(max_iter=200))
    print(f"\nsalient state from RAW expression (grouped by slice, n={len(idx)}):")
    print(f"    linear probe AUROC     {lin:.3f}")
    print(f"    non-linear probe AUROC {nl:.3f}")
    print("    -> if the linear probe is already ~1.0, any latent that retains "
          "expression recovers the state for free; it is not evidence of "
          "disentanglement")


if __name__ == "__main__":
    main()
