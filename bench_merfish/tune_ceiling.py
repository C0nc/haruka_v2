"""Find a simulator configuration whose spatial factors are actually learnable.

WHY. `diag_representable.py` showed the GAT already extracts essentially all the
information the neighbourhood contains (0.485 vs an unconstrained MLP's 0.484 on
the salient pair; 0.268 vs 0.341 on the background pair). The operator is fine.
What is wrong is the CEILING: even a perfect reader recovers only R^2 0.34-0.48 of
the true spatial latent, so the two spatial quadrants cannot discriminate between
model variants and every `both` vs `background` difference measured there has been
compressed into noise.

WHAT DOES *NOT* RAISE IT. Scaling the codebook (`A_niche`, `A_sdom`). The ceiling
is how well the LABEL is determined by the neighbourhood; scaling multiplies both
sides of the regression equally and leaves R^2 unchanged. This was my first
instinct and it is wrong.

WHAT DOES. The label is terciles of a residualised neighbourhood composition, so
determinacy is set by how much that composition VARIES and how precisely it is
estimated:

    arch_ls   spatial granularity of the parent field. Coarse regions mean a
              cell's neighbourhood is almost all its own class, so after
              residualising against own class almost nothing is left. Finer
              regions mix neighbourhoods and put real variance in the residual.
    k         neighbours pooled. More neighbours estimate the composition more
              precisely, at the cost of a wider effective radius.
    n_out     number of label classes. Terciles throw away more of a continuous
              score than a binary split does.

This sweeps those three and reports the ceiling for each, WITHOUT building h5ads
or training anything -- labels and codebooks only, so a full sweep is seconds.

    python tune_ceiling.py
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor

import build_sim_counts as B


def ceiling(arch_ls: float, k: int, n_out: int, n_side: int = 95, seed: int = 0,
            n_parent: int = 7, d_child: int = 6) -> tuple[float, float]:
    """Ceiling R^2 for predicting the child latent from the neighbourhood, and the
    fraction of cells whose neighbourhood is NOT dominated by their own class."""
    rng = np.random.default_rng(1234 + seed)
    xy = B._lattice(n_side, rng)

    # parent field at the requested granularity
    grid = 128
    fields = [np.asarray(_smooth(rng, grid, arch_ls)) for _ in range(n_parent)]
    u = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
    ij = np.clip((u * (grid - 1)).round().astype(int), 0, grid - 1)
    parent = np.stack([f[ij[:, 1], ij[:, 0]] for f in fields], 1).argmax(1)

    proj = rng.normal(0, 1, n_parent)
    _, child_lab = B._nbr_factor(xy, parent, n_parent, proj, k=k, n_out=n_out)

    A_child = rng.normal(0, 1.4, (n_out, d_child))
    child = A_child[child_lab]

    # information ceiling: unconstrained non-linear read of the neighbourhood
    kk = min(k, len(xy) - 1)
    _, idx = cKDTree(xy).query(xy, k=kk + 1)
    idx = idx[:, 1:]
    onehot = np.eye(n_parent)[parent]
    X = np.hstack([onehot[idx].mean(1), onehot])
    n = len(xy)
    tr = np.zeros(n, bool); tr[: n // 2] = True
    m = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300, random_state=0)
    m.fit(X[tr], child[tr])
    p = m.predict(X[~tr]); y = child[~tr]
    r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

    # how mixed are neighbourhoods? fraction of cells whose own class is NOT the
    # neighbourhood plurality -- the residual only has variance where this is true
    plurality = onehot[idx].mean(1).argmax(1)
    mixed = float((plurality != parent).mean())
    return r2, mixed


def _smooth(rng, grid, ls):
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal((grid, grid)), sigma=ls * grid, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.75)
    args = ap.parse_args()

    print(f"{'arch_ls':>9}{'k':>6}{'n_out':>7}{'ceiling R2':>13}{'frac mixed':>12}")
    best = None
    for arch_ls, k, n_out in itertools.product((0.12, 0.06, 0.03), (60, 120), (3, 2)):
        r2, mixed = ceiling(arch_ls, k, n_out)
        flag = "  <-- meets target" if r2 >= args.target else ""
        print(f"{arch_ls:>9.3f}{k:>6}{n_out:>7}{r2:>13.3f}{mixed:>12.3f}{flag}")
        if best is None or r2 > best[0]:
            best = (r2, arch_ls, k, n_out, mixed)
    print(f"\nbest: R2={best[0]:.3f} at arch_ls={best[1]}, k={best[2]}, "
          f"n_out={best[3]} (frac mixed {best[4]:.3f})")
    print("current build uses arch_ls=0.12, k=60, n_out=3")


if __name__ == "__main__":
    main()
