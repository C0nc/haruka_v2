"""Bounded-memory mini-batch approximation to CINEMA-OT.

PCA, ICA, and confounder-component selection are fitted globally exactly once.
The dense treated-by-control transport is then replaced by independent balanced
Sinkhorn problems on randomized arm-matched mini-batches. Repeating the random
partition and averaging treatment effects reduces partition noise. This changes
only the transport approximation; all outputs retain CINEMA-OT's background
confounder and treated-minus-counterfactual interpretation.
"""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import FastICA
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Timer, save_embeddings  # noqa: E402


def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def global_confounders(a, dim, threshold):
    from cinemaot.cinemaot import Xi

    sc.pp.pca(a, n_comps=max(dim, 50), svd_solver="arpack")
    transformed = FastICA(
        n_components=dim, random_state=0, whiten="arbitrary-variance"
    ).fit_transform(a.obsm["X_pca"][:, :dim])
    group = (a.obs["cond"].astype(str).to_numpy() == "disease").astype(int)
    xi = np.asarray([Xi(component, group).correlation for component in transformed.T])
    keep = xi < threshold
    if not keep.any():
        raise ValueError("No confounder components identified; increase --thres")
    return transformed[:, keep].astype(np.float64), xi


def minibatch_effect(
    confounders,
    expression,
    control_rows,
    treated_rows,
    batch_size,
    n_ensembles,
    smoothness,
    eps,
    seed,
):
    from cinemaot import sinkhorn_knopp as skp

    if len(control_rows) != len(treated_rows):
        raise ValueError("Current mini-batch runner requires equal treatment arms")
    rng = np.random.default_rng(seed)
    effect_sum = np.zeros(
        (len(treated_rows), expression.shape[1]), dtype=np.float64
    )
    treated_position = {row: pos for pos, row in enumerate(treated_rows)}
    scale = smoothness * confounders.shape[1]

    for ensemble in range(n_ensembles):
        controls = rng.permutation(control_rows)
        treated = rng.permutation(treated_rows)
        for left in range(0, len(treated), batch_size):
            control_batch = controls[left : left + batch_size]
            treated_batch = treated[left : left + batch_size]
            distance = pairwise_distances(
                confounders[control_batch], confounders[treated_batch]
            )
            affinity = np.exp(-(distance * distance) / scale)
            n_control, n_treated = affinity.shape
            row_marginal = np.full((n_control, 1), 1.0 / n_control)
            col_marginal = np.full((n_treated, 1), 1.0 / n_treated)
            sinkhorn = skp.SinkhornKnopp(
                setr=row_marginal, setc=col_marginal, epsilon=eps
            )
            transport = sinkhorn.fit(affinity).T
            transport /= transport.sum(axis=1, keepdims=True)
            counterfactual = transport @ expression[control_batch]
            batch_effect = expression[treated_batch] - counterfactual
            output_rows = np.asarray(
                [treated_position[row] for row in treated_batch], dtype=int
            )
            effect_sum[output_rows] += batch_effect
        print(
            f"[mini-cinemaot] ensemble {ensemble + 1}/{n_ensembles} complete",
            flush=True,
        )
    return (effect_sum / n_ensembles).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--thres", type=float, default=0.15)
    parser.add_argument("--smoothness", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--transport_batch_size", type=int, default=2048)
    parser.add_argument("--n_ensembles", type=int, default=3)
    args = parser.parse_args()

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(
        args.out_root,
        tag,
        f"cinemaot_minibatch_b{args.transport_batch_size}_e{args.n_ensembles}_seed{args.seed}",
    )
    data = ad.read_h5ad(args.data)
    data.layers["counts"] = data.X.copy()
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)

    names_all, background_all, salient_all = [], [], []
    kept_components = []
    with Timer() as timer:
        for timepoint in data.obs["timepoint"].astype(str).unique():
            one = data[data.obs["timepoint"].astype(str) == timepoint].copy()
            one.obs["cond"] = one.obs["condition"].astype(str)
            confounders, xi = global_confounders(one, args.dim, args.thres)
            expression = dense(one.X).astype(np.float64)
            control = np.flatnonzero(one.obs["cond"].to_numpy() == "control")
            treated = np.flatnonzero(one.obs["cond"].to_numpy() == "disease")
            treated_effect = minibatch_effect(
                confounders,
                expression,
                control,
                treated,
                args.transport_batch_size,
                args.n_ensembles,
                args.smoothness,
                args.eps,
                args.seed,
            )
            effect = np.zeros((one.n_obs, one.n_vars), dtype=np.float32)
            effect[treated] = treated_effect
            names_all.append(one.obs_names.to_numpy())
            background_all.append(confounders.astype(np.float32))
            salient_all.append(effect)
            kept_components.append(int((xi < args.thres).sum()))

    width = max(x.shape[1] for x in background_all)
    background_all = [
        np.pad(x, ((0, 0), (0, width - x.shape[1]))) for x in background_all
    ]
    save_embeddings(
        out_dir,
        background=np.concatenate(background_all),
        salient=np.concatenate(salient_all),
        obs_names=np.concatenate(names_all),
        method="cinemaot_minibatch",
        single_rep=False,
        runtime_s=timer.seconds,
        dim=args.dim,
        thres=args.thres,
        smoothness=args.smoothness,
        eps=args.eps,
        transport_batch_size=args.transport_batch_size,
        n_ensembles=args.n_ensembles,
        kept_confounder_components=kept_components,
        exact_cinemaot=False,
        note=(
            "Global PCA/ICA and confounder selection; randomized arm-matched "
            "balanced Sinkhorn mini-batches; treatment effects averaged across "
            "partitions. Bounded memory O(batch_size^2), compute O(N*batch_size)."
        ),
    )


if __name__ == "__main__":
    main()
