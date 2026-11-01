"""CINEMA-OT on paired BE CODEX proteins (run in the `vio` environment)."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA

MERFISH = os.path.join(os.path.dirname(__file__), "..", "bench_merfish")
sys.path.insert(0, MERFISH)
from common import Timer, save_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="bench_be_protein/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--thres", type=float, default=0.15)
    args = parser.parse_args()

    import cinemaot as co

    a = ad.read_h5ad(args.data)
    raw = a.layers["protein"] if "protein" in a.layers else a.X
    raw = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
    a.X = ((np.clip(raw.astype(np.float32), -5.0, 5.0) + 5.0) / 5.0)
    a.obs["cond"] = a.obs["condition"].astype(str)
    a.obsm["X_pca"] = PCA(
        n_components=min(max(args.dim, 50), a.n_vars), random_state=args.seed
    ).fit_transform(a.X)

    np.random.seed(args.seed)
    with Timer() as timer:
        cf, ot, effect = co.cinemaot.cinemaot_unweighted(
            a, obs_label="cond", ref_label="disease", expr_label="control",
            dim=args.dim, thres=args.thres, smoothness=1e-4, eps=1e-3,
            mode="parametric",
        )
    treated = a.obs["cond"].to_numpy() == "disease"
    effect_x = effect.X.toarray() if sp.issparse(effect.X) else np.asarray(effect.X)
    effect_full = np.zeros((a.n_obs, effect_x.shape[1]), dtype=np.float32)
    effect_full[treated] = effect_x.astype(np.float32)
    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(args.out_root, tag, f"cinemaot_raw_seed{args.seed}")
    save_embeddings(
        out_dir, background=np.asarray(cf, dtype=np.float32),
        salient=effect_full, obs_names=a.obs_names.to_numpy(),
        method="cinemaot_protein", runtime_s=timer.seconds,
        likelihood="not applicable (optimal transport)",
        protein_transform="source feature-z-score clipped to [-5,5], shifted/scaled to [0,2]",
        note="CINEMA-OT on continuous CODEX protein intensity; ref=disease",
    )


if __name__ == "__main__":
    main()
