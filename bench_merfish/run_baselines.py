"""Non-Haruka baselines that all live in the `cellcharter` conda env.

    pca            log-normalised expression, no spatial and no contrast.
                   The floor: what you get by clustering cells.
    cellcharter    scVI (batched on slice) -> Delaunay graph -> 3-hop neighbourhood
                   concatenation. The state of the art for spatial DOMAINS, but it
                   has no notion of a control arm, so it must answer both the
                   anatomy question and the treatment question with one embedding.
    contrastivevi  upstream `scvi.external.ContrastiveVI`, unpatched. The state of
                   the art for CONTRAST, but with no notion of space.
    contrastivevi_cc
                   contrastiveVI's two latents put through CellCharter's
                   neighbourhood aggregation. This is the control that matters for
                   interpreting Haruka: Haruka's `extract_rep` ALSO ends in
                   `cc.gr.aggregate_neighbors`, so without this arm any Haruka win
                   could just be the aggregation step, not the model.

Every method writes representations for ALL cells (both arms). Restricting to the
treated arm is the scorer's job, and the background-vs-condition leak test needs
control cells present.

    python run_baselines.py --data data/sim_d0.6_s0.h5ad --methods pca cellcharter \
        contrastivevi contrastivevi_cc --out_root results
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from common import Timer, save_embeddings

N_LATENT = 32
N_HOPS = 3


def _prep(adata):
    a = adata.copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def _spatial_graph(a):
    import cellcharter as cc
    import squidpy as sq
    sq.gr.spatial_neighbors(a, library_key="slice_id", coord_type="generic",
                            delaunay=True, spatial_key="spatial")
    cc.gr.remove_long_links(a)
    return a


def _aggregate(a, use_rep, out_key, n_layers=N_HOPS):
    import cellcharter as cc
    cc.gr.aggregate_neighbors(a, n_layers=n_layers, use_rep=use_rep,
                              out_key=out_key, sample_key="slice_id")
    return a.obsm[out_key]


def run_pca(adata, out_dir, seed=0, **_):
    a = _prep(adata)
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    with Timer() as t:
        pass
    save_embeddings(out_dir, background=X, salient=X, obs_names=adata.obs_names.to_numpy(),
                    method="pca", single_rep=True, runtime_s=t.seconds,
                    note="log-normalised expression; the scorer applies PCA-50 + GMM")


def run_cellcharter(adata, out_dir, seed=0, max_epochs=200, **_):
    import scvi
    a = _prep(adata)
    with Timer() as t:
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(a, layer="counts", batch_key="slice_id")
        model = scvi.model.SCVI(a, n_latent=N_LATENT)
        model.train(max_epochs=max_epochs, early_stopping=True,
                    early_stopping_monitor="elbo_validation")
        a.obsm["X_scVI"] = model.get_latent_representation().astype(np.float32)
        _spatial_graph(a)
        rep = _aggregate(a, "X_scVI", "X_cellcharter")
    save_embeddings(out_dir, background=rep, salient=rep,
                    obs_names=adata.obs_names.to_numpy(), method="cellcharter",
                    single_rep=True, runtime_s=t.seconds, n_latent=N_LATENT, n_hops=N_HOPS,
                    note="scVI(batch=slice_id) + 3-hop CellCharter neighbourhood concat")


def _fit_contrastivevi(adata, seed, max_epochs, batch_size):
    import scvi
    from scvi.external import ContrastiveVI
    a = _prep(adata)
    scvi.settings.seed = seed
    ContrastiveVI.setup_anndata(a, layer="counts", batch_key="slice_id")
    model = ContrastiveVI(a, n_background_latent=N_LATENT, n_salient_latent=N_LATENT,
                          use_observed_lib_size=True)
    bg = np.where(a.obs["condition"].astype(str) != "disease")[0]
    tg = np.where(a.obs["condition"].astype(str) == "disease")[0]
    model.train(background_indices=bg, target_indices=tg, max_epochs=max_epochs,
                batch_size=batch_size, train_size=1.0, early_stopping=False,
                gradient_clip_val=1.0)
    back = model.get_latent_representation(a, representation_kind="background").astype(np.float32)
    sal = model.get_latent_representation(a, representation_kind="salient").astype(np.float32)
    return a, back, sal


def run_contrastivevi(adata, out_dir, seed=0, max_epochs=400, batch_size=512, **_):
    with Timer() as t:
        _, back, sal = _fit_contrastivevi(adata, seed, max_epochs, batch_size)
    save_embeddings(out_dir, background=back, salient=sal,
                    obs_names=adata.obs_names.to_numpy(), method="contrastivevi",
                    single_rep=False, runtime_s=t.seconds, n_latent=N_LATENT,
                    max_epochs=max_epochs, note="upstream scvi.external.ContrastiveVI")


def run_contrastivevi_cc(adata, out_dir, seed=0, max_epochs=400, batch_size=512, **_):
    with Timer() as t:
        a, back, sal = _fit_contrastivevi(adata, seed, max_epochs, batch_size)
        a.obsm["bg"], a.obsm["sal"] = back, sal
        _spatial_graph(a)
        back_a = _aggregate(a, "bg", "X_cc_bg")
        sal_a = _aggregate(a, "sal", "X_cc_sal")
    save_embeddings(out_dir, background=back_a, salient=sal_a,
                    obs_names=adata.obs_names.to_numpy(), method="contrastivevi_cc",
                    single_rep=False, runtime_s=t.seconds, n_latent=N_LATENT, n_hops=N_HOPS,
                    max_epochs=max_epochs,
                    note="upstream ContrastiveVI + CellCharter 3-hop aggregation; "
                         "isolates how much of any spatial-contrastive win is the "
                         "aggregation step alone")


def run_contrastivevi_pair(adata, out_dir_root, tag, seed=0, max_epochs=400,
                           batch_size=512, **_):
    """Both contrastiveVI arms from ONE fit.

    `contrastivevi` and `contrastivevi_cc` differ only in whether CellCharter's
    neighbourhood aggregation is applied afterwards. Fitting the model twice would
    double the most expensive baseline AND introduce a training-seed difference
    between the two arms, which would muddy exactly the comparison the pair
    exists to make.
    """
    with Timer() as t:
        a, back, sal = _fit_contrastivevi(adata, seed, max_epochs, batch_size)
    save_embeddings(os.path.join(out_dir_root, tag, f"contrastivevi_seed{seed}"),
                    background=back, salient=sal,
                    obs_names=adata.obs_names.to_numpy(), method="contrastivevi",
                    single_rep=False, runtime_s=t.seconds, n_latent=N_LATENT,
                    max_epochs=max_epochs, note="upstream scvi.external.ContrastiveVI")

    with Timer() as t2:
        a.obsm["bg"], a.obsm["sal"] = back, sal
        _spatial_graph(a)
        back_a = _aggregate(a, "bg", "X_cc_bg")
        sal_a = _aggregate(a, "sal", "X_cc_sal")
    save_embeddings(os.path.join(out_dir_root, tag, f"contrastivevi_cc_seed{seed}"),
                    background=back_a, salient=sal_a,
                    obs_names=adata.obs_names.to_numpy(), method="contrastivevi_cc",
                    single_rep=False, runtime_s=t.seconds + t2.seconds,
                    n_latent=N_LATENT, n_hops=N_HOPS, max_epochs=max_epochs,
                    note="same fit as `contrastivevi`, plus CellCharter 3-hop "
                         "aggregation; isolates how much of any spatial-contrastive "
                         "win is the aggregation step alone")


RUNNERS = dict(pca=run_pca, cellcharter=run_cellcharter,
               contrastivevi=run_contrastivevi, contrastivevi_cc=run_contrastivevi_cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--methods", nargs="+", default=list(RUNNERS))
    ap.add_argument("--out_root", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    tag = os.path.basename(args.data).replace("sim_", "").replace(".h5ad", "")
    adata = ad.read_h5ad(args.data)

    methods = list(args.methods)
    if "contrastivevi" in methods and "contrastivevi_cc" in methods:
        methods = [m for m in methods if not m.startswith("contrastivevi")]
        print(f"\n===== contrastivevi (+cc) on {tag} =====", flush=True)
        run_contrastivevi_pair(adata, args.out_root, tag, seed=args.seed,
                               max_epochs=args.max_epochs, batch_size=args.batch_size)

    for m in methods:
        out_dir = os.path.join(args.out_root, tag, f"{m}_seed{args.seed}")
        print(f"\n===== {m} on {tag} =====", flush=True)
        RUNNERS[m](adata, out_dir, seed=args.seed, max_epochs=args.max_epochs,
                   batch_size=args.batch_size)


if __name__ == "__main__":
    main()
