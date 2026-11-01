"""Baselines mapped onto all four quadrants of the 2x2, so the same matrix applies.

Every method has to fill four slots -- `background`, `spatial`, `salient`,
`salient_spatial` -- even when its design provides fewer distinct representations.
The mapping below is the honest one for each method's actual claim:

    pca              one embedding, reused in all four slots
    cellcharter      scVI latent -> intrinsic slots; the CellCharter neighbourhood
                     aggregation of it -> spatial slots. This is CellCharter's own
                     construction, split across the intrinsic/spatial axis.
    contrastivevi    z -> background, s -> salient (its real contrast), and the
                     CC aggregation of each -> the corresponding spatial slot
    cinemaot         confounder embedding -> background, per-cell treatment effect
                     -> salient, each CC-aggregated for its spatial slot

The point of giving the SPATIAL-ONLY methods the same representation in both the
background and salient slots is not to flatter them: it is what their design
actually asserts. CellCharter has no notion of a control arm, so its single
domain map is its answer to both "what is the tissue" and "what did the condition
do", and the matrix shows what that costs. Likewise reusing one embedding across
the intrinsic/spatial axis for PCA.

Conversely, giving contrastiveVI and CINEMA-OT a CC-aggregated spatial slot is
generous rather than punitive -- neither has a spatial branch, so this hands them
the neighbourhood pooling they lack and asks whether that alone recovers the
spatial quadrants.

    python run_baselines_2x2.py --data data/simv3_d1_s0.h5ad --out_root results_v3
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
BATCH_KEY = "timepoint"  # matched levels contain both control and disease


def _prep(adata):
    a = adata.copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def _cc_aggregate(a, rep, name):
    """CellCharter neighbourhood aggregation, returning the pooled block only.

    `aggregate_neighbors` concatenates [own | hop1 | hop2 | hop3]; the own block is
    the input we already have, so the SPATIAL slot takes the hop blocks. Keeping
    the own block would make the spatial slot a superset of the intrinsic one and
    guarantee it scores at least as well on the intrinsic quadrant, which would be
    an artefact of the construction rather than a property of the method.
    """
    import cellcharter as cc
    import squidpy as sq

    a.obsm[name] = np.asarray(rep, dtype=np.float32)
    if "spatial_connectivities" not in a.obsp:
        sq.gr.spatial_neighbors(a, library_key="slice_id", coord_type="generic",
                                delaunay=True, spatial_key="spatial")
        cc.gr.remove_long_links(a)
    cc.gr.aggregate_neighbors(a, n_layers=N_HOPS, use_rep=name,
                              out_key=f"{name}_cc", sample_key="slice_id")
    full = np.asarray(a.obsm[f"{name}_cc"], dtype=np.float32)
    d = rep.shape[1]
    return full[:, d:]          # hop blocks only


def run_pca(adata, out_dir, seed, **_):
    a = _prep(adata)
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    sp_ = _cc_aggregate(a, X, "pca_rep")
    save_embeddings(out_dir, background=X, salient=X, spatial=sp_, salient_spatial=sp_,
                    obs_names=adata.obs_names.to_numpy(), method="pca", single_rep=True,
                    runtime_s=0.0, n_layer=0, note="one embedding in the intrinsic slots, its "
                                        "CC neighbourhood aggregation in the spatial slots")


def run_cellcharter(adata, out_dir, seed, max_epochs=150, batch_size=2048, **_):
    import scvi
    a = _prep(adata)
    with Timer() as t:
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(a, layer="counts", batch_key=BATCH_KEY)
        m = scvi.model.SCVI(a, n_latent=N_LATENT)
        m.train(max_epochs=max_epochs, batch_size=batch_size, early_stopping=True,
                early_stopping_monitor="elbo_validation")
        z = m.get_latent_representation().astype(np.float32)
        z_sp = _cc_aggregate(a, z, "scvi_rep")
    save_embeddings(out_dir, background=z, salient=z, spatial=z_sp, salient_spatial=z_sp,
                    obs_names=adata.obs_names.to_numpy(), method="cellcharter",
                    single_rep=True, runtime_s=t.seconds, n_layer=0, batch_key=BATCH_KEY,
                    note="scVI latent -> intrinsic slots; CellCharter 3-hop "
                         "aggregation of it -> spatial slots. No control arm, so the "
                         "same representation answers both background and salient.")


def run_contrastivevi(adata, out_dir, seed, max_epochs=150, batch_size=2048, **_):
    import scvi
    from scvi.external import ContrastiveVI
    a = _prep(adata)
    with Timer() as t:
        scvi.settings.seed = seed
        ContrastiveVI.setup_anndata(a, layer="counts", batch_key=BATCH_KEY)
        m = ContrastiveVI(a, n_me=0, n_background_latent=N_LATENT,
                          n_salient_latent=N_LATENT, use_observed_lib_size=True)
        bg = np.where(a.obs["condition"].astype(str) != "disease")[0]
        tg = np.where(a.obs["condition"].astype(str) == "disease")[0]
        m.train(background_indices=bg, target_indices=tg, max_epochs=max_epochs,
                batch_size=batch_size, train_size=1.0, early_stopping=False,
                gradient_clip_val=1.0)
        z = m.get_latent_representation(a, representation_kind="background").astype(np.float32)
        s = m.get_latent_representation(a, representation_kind="salient").astype(np.float32)
        z_sp = _cc_aggregate(a, z, "cvi_bg")
        s_sp = _cc_aggregate(a, s, "cvi_sal")
    save_embeddings(out_dir, background=z, salient=s, spatial=z_sp, salient_spatial=s_sp,
                    obs_names=adata.obs_names.to_numpy(), method="contrastivevi_cc4",
                    single_rep=False, runtime_s=t.seconds, n_layer=0, batch_key=BATCH_KEY,
                    note="upstream ContrastiveVI (no spatial branch) + CellCharter "
                         "aggregation of each latent as its spatial slot")


RUNNERS = dict(pca=run_pca, cellcharter=run_cellcharter, contrastivevi=run_contrastivevi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--methods", nargs="+", default=list(RUNNERS))
    ap.add_argument("--out_root", default="results_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=2048)
    args = ap.parse_args()

    tag = os.path.basename(args.data).replace(".h5ad", "")
    adata = ad.read_h5ad(args.data)
    for m in args.methods:
        out_dir = os.path.join(args.out_root, tag, f"{m}_2x2_seed{args.seed}")
        print(f"\n===== {m} on {tag} =====", flush=True)
        RUNNERS[m](adata, out_dir, args.seed, max_epochs=args.max_epochs,
                   batch_size=args.batch_size)


if __name__ == "__main__":
    main()
