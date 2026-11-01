"""ContrastiveVI with scVIVA-style niche targets and CellCharter readout.

The original Haruka auxiliary head predicts a flattened COVET covariance block.
This replaces it with label-free analogues of scVIVA's niche composition and
per-state neighbor-expression targets for both contrastive arms. Pseudo-states
come from a spatially unaware ContrastiveVI reference, never ground-truth labels.
There is no SIMVI latent, GAT, spatial KL, or mutual-information loss.
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from common import Timer, load_embeddings, save_embeddings
from run_baselines_2x2 import _cc_aggregate


class DualNicheDecoder(torch.nn.Module):
    """Independent environment heads: z predicts background, s predicts salient."""

    def __init__(self, z_dim, s_dim, bg_out, sal_out, hidden=256):
        super().__init__()
        self.z_dim, self.s_dim = z_dim, s_dim
        block = lambda n_in, n_out: torch.nn.Sequential(  # noqa: E731
            torch.nn.Linear(n_in, hidden), torch.nn.LayerNorm(hidden),
            torch.nn.GELU(), torch.nn.Linear(hidden, n_out),
        )
        self.background = block(z_dim, bg_out)
        self.salient = block(s_dim, sal_out)

    def forward(self, latent, *_):
        z = latent[..., : self.z_dim]
        s = latent[..., self.z_dim : self.z_dim + self.s_dim]
        return torch.cat([self.background(z), self.salient(s)], dim=-1)


def prep_expression(adata):
    out = adata.copy()
    if "counts" not in out.layers:
        out.layers["counts"] = out.X.copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def load_reference(adata, result_dir):
    loaded = load_embeddings(result_dir)
    saved = pd.Index(loaded.get("obs_names", adata.obs_names).astype(str))
    pos = saved.get_indexer(pd.Index(adata.obs_names.astype(str)))
    if (pos < 0).any():
        raise ValueError("Reference embeddings do not cover every input cell")
    return loaded["background"][pos], loaded["salient"][pos]


def fit_reference(adata, args):
    import scvi
    from scvi.external import ContrastiveVI

    a = prep_expression(adata)
    target = a.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    scvi.settings.seed = args.seed
    ContrastiveVI.setup_anndata(a, layer="counts", batch_key=args.batch_key)
    model = ContrastiveVI(
        a, n_me=0, n_background_latent=args.n_latent,
        n_salient_latent=args.n_latent, use_observed_lib_size=True,
    )
    model.train(
        background_indices=np.flatnonzero(~target),
        target_indices=np.flatnonzero(target), max_epochs=args.reference_epochs,
        batch_size=args.batch_size, train_size=1.0, early_stopping=False,
        gradient_clip_val=1.0,
        datasplitter_kwargs={
            "num_workers": getattr(args, "num_workers", 0),
            "persistent_workers": getattr(args, "num_workers", 0) > 0,
        },
    )
    z = model.get_latent_representation(a, representation_kind="background")
    s = model.get_latent_representation(a, representation_kind="salient")
    return z.astype(np.float32), s.astype(np.float32)


def neighbor_index(adata, k, sample_key):
    sample = adata.obs[sample_key].astype(str).to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    result = np.empty((adata.n_obs, k), dtype=np.int64)
    for value in tqdm(
        np.unique(sample), desc="Spatial kNN", unit="slice", leave=False
    ):
        rows = np.flatnonzero(sample == value)
        if len(rows) <= k:
            raise ValueError(f"Sample {value!r} has {len(rows)} cells; need > k={k}")
        local = NearestNeighbors(n_neighbors=k + 1).fit(coords[rows]).kneighbors(
            return_distance=False
        )[:, 1:]
        result[rows] = rows[local]
    return result


def pseudo_states(rep, fit_mask, n_states, seed):
    scaled = StandardScaler().fit_transform(np.asarray(rep, dtype=np.float64))
    model = MiniBatchKMeans(
        n_clusters=n_states, random_state=seed, n_init=10,
        batch_size=4096, max_iter=300,
    ).fit(scaled[fit_mask])
    return model.predict(scaled).astype(np.int32), scaled.astype(np.float32)


def niche_features(rep, states, neighbors, n_states):
    """scVIVA composition plus per-pseudo-state mean neighbor embedding."""
    n, k = neighbors.shape
    d = rep.shape[1]
    rows = np.repeat(np.arange(n), k)
    nbr = neighbors.reshape(-1)
    nbr_state = states[nbr]
    composition = np.zeros((n, n_states), dtype=np.float32)
    np.add.at(composition, (rows, nbr_state), 1.0)
    composition /= float(k)
    activation = np.zeros((n, n_states, d), dtype=np.float32)
    for state in range(n_states):
        take = nbr_state == state
        if not np.any(take):
            continue
        np.add.at(activation[:, state], rows[take], rep[nbr[take]])
        denom = composition[:, state] * float(k)
        present = denom > 0
        activation[present, state] /= denom[present, None]
    return composition, activation.reshape(n, n_states * d)


def augment_with_targets(
    adata, targets, *, force_dense=False, preserve_processed_x=False
):
    # scVI is registered against layers['counts']; X is not an optimization
    # input.  Large Xenium objects can therefore preserve their existing
    # processed X and avoid an otherwise redundant full-matrix normalization.
    base = adata.copy() if preserve_processed_x else prep_expression(adata)
    targets = np.asarray(targets, dtype=np.float32)
    names = list(base.var_names.astype(str)) + [
        f"niche_target_{j}" for j in range(targets.shape[1])
    ]
    counts = base.layers["counts"]
    if sp.issparse(base.X) and sp.issparse(counts) and not force_dense:
        # AnnData/scvi can slice sparse matrices into dense minibatches.  Keeping
        # the augmented matrices sparse avoids materializing two n_cells x
        # (n_genes + n_targets) dense arrays, which is prohibitive for million-
        # cell, 5K-gene Xenium experiments but does not alter any target or loss.
        target_csr = sp.csr_matrix(targets)
        x_augmented = sp.hstack(
            [base.X.astype(np.float32), target_csr], format="csr"
        )
        counts_augmented = sp.hstack(
            [counts.astype(np.float32), target_csr], format="csr"
        )
    else:
        x = base.X.toarray() if sp.issparse(base.X) else np.asarray(base.X)
        counts_array = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
        x_augmented = np.concatenate([x.astype(np.float32), targets], axis=1)
        counts_augmented = np.concatenate(
            [counts_array.astype(np.float32), targets], axis=1
        )
    out = ad.AnnData(
        X=x_augmented,
        obs=base.obs.copy(), var=pd.DataFrame(index=pd.Index(names)),
    )
    out.layers["counts"] = counts_augmented
    for key, value in base.obsm.items():
        out.obsm[key] = np.asarray(value).copy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="results_cvi_scviva_cellcharter")
    parser.add_argument("--initial_emb_dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference_epochs", type=int, default=75)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_bg_states", type=int, default=12)
    parser.add_argument("--n_sal_states", type=int, default=8)
    parser.add_argument("--k_niche", type=int, default=20)
    parser.add_argument("--spatial_weight", type=float, default=10.0)
    parser.add_argument("--batch_key", default="timepoint")
    parser.add_argument("--sample_key", default="slice_id")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--target_label", default="disease")
    args = parser.parse_args()

    import scvi
    from scvi.external import ContrastiveVI

    adata = ad.read_h5ad(args.data)
    required = {args.batch_key, args.sample_key, args.condition_key}
    missing = sorted(required.difference(adata.obs.columns))
    if missing or "spatial" not in adata.obsm:
        raise KeyError(f"Missing obs={missing}, spatial={'spatial' not in adata.obsm}")
    target_mask = adata.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    background_idx, target_idx = np.flatnonzero(~target_mask), np.flatnonzero(target_mask)
    if not len(background_idx) or not len(target_idx):
        raise ValueError("ContrastiveVI requires non-empty target and background arms")

    with Timer() as timer:
        if args.initial_emb_dir:
            z_ref, s_ref = load_reference(adata, args.initial_emb_dir)
            reference_source = args.initial_emb_dir
        else:
            z_ref, s_ref = fit_reference(adata, args)
            reference_source = f"internal ContrastiveVI ({args.reference_epochs} epochs)"
        neighbors = neighbor_index(adata, args.k_niche, args.sample_key)
        bg_states, z_scaled = pseudo_states(
            z_ref, np.ones(adata.n_obs, bool), args.n_bg_states, args.seed
        )
        sal_states, s_scaled = pseudo_states(
            s_ref, target_mask, args.n_sal_states, args.seed + 1
        )
        bg_comp, bg_act = niche_features(z_scaled, bg_states, neighbors, args.n_bg_states)
        sal_comp, sal_act = niche_features(s_scaled, sal_states, neighbors, args.n_sal_states)
        sal_comp[~target_mask], sal_act[~target_mask] = 0.0, 0.0
        targets = np.concatenate([bg_comp, bg_act, sal_comp, sal_act], axis=1)
        a = augment_with_targets(adata, targets)

        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(a, layer="counts", batch_key=args.batch_key)
        model = ContrastiveVI(
            a, n_me=targets.shape[1], n_hidden=256,
            n_background_latent=args.n_latent, n_salient_latent=args.n_latent,
            use_observed_lib_size=True, wasserstein_penalty=0.2,
            gene_likelihood="poisson", me_weight=args.spatial_weight,
            me_reduction="mean", n_spatial_latent=0, niche_context=False,
        )
        bg_out = bg_comp.shape[1] + bg_act.shape[1]
        sal_out = sal_comp.shape[1] + sal_act.shape[1]
        model.module.decoder_me = DualNicheDecoder(
            args.n_latent, args.n_latent, bg_out, sal_out
        )
        cuts = np.cumsum([0, bg_comp.shape[1], bg_act.shape[1],
                          sal_comp.shape[1], sal_act.shape[1]])

        def niche_loss(x, index, prediction, reduction="mean"):
            truth = x[..., index:]
            terms = [
                (prediction[..., left:right] - truth[..., left:right]).square().mean(-1)
                for left, right in zip(cuts[:-1], cuts[1:])
            ]
            return torch.stack(terms).sum(0)

        model.module.me_reconstruction_loss = niche_loss
        model.train(
            background_indices=background_idx, target_indices=target_idx,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            train_size=1.0, early_stopping=False, gradient_clip_val=1.0,
        )
        z = model.get_latent_representation(a, representation_kind="background").astype(np.float32)
        s = model.get_latent_representation(a, representation_kind="salient").astype(np.float32)
        s_for_spatial = s.copy()
        s_for_spatial[~target_mask] = 0.0
        z_sp = _cc_aggregate(a, z, "haruka_scviva_background")
        s_sp = _cc_aggregate(a, s_for_spatial, "haruka_scviva_salient")

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(args.out_root, tag, f"haruka_cvi_scviva_cc_seed{args.seed}")
    save_embeddings(
        out_dir, background=z, salient=s, spatial=z_sp,
        salient_spatial=s_sp, obs_names=a.obs_names.to_numpy(),
        method="haruka_cvi_scviva_cellcharter", seed=args.seed,
        runtime_s=timer.seconds, reference_source=reference_source,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        n_latent=args.n_latent, n_bg_states=args.n_bg_states,
        n_sal_states=args.n_sal_states, k_niche=args.k_niche,
        spatial_weight=args.spatial_weight,
        niche_target_widths={"bg_composition": bg_comp.shape[1],
            "bg_expression": bg_act.shape[1], "sal_composition": sal_comp.shape[1],
            "sal_expression": sal_act.shape[1]},
        pseudo_state_source="unsupervised ContrastiveVI; no ground-truth labels",
        intrinsic_model="ContrastiveVI", simvi_components=False,
        spatial_training_target="scVIVA-style composition + per-state neighbor expression",
        spatial_readout="CellCharter aggregate_neighbors, 3 hop blocks",
        supervised_labels=False, postfit=False,
    )


if __name__ == "__main__":
    main()
