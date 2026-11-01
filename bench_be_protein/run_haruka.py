"""Default Haruka-MENDER for paired continuous CODEX protein data."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(__file__)
MERFISH = os.path.join(HERE, "..", "bench_merfish")
sys.path.insert(0, MERFISH)
sys.path.insert(0, HERE)

from common import Timer, save_embeddings  # noqa: E402
from mender_composition import aggregate_mender_compositions  # noqa: E402
from protein_models import (  # noqa: E402
    PROTEIN_LAYER,
    attach_batch_aware_protein_mse_decoder,
    attach_protein_mse_decoder,
    prepare_protein_adata,
    protein_transform_description,
)
from run_haruka_cvi_scviva import (  # noqa: E402
    DualNicheDecoder,
    neighbor_index,
    niche_features,
    pseudo_states,
)


def augment_with_targets(base: ad.AnnData, targets: np.ndarray) -> ad.AnnData:
    protein = np.asarray(base.layers[PROTEIN_LAYER], dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    names = list(base.var_names.astype(str)) + [
        f"niche_target_{j}" for j in range(targets.shape[1])
    ]
    joined = np.concatenate([protein, targets], axis=1)
    out = ad.AnnData(
        X=joined.copy(),
        obs=base.obs.copy(),
        var=pd.DataFrame(index=pd.Index(names)),
    )
    out.layers[PROTEIN_LAYER] = joined.copy()
    for key, value in base.obsm.items():
        out.obsm[key] = np.asarray(value).copy()
    return out


def fit_reference(a, target_mask, args):
    import scvi
    from scvi.external import ContrastiveVI

    scvi.settings.seed = args.seed
    ContrastiveVI.setup_anndata(a, layer=PROTEIN_LAYER, batch_key=args.batch_key)
    model = ContrastiveVI(
        a,
        n_me=0,
        n_hidden=256,
        n_background_latent=args.n_latent,
        n_salient_latent=args.n_latent,
        use_observed_lib_size=True,
        wasserstein_penalty=0.2,
        zs_independence_penalty=args.zs_independence_penalty,
        symmetric_zs_independence=args.symmetric_zs_independence,
        batch_injection_mode=args.batch_injection_mode,
        gene_likelihood="normal",
    )
    if args.decoder_batch_dim > 0:
        attach_batch_aware_protein_mse_decoder(
            model,
            2 * args.n_latent,
            a.n_vars,
            n_batches=a.obs[args.batch_key].nunique(),
            batch_embedding_dim=args.decoder_batch_dim,
        )
    else:
        attach_protein_mse_decoder(model, 2 * args.n_latent, a.n_vars)
    model.train(
        background_indices=np.flatnonzero(~target_mask),
        target_indices=np.flatnonzero(target_mask),
        max_epochs=args.reference_epochs,
        batch_size=args.batch_size,
        train_size=1.0,
        early_stopping=False,
        gradient_clip_val=1.0,
    )
    z = model.get_latent_representation(a, representation_kind="background")
    s = model.get_latent_representation(a, representation_kind="salient")
    return z.astype(np.float32), s.astype(np.float32)


def auto_mender_radius(a, sample_key):
    sample = a.obs[sample_key].astype(str).to_numpy()
    coords = np.asarray(a.obsm["spatial"], dtype=np.float64)
    medians = []
    for value in np.unique(sample):
        rows = np.flatnonzero(sample == value)
        distances = cKDTree(coords[rows]).query(coords[rows], k=2)[0][:, 1]
        medians.append(float(np.median(distances)))
    return float(np.median(medians))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="bench_be_protein/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference_epochs", type=int, default=75)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_bg_states", type=int, default=12)
    parser.add_argument("--n_sal_states", type=int, default=8)
    parser.add_argument("--k_niche", type=int, default=20)
    parser.add_argument("--spatial_weight", type=float, default=10.0)
    parser.add_argument("--mender_scale", type=int, default=6)
    parser.add_argument("--mender_radius", type=float, default=0.0,
                        help="0 chooses the median within-slice 1-NN distance")
    parser.add_argument("--batch_key", default="section")
    parser.add_argument("--sample_key", default="slice_id")
    parser.add_argument(
        "--decoder_batch_dim", type=int, default=0,
        help="Positive value injects a learned batch embedding only into the protein decoder",
    )
    parser.add_argument(
        "--batch_injection_mode",
        choices=("all", "salient_encoder", "decoder_only"),
        default="all",
        help="Where ContrastiveVI receives the registered batch covariate.",
    )
    parser.add_argument(
        "--zs_independence_penalty", type=float, default=0.0,
        help="Directional target-cell z<-s ridge-predictability penalty.",
    )
    parser.add_argument(
        "--symmetric_zs_independence", action="store_true",
        help="Also penalize target-cell s variation predictable from z.",
    )
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--target_label", default="disease")
    args = parser.parse_args()

    import scvi
    from scvi.external import ContrastiveVI

    raw = ad.read_h5ad(args.data)
    a = prepare_protein_adata(raw)
    target_mask = (
        a.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    )
    background_idx = np.flatnonzero(~target_mask)
    target_idx = np.flatnonzero(target_mask)
    if not len(background_idx) or not len(target_idx):
        raise ValueError("Haruka requires non-empty control and condition arms")

    with Timer() as timer:
        z_ref, s_ref = fit_reference(a, target_mask, args)
        neighbors = neighbor_index(a, args.k_niche, args.sample_key)
        bg_states, z_scaled = pseudo_states(
            z_ref, np.ones(a.n_obs, bool), args.n_bg_states, args.seed
        )
        sal_states, s_scaled = pseudo_states(
            s_ref, target_mask, args.n_sal_states, args.seed + 1
        )
        bg_comp, bg_act = niche_features(
            z_scaled, bg_states, neighbors, args.n_bg_states
        )
        sal_comp, sal_act = niche_features(
            s_scaled, sal_states, neighbors, args.n_sal_states
        )
        sal_comp[~target_mask], sal_act[~target_mask] = 0.0, 0.0
        targets = np.concatenate([bg_comp, bg_act, sal_comp, sal_act], axis=1)
        augmented = augment_with_targets(a, targets)

        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(
            augmented, layer=PROTEIN_LAYER, batch_key=args.batch_key
        )
        model = ContrastiveVI(
            augmented,
            n_me=targets.shape[1],
            n_hidden=256,
            n_background_latent=args.n_latent,
            n_salient_latent=args.n_latent,
            use_observed_lib_size=True,
            wasserstein_penalty=0.2,
            zs_independence_penalty=args.zs_independence_penalty,
            symmetric_zs_independence=args.symmetric_zs_independence,
            batch_injection_mode=args.batch_injection_mode,
            gene_likelihood="normal",
            me_weight=args.spatial_weight,
            me_reduction="mean",
            n_spatial_latent=0,
            niche_context=False,
        )
        if args.decoder_batch_dim > 0:
            attach_batch_aware_protein_mse_decoder(
                model,
                2 * args.n_latent,
                raw.n_vars,
                n_batches=augmented.obs[args.batch_key].nunique(),
                batch_embedding_dim=args.decoder_batch_dim,
            )
        else:
            attach_protein_mse_decoder(model, 2 * args.n_latent, raw.n_vars)
        bg_out = bg_comp.shape[1] + bg_act.shape[1]
        sal_out = sal_comp.shape[1] + sal_act.shape[1]
        model.module.decoder_me = DualNicheDecoder(
            args.n_latent, args.n_latent, bg_out, sal_out
        )
        cuts = np.cumsum(
            [0, bg_comp.shape[1], bg_act.shape[1], sal_comp.shape[1], sal_act.shape[1]]
        )

        def niche_loss(x, index, prediction, reduction="mean"):
            truth = x[..., index:]
            return torch.stack(
                [
                    (prediction[..., left:right] - truth[..., left:right])
                    .square()
                    .mean(-1)
                    for left, right in zip(cuts[:-1], cuts[1:])
                ]
            ).sum(0)

        model.module.me_reconstruction_loss = niche_loss
        model.train(
            background_indices=background_idx,
            target_indices=target_idx,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            train_size=1.0,
            early_stopping=False,
            gradient_clip_val=1.0,
        )
        z = model.get_latent_representation(
            augmented, representation_kind="background"
        ).astype(np.float32)
        s = model.get_latent_representation(
            augmented, representation_kind="salient"
        ).astype(np.float32)

        device = next(model.module.parameters()).device
        predictions = []
        model.module.eval()
        with torch.no_grad():
            for left in range(0, len(z), 8192):
                latent = torch.as_tensor(
                    np.concatenate([z[left:left + 8192], s[left:left + 8192]], axis=1),
                    device=device,
                )
                predictions.append(model.module.decoder_me(latent, None).cpu().numpy())
        predictions = np.concatenate(predictions).astype(np.float32)
        bg_decoder = predictions[:, :bg_out]
        sal_decoder = predictions[:, bg_out:]
        sal_decoder[~target_mask] = 0.0

        radius = args.mender_radius or auto_mender_radius(augmented, args.sample_key)
        spatial, salient_spatial = aggregate_mender_compositions(
            augmented.obsm["spatial"],
            augmented.obs[args.sample_key].astype(str).to_numpy(),
            bg_decoder[:, : args.n_bg_states],
            sal_decoder[:, : args.n_sal_states],
            scale=args.mender_scale,
            radius=radius,
        )

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    run_name = (
        f"haruka_mender_mse_decoderbatch{args.decoder_batch_dim}_seed{args.seed}"
        if args.decoder_batch_dim > 0
        else f"haruka_mender_mse_seed{args.seed}"
    )
    if args.zs_independence_penalty > 0:
        run_name = run_name.replace(
            f"_seed{args.seed}",
            f"_zsind{args.zs_independence_penalty:g}_seed{args.seed}",
        )
    if args.symmetric_zs_independence:
        run_name = run_name.replace(f"_seed{args.seed}", f"_sym_seed{args.seed}")
    if args.batch_injection_mode != "all":
        run_name = run_name.replace(
            f"_seed{args.seed}",
            f"_batchmode-{args.batch_injection_mode}_seed{args.seed}",
        )
    out_dir = os.path.join(args.out_root, tag, run_name)
    save_embeddings(
        out_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=augmented.obs_names.to_numpy(),
        method="haruka_mender_protein_mse",
        seed=args.seed,
        runtime_s=timer.seconds,
        max_epochs=args.max_epochs,
        reference_epochs=args.reference_epochs,
        n_latent=args.n_latent,
        n_bg_states=args.n_bg_states,
        n_sal_states=args.n_sal_states,
        k_niche=args.k_niche,
        spatial_weight=args.spatial_weight,
        mender_scale=args.mender_scale,
        mender_radius=radius,
        mender_radius_source=("auto_median_1nn" if args.mender_radius == 0 else "explicit"),
        likelihood="exact summed MSE",
        protein_transform=protein_transform_description(),
        training_cells="all control and disease cells; train_size=1.0",
        decoder_batch_injection=(args.decoder_batch_dim > 0),
        decoder_batch_key=args.batch_key,
        decoder_batch_embedding_dim=args.decoder_batch_dim,
        zs_independence_penalty=args.zs_independence_penalty,
        symmetric_zs_independence=args.symmetric_zs_independence,
        batch_injection_mode=args.batch_injection_mode,
        encoder_batch_covariate_from_setup_anndata=(
            args.batch_key if args.batch_injection_mode != "decoder_only" else None
        ),
        decoder_composition_only=True,
        supervised_labels=False,
        postfit=False,
    )


if __name__ == "__main__":
    main()
