"""Canonical Haruka runner: ContrastiveVI dual decoder + MENDER composition."""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import torch

MERFISH = os.path.join(os.path.dirname(__file__), "..", "bench_merfish")
sys.path.insert(0, MERFISH)

from common import Timer, save_embeddings  # noqa: E402
from mender_composition import aggregate_mender_compositions  # noqa: E402
from run_haruka_cvi_scviva import (  # noqa: E402
    DualNicheDecoder,
    augment_with_targets,
    fit_reference,
    load_reference,
    neighbor_index,
    niche_features,
    pseudo_states,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the default composition-only Haruka-MENDER method."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="results_haruka")
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
    parser.add_argument("--mender_scale", type=int, default=6)
    parser.add_argument("--mender_radius", type=float, default=15.0)
    parser.add_argument(
        "--mender_neighbors",
        type=int,
        default=15,
        help="Set each cell's base physical radius to its kth-neighbor distance.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_key", default="timepoint")
    parser.add_argument("--sample_key", default="slice_id")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--target_label", default="disease")
    args = parser.parse_args()

    import scvi
    from scvi.external import ContrastiveVI

    adata = ad.read_h5ad(args.data)
    required = {args.batch_key, args.sample_key, args.condition_key}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"Missing required obs columns: {sorted(missing)}")
    if "counts" not in adata.layers:
        raise KeyError("Haruka requires raw counts in adata.layers['counts']")
    if "spatial" not in adata.obsm:
        raise KeyError("Haruka-MENDER requires coordinates in adata.obsm['spatial']")

    target_mask = (
        adata.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    )
    background_idx = np.flatnonzero(~target_mask)
    target_idx = np.flatnonzero(target_mask)
    if not len(background_idx) or not len(target_idx):
        raise ValueError("ContrastiveVI requires non-empty target and background arms")

    with Timer() as timer:
        if args.initial_emb_dir:
            z_ref, s_ref = load_reference(adata, args.initial_emb_dir)
            reference_source = args.initial_emb_dir
        else:
            z_ref, s_ref = fit_reference(adata, args)
            reference_source = (
                f"internal ContrastiveVI ({args.reference_epochs} epochs)"
            )

        neighbors = neighbor_index(adata, args.k_niche, args.sample_key)
        bg_states, z_scaled = pseudo_states(
            z_ref, np.ones(adata.n_obs, bool), args.n_bg_states, args.seed
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
        augmented = augment_with_targets(adata, targets)

        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(
            augmented, layer="counts", batch_key=args.batch_key
        )
        model = ContrastiveVI(
            augmented,
            n_me=targets.shape[1],
            n_hidden=256,
            n_background_latent=args.n_latent,
            n_salient_latent=args.n_latent,
            use_observed_lib_size=True,
            wasserstein_penalty=0.2,
            gene_likelihood="poisson",
            me_weight=args.spatial_weight,
            me_reduction="mean",
            n_spatial_latent=0,
            niche_context=False,
        )
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
            datasplitter_kwargs={
                "num_workers": args.num_workers,
                "persistent_workers": args.num_workers > 0,
            },
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
                    np.concatenate(
                        [z[left : left + 8192], s[left : left + 8192]], axis=1
                    ),
                    device=device,
                )
                predictions.append(model.module.decoder_me(latent, None).cpu().numpy())
        predictions = np.concatenate(predictions).astype(np.float32)
        bg_decoder = predictions[:, :bg_out]
        sal_decoder = predictions[:, bg_out:]
        sal_decoder[~target_mask] = 0.0

        # The canonical spatial representations use composition only. Decoder
        # activation blocks are deliberately excluded from the MENDER readout.
        spatial, salient_spatial, radius_diagnostics = aggregate_mender_compositions(
            augmented.obsm["spatial"],
            augmented.obs[args.sample_key].astype(str).to_numpy(),
            bg_decoder[:, : args.n_bg_states],
            sal_decoder[:, : args.n_sal_states],
            scale=args.mender_scale,
            radius=args.mender_radius,
            radius_neighbors=args.mender_neighbors,
            return_diagnostics=True,
        )

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(args.out_root, tag, f"haruka_mender_seed{args.seed}")
    save_embeddings(
        out_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=augmented.obs_names.to_numpy(),
        method="haruka_mender",
        seed=args.seed,
        runtime_s=timer.seconds,
        reference_source=reference_source,
        max_epochs=args.max_epochs,
        n_latent=args.n_latent,
        n_bg_states=args.n_bg_states,
        n_sal_states=args.n_sal_states,
        k_niche=args.k_niche,
        spatial_weight=args.spatial_weight,
        mender_scale=args.mender_scale,
        mender_radius=args.mender_radius,
        mender_neighbors=args.mender_neighbors,
        mender_radius_mode="cell_local_kth_neighbor",
        default_haruka_method=True,
        spatial_readout=(
            "MENDER-style soft composition counts in non-overlapping physical "
            "rings; each cell's first radius is its kth-neighbor distance"
        ),
        decoder_composition_only=True,
        pseudo_state_source="unsupervised ContrastiveVI; no ground-truth labels",
        simvi_components=False,
        cellcharter_components=False,
        supervised_labels=False,
        postfit=False,
    )
    np.save(
        os.path.join(out_dir, "background_composition.npy"),
        bg_decoder[:, : args.n_bg_states],
    )
    np.save(
        os.path.join(out_dir, "salient_composition.npy"),
        sal_decoder[:, : args.n_sal_states],
    )
    with open(os.path.join(out_dir, "mender_radius_diagnostics.json"), "w") as handle:
        json.dump(radius_diagnostics, handle, indent=2)


if __name__ == "__main__":
    main()
