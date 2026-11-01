"""scVIVA-target Haruka with decoder spatial slots and optional CellCharter slots."""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np
import torch

from common import Timer, save_embeddings
from run_baselines_2x2 import _cc_aggregate
from run_haruka_cvi_scviva import (
    DualNicheDecoder,
    augment_with_targets,
    fit_reference,
    load_reference,
    neighbor_index,
    niche_features,
    pseudo_states,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="results_cvi_scviva_decoder")
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
            return torch.stack([
                (prediction[..., left:right] - truth[..., left:right]).square().mean(-1)
                for left, right in zip(cuts[:-1], cuts[1:])
            ]).sum(0)

        model.module.me_reconstruction_loss = niche_loss
        model.train(
            background_indices=background_idx, target_indices=target_idx,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            train_size=1.0, early_stopping=False, gradient_clip_val=1.0,
        )
        z = model.get_latent_representation(a, representation_kind="background").astype(np.float32)
        s = model.get_latent_representation(a, representation_kind="salient").astype(np.float32)

        device = next(model.module.parameters()).device
        pred = []
        model.module.eval()
        with torch.no_grad():
            for left in range(0, len(z), 8192):
                latent = torch.as_tensor(
                    np.concatenate([z[left:left + 8192], s[left:left + 8192]], axis=1),
                    device=device,
                )
                pred.append(model.module.decoder_me(latent, None).cpu().numpy())
        pred = np.concatenate(pred).astype(np.float32)
        bg_niche, sal_niche = pred[:, :bg_out], pred[:, bg_out:]
        sal_niche[~target_mask] = 0.0

        s_cc = s.copy()
        s_cc[~target_mask] = 0.0
        z_sp_cc = _cc_aggregate(a, z, "haruka_scviva_background")
        s_sp_cc = _cc_aggregate(a, s_cc, "haruka_scviva_salient")

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(args.out_root, tag, f"haruka_cvi_scviva_decoder_seed{args.seed}")
    save_embeddings(
        out_dir, background=z, salient=s, spatial=bg_niche,
        salient_spatial=sal_niche, obs_names=a.obs_names.to_numpy(),
        method="haruka_cvi_scviva_decoder", seed=args.seed,
        runtime_s=timer.seconds, reference_source=reference_source,
        max_epochs=args.max_epochs, n_latent=args.n_latent,
        n_bg_states=args.n_bg_states, n_sal_states=args.n_sal_states,
        k_niche=args.k_niche, spatial_weight=args.spatial_weight,
        pseudo_state_source="unsupervised ContrastiveVI; no ground-truth labels",
        spatial_readout="scVIVA-style dual niche decoder",
        alternate_spatial_readout="CellCharter aggregate_neighbors, 3 hop blocks",
        simvi_components=False, supervised_labels=False, postfit=False,
    )
    np.save(os.path.join(out_dir, "spatial_cellcharter.npy"), z_sp_cc)
    np.save(os.path.join(out_dir, "salient_spatial_cellcharter.npy"), s_sp_cc)


if __name__ == "__main__":
    main()
