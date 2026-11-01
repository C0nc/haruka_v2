"""Haruka with paired identifiability and density-adaptive MENDER fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
MERFISH = os.path.join(HERE, "..", "bench_merfish")
sys.path.insert(0, MERFISH)
sys.path.insert(0, HERE)

from common import Timer, save_embeddings  # noqa: E402
from mender_composition import _normalize_log_counts, valid_composition  # noqa: E402
from protein_models import (  # noqa: E402
    PROTEIN_LAYER,
    attach_protein_mse_decoder,
    prepare_protein_adata,
    protein_transform_description,
)
from run_haruka import augment_with_targets  # noqa: E402
from run_haruka_cvi_scviva import (  # noqa: E402
    DualNicheDecoder,
    neighbor_index,
    niche_features,
    pseudo_states,
)


def install_paired_objective(model, delta_weight: float, z_weight: float) -> None:
    """Add paired background invariance and counterfactual-delta reconstruction."""
    from scvi import REGISTRY_KEYS
    from scvi.external.contrastivevi._contrastive_data_splitting import (
        ContrastiveDataSplitter,
    )
    from scvi.external.contrastivevi._contrastive_dataloader import (
        ContrastiveDataLoader,
    )

    class PairedDataSplitter(ContrastiveDataSplitter):
        def train_dataloader(self):
            return ContrastiveDataLoader(
                adata_manager=self.adata_manager,
                background_indices=self.background_train_idx,
                target_indices=self.target_train_idx,
                shuffle=False,
                drop_last=self.drop_last,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )

    model._data_splitter_cls = PairedDataSplitter
    module = model.module
    base_loss = module.loss

    def paired_loss(self, tensors, inference, generative, kl_weight=1.0):
        output = base_loss(tensors, inference, generative, kl_weight=kl_weight)
        bg = inference["background"]
        tg = inference["target"]
        z_bg = bg["qz_m"]
        z_tg = tg["qz_m"]
        s_tg = tg["qs_m"]
        if z_bg.ndim > 2:
            z_bg, z_tg, s_tg = z_bg[0], z_tg[0], s_tg[0]
        batch = min(z_bg.shape[0], z_tg.shape[0])
        z_bg = z_bg[:batch]
        z_tg = z_tg[:batch]
        s_tg = s_tg[:batch]

        # The same anchored background code is decoded with and without target s.
        # Their difference therefore cannot be explained by background variation.
        anchor = z_bg.detach()
        latent_control = torch.cat([anchor, torch.zeros_like(s_tg)], dim=-1)
        latent_target = torch.cat([anchor, s_tg], dim=-1)
        pred_control = self.decoder(None, latent_control, None, None)[2]
        pred_target = self.decoder(None, latent_target, None, None)[2]
        true_control = tensors["background"][REGISTRY_KEYS.X_KEY][:batch, : self.n_input]
        true_target = tensors["target"][REGISTRY_KEYS.X_KEY][:batch, : self.n_input]
        delta_true = true_target - true_control
        delta_pred = pred_target - pred_control
        paired_delta = torch.square(delta_pred - delta_true).sum(-1).mean()
        paired_z = torch.square(z_tg - z_bg.detach()).mean()
        output.loss = output.loss + delta_weight * paired_delta + z_weight * paired_z
        output.extra_metrics["paired_delta_mse"] = paired_delta.detach()
        output.extra_metrics["paired_background_mse"] = paired_z.detach()
        return output

    module.loss = types.MethodType(paired_loss, module)


def fit_reference(a, target_mask, args):
    import scvi
    from scvi.external import ContrastiveVI

    scvi.settings.seed = args.seed
    ContrastiveVI.setup_anndata(a, layer=PROTEIN_LAYER, batch_key=args.batch_key)
    model = ContrastiveVI(
        a,
        n_me=0,
        n_hidden=args.hidden,
        n_background_latent=args.n_latent,
        n_salient_latent=args.n_latent,
        use_observed_lib_size=True,
        wasserstein_penalty=args.wasserstein,
        gene_likelihood="normal",
    )
    attach_protein_mse_decoder(model, 2 * args.n_latent, a.n_vars)
    install_paired_objective(model, args.delta_weight, args.z_pair_weight)
    model.train(
        background_indices=np.flatnonzero(~target_mask),
        target_indices=np.flatnonzero(target_mask),
        max_epochs=args.reference_epochs,
        batch_size=args.batch_size,
        train_size=1.0,
        shuffle_set_split=False,
        early_stopping=False,
        gradient_clip_val=1.0,
        datasplitter_kwargs={
            "num_workers": args.num_workers,
            "persistent_workers": args.num_workers > 0,
        },
    )
    z = model.get_latent_representation(a, representation_kind="background")
    s = model.get_latent_representation(a, representation_kind="salient")
    return z.astype(np.float32), s.astype(np.float32)


def knn_ring_aggregate(coords, samples, membership, scale=6, ring_k=12):
    """MENDER rings defined by neighbor rank, invariant to section density."""
    from sklearn.neighbors import NearestNeighbors

    membership = valid_composition(membership)
    samples = np.asarray(samples).astype(str)
    out = np.zeros((len(coords), scale * membership.shape[1]), dtype=np.float32)
    for sample in np.unique(samples):
        rows = np.flatnonzero(samples == sample)
        width = min(scale * ring_k, len(rows))
        indices = NearestNeighbors(n_neighbors=width).fit(coords[rows]).kneighbors(
            return_distance=False
        )
        blocks = []
        for ring in range(scale):
            left = min(ring * ring_k, width)
            right = min((ring + 1) * ring_k, width)
            if right <= left:
                blocks.append(np.zeros((len(rows), membership.shape[1]), np.float32))
            else:
                blocks.append(membership[rows][indices[:, left:right]].sum(1))
        out[rows] = _normalize_log_counts(np.concatenate(blocks, axis=1))
    return out


def balanced_fusion(cell, spatial, mask, dim, zero_outside=False):
    """Preserve cell information while giving the spatial block equal dimension."""
    mask = np.asarray(mask, dtype=bool)
    result = []
    for block in (cell, spatial):
        scaler = StandardScaler().fit(block[mask])
        scaled_fit = scaler.transform(block[mask])
        n_dim = min(dim, scaled_fit.shape[1], max(1, scaled_fit.shape[0] - 1))
        pca = PCA(n_components=n_dim, random_state=0).fit(scaled_fit)
        result.append(pca.transform(scaler.transform(block)).astype(np.float32))
    fused = np.concatenate(result, axis=1)
    if zero_outside:
        fused[~mask] = 0.0
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="bench_be_protein/results_2domain_fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference_epochs", type=int, default=75)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_bg_states", type=int, default=12)
    parser.add_argument("--n_sal_states", type=int, default=8)
    parser.add_argument("--k_niche", type=int, default=20)
    parser.add_argument("--spatial_weight", type=float, default=3.0)
    parser.add_argument("--wasserstein", type=float, default=0.05)
    parser.add_argument("--delta_weight", type=float, default=20.0)
    parser.add_argument("--z_pair_weight", type=float, default=2.0)
    parser.add_argument("--mender_scale", type=int, default=6)
    parser.add_argument("--ring_k", type=int, default=15)
    parser.add_argument("--fusion_dim", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_key", default="section")
    parser.add_argument("--sample_key", default="slice_id")
    args = parser.parse_args()

    import scvi
    from scvi.external import ContrastiveVI

    raw = ad.read_h5ad(args.data)
    a = prepare_protein_adata(raw)
    target = a.obs["condition"].astype(str).to_numpy() == "disease"
    bg_idx, tg_idx = np.flatnonzero(~target), np.flatnonzero(target)
    if len(bg_idx) != len(tg_idx):
        raise ValueError("Paired objective requires equal control and condition sizes")

    with Timer() as timer:
        z_ref, s_ref = fit_reference(a, target, args)
        neighbors = neighbor_index(a, args.k_niche, args.sample_key)
        bg_states, z_scaled = pseudo_states(
            z_ref, np.ones(a.n_obs, bool), args.n_bg_states, args.seed
        )
        sal_states, s_scaled = pseudo_states(
            s_ref, target, args.n_sal_states, args.seed + 1
        )
        bg_comp, bg_act = niche_features(
            z_scaled, bg_states, neighbors, args.n_bg_states
        )
        sal_comp, sal_act = niche_features(
            s_scaled, sal_states, neighbors, args.n_sal_states
        )
        sal_comp[~target], sal_act[~target] = 0.0, 0.0
        targets = np.concatenate([bg_comp, bg_act, sal_comp, sal_act], axis=1)
        augmented = augment_with_targets(a, targets)

        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(
            augmented, layer=PROTEIN_LAYER, batch_key=args.batch_key
        )
        model = ContrastiveVI(
            augmented,
            n_me=targets.shape[1],
            n_hidden=args.hidden,
            n_background_latent=args.n_latent,
            n_salient_latent=args.n_latent,
            use_observed_lib_size=True,
            wasserstein_penalty=args.wasserstein,
            gene_likelihood="normal",
            me_weight=args.spatial_weight,
            me_reduction="mean",
            n_spatial_latent=0,
            niche_context=False,
        )
        attach_protein_mse_decoder(model, 2 * args.n_latent, raw.n_vars)
        install_paired_objective(model, args.delta_weight, args.z_pair_weight)
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
            return torch.stack([
                torch.square(prediction[..., left:right] - truth[..., left:right]).mean(-1)
                for left, right in zip(cuts[:-1], cuts[1:])
            ]).sum(0)

        model.module.me_reconstruction_loss = niche_loss
        model.train(
            background_indices=bg_idx,
            target_indices=tg_idx,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            train_size=1.0,
            shuffle_set_split=False,
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
                    np.concatenate([z[left:left + 8192], s[left:left + 8192]], axis=1),
                    device=device,
                )
                predictions.append(model.module.decoder_me(latent, None).cpu().numpy())
        predictions = np.concatenate(predictions).astype(np.float32)
        bg_membership = predictions[:, :bg_out][:, : args.n_bg_states]
        sal_membership = predictions[:, bg_out:][:, : args.n_sal_states]
        sal_membership[~target] = 0.0
        coords = np.asarray(augmented.obsm["spatial"])
        samples = augmented.obs[args.sample_key].astype(str).to_numpy()
        bg_mender = knn_ring_aggregate(
            coords, samples, bg_membership, args.mender_scale, args.ring_k
        )
        sal_mender = knn_ring_aggregate(
            coords, samples, sal_membership, args.mender_scale, args.ring_k
        )
        spatial = balanced_fusion(
            z, bg_mender, np.ones(len(z), bool), args.fusion_dim
        )
        salient_spatial = balanced_fusion(
            s, sal_mender, target, args.fusion_dim, zero_outside=True
        )

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(
        args.out_root, tag, f"haruka_paired_adaptive_mse_seed{args.seed}"
    )
    save_embeddings(
        out_dir,
        background=z,
        salient=s,
        spatial=spatial,
        salient_spatial=salient_spatial,
        obs_names=augmented.obs_names.to_numpy(),
        method="haruka_paired_adaptive_protein_mse",
        seed=args.seed,
        runtime_s=timer.seconds,
        likelihood="exact summed MSE plus paired counterfactual-delta MSE",
        protein_transform=protein_transform_description(),
        paired_delta_weight=args.delta_weight,
        paired_background_weight=args.z_pair_weight,
        wasserstein_penalty=args.wasserstein,
        spatial_weight=args.spatial_weight,
        mender="density-normalized KNN rings",
        ring_k=args.ring_k,
        mender_scale=args.mender_scale,
        fusion="balanced PCA(cell latent, KNN-MENDER)",
        supervised_labels=False,
        postfit=False,
    )
    np.save(os.path.join(out_dir, "background_composition.npy"), bg_membership)
    np.save(os.path.join(out_dir, "salient_composition.npy"), sal_membership)
    with open(os.path.join(out_dir, "mender_radius_diagnostics.json"), "w") as handle:
        json.dump(
            {
                "radius_mode": "neighbor_rank",
                "target_neighbors_per_ring": args.ring_k,
                "first_ring_neighbors_median": args.ring_k,
                "mender_scale": args.mender_scale,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
