"""One-stage Haruka with direct z-only and s-only composition decoders.

No MENDER, graph aggregation, PCA, reference pretraining, domain supervision, or
post-fit model is used. The saved spatial slots are the decoder outputs directly:
D_bg(z) predicts paired-control neighborhood composition and D_sal(s) predicts
the signed condition-minus-control neighborhood-composition shift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

MERFISH = os.path.join(os.path.dirname(__file__), "..", "bench_merfish")
sys.path.insert(0, MERFISH)

from common import Timer, save_embeddings  # noqa: E402
from run_haruka_cvi_scviva import augment_with_targets, neighbor_index  # noqa: E402
from run_haruka_shared_decoder import annotated_composition  # noqa: E402


class DirectCompositionDecoders(torch.nn.Module):
    """Independent z-only background and s-only salient composition heads."""

    def __init__(self, z_dim: int, s_dim: int, n_types: int, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.s_dim = s_dim

        def head(n_in):
            return torch.nn.Sequential(
                torch.nn.Linear(n_in, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, n_types),
            )

        self.background = head(z_dim)
        self.salient = head(s_dim)

    def forward(self, latent, *_):
        z = latent[..., : self.z_dim]
        s = latent[..., self.z_dim : self.z_dim + self.s_dim]
        return torch.cat([self.background(z), self.salient(s)], dim=-1)


def paired_targets(composition, target_mask):
    """Construct background composition and signed salient shift targets."""
    background_rows = np.flatnonzero(~target_mask)
    target_rows = np.flatnonzero(target_mask)
    if len(background_rows) != len(target_rows):
        raise ValueError("Direct paired targets require equal control/condition cell counts")
    # The simulator preserves coordinate order between the two arms.
    background_target = np.empty_like(composition)
    background_target[background_rows] = composition[background_rows]
    background_target[target_rows] = composition[background_rows]
    salient_target = np.zeros_like(composition)
    salient_target[target_rows] = (
        composition[target_rows] - composition[background_rows]
    )
    return background_target, salient_target


def decode_heads(decoder, z, s, device, n_types, batch_size=8192):
    background, salient = [], []
    decoder.eval()
    with torch.no_grad():
        for left in range(0, len(z), batch_size):
            latent = torch.as_tensor(
                np.concatenate(
                    [z[left : left + batch_size], s[left : left + batch_size]], axis=1
                ),
                device=device,
            )
            prediction = decoder(latent)
            background.append(F.softmax(prediction[:, :n_types], dim=-1).cpu().numpy())
            salient.append(prediction[:, n_types:].cpu().numpy())
    return (
        np.concatenate(background).astype(np.float32),
        np.concatenate(salient).astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="results_direct_decoders")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--k_niche", type=int, default=20)
    parser.add_argument("--spatial_weight", type=float, default=10.0)
    parser.add_argument("--salient_loss_weight", type=float, default=10.0)
    parser.add_argument("--background_wasserstein_penalty", type=float, default=0.0)
    parser.add_argument(
        "--zs_independence_penalty",
        type=float,
        default=2.0,
        help="Directional z <- s disentanglement weight (Haruka default: 2.0)",
    )
    parser.add_argument("--mask_salient_mean_on_background", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_key", default="timepoint")
    parser.add_argument("--sample_key", default="slice_id")
    parser.add_argument("--condition_key", default="condition")
    parser.add_argument("--target_label", default="disease")
    parser.add_argument("--cell_type_key", default="observed_celltype")
    parser.add_argument(
        "--target_mode",
        choices=("paired", "observed"),
        default="paired",
        help="Use exact paired shifts or unpaired direct observed compositions.",
    )
    args = parser.parse_args()

    import scvi
    from scvi.external import ContrastiveVI

    adata = ad.read_h5ad(args.data)
    required = {
        args.batch_key,
        args.sample_key,
        args.condition_key,
        args.cell_type_key,
        "gt_background_subregion",
        "gt_sal_domain",
    }
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"Missing required obs columns: {sorted(missing)}")
    target_mask = (
        adata.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    )
    background_idx = np.flatnonzero(~target_mask)
    target_idx = np.flatnonzero(target_mask)

    labels = adata.obs[args.cell_type_key].astype(str).to_numpy()
    categories = np.unique(labels)
    neighbors = neighbor_index(adata, args.k_niche, args.sample_key)
    observed_composition = annotated_composition(labels, neighbors, categories)
    if args.target_mode == "paired":
        background_target, salient_target = paired_targets(
            observed_composition, target_mask
        )
        targets = np.concatenate([background_target, salient_target], axis=1)
    else:
        # One target per observed cell. The final flag masks salient-head loss
        # to condition cells; no control cell, pair ID, or row order is used.
        targets = np.concatenate(
            [observed_composition, target_mask[:, None].astype(np.float32)], axis=1
        )
    augmented = augment_with_targets(adata, targets)

    with Timer() as timer:
        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(
            augmented, layer="counts", batch_key=args.batch_key
        )
        model = ContrastiveVI(
            augmented,
            n_me=targets.shape[1],
            n_hidden=args.n_hidden,
            n_background_latent=args.n_latent,
            n_salient_latent=args.n_latent,
            use_observed_lib_size=True,
            wasserstein_penalty=0.2,
            background_wasserstein_penalty=args.background_wasserstein_penalty,
            zs_independence_penalty=args.zs_independence_penalty,
            mask_salient_mean_on_background=args.mask_salient_mean_on_background,
            gene_likelihood="poisson",
            me_weight=args.spatial_weight,
            me_reduction="mean",
            n_spatial_latent=0,
            niche_context=False,
        )
        model.module.decoder_me = DirectCompositionDecoders(
            args.n_latent, args.n_latent, len(categories), args.n_hidden
        )
        n_types = len(categories)

        def direct_composition_loss(x, index, prediction, reduction="mean"):
            truth = x[..., index:]
            if args.target_mode == "observed":
                composition_truth = truth[..., :n_types]
                is_target = truth[..., n_types]
                bg_loss = -(
                    composition_truth
                    * F.log_softmax(prediction[..., :n_types], dim=-1)
                ).sum(-1)
                sal_loss = -(
                    composition_truth
                    * F.log_softmax(prediction[..., n_types:], dim=-1)
                ).sum(-1)
                return bg_loss + args.salient_loss_weight * is_target * sal_loss
            bg_truth = truth[..., :n_types]
            sal_truth = truth[..., n_types:]
            bg_loss = -(
                bg_truth * F.log_softmax(prediction[..., :n_types], dim=-1)
            ).sum(-1)
            sal_loss = (
                prediction[..., n_types:] - sal_truth
            ).square().mean(-1)
            return bg_loss + args.salient_loss_weight * sal_loss

        model.module.me_reconstruction_loss = direct_composition_loss
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
        background_output, salient_output = decode_heads(
            model.module.decoder_me, z, s, device, n_types
        )
        if args.target_mode == "observed":
            shifted = salient_output - salient_output.max(axis=1, keepdims=True)
            salient_output = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        salient_output[~target_mask] = 0.0

    background_truth = augmented.obs["gt_background_subregion"].astype(str).to_numpy()
    salient_truth = augmented.obs["gt_sal_domain"].astype(str).to_numpy()[target_mask]
    background_clusters = KMeans(
        n_clusters=len(np.unique(background_truth)), random_state=args.seed, n_init=20
    ).fit_predict(background_output)
    salient_clusters = KMeans(
        n_clusters=len(np.unique(salient_truth)), random_state=args.seed, n_init=20
    ).fit_predict(salient_output[target_mask])
    scores = {
        "background_subregion": {
            "ari": float(adjusted_rand_score(background_truth, background_clusters)),
            "n_cells": int(len(background_truth)),
            "n_clusters": int(len(np.unique(background_truth))),
        },
        "salient_domain": {
            "ari": float(adjusted_rand_score(salient_truth, salient_clusters)),
            "n_cells": int(target_mask.sum()),
            "n_clusters": int(len(np.unique(salient_truth))),
        },
    }

    tag = os.path.basename(args.data).removesuffix(".h5ad")
    run_name = (
        f"haruka_direct_observed_seed{args.seed}"
        if args.target_mode == "observed"
        else f"haruka_direct_decoders_seed{args.seed}"
    )
    out_dir = os.path.join(args.out_root, tag, run_name)
    save_embeddings(
        out_dir,
        background=z,
        salient=s,
        spatial=background_output,
        salient_spatial=salient_output,
        obs_names=augmented.obs_names.to_numpy(),
        method=(
            "haruka_direct_unpaired_observed_composition"
            if args.target_mode == "observed"
            else "haruka_direct_z_s_composition_decoders"
        ),
        seed=args.seed,
        runtime_s=timer.seconds,
        max_epochs=args.max_epochs,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        k_niche=args.k_niche,
        spatial_weight=args.spatial_weight,
        salient_loss_weight=args.salient_loss_weight,
        background_wasserstein_penalty=args.background_wasserstein_penalty,
        zs_independence_penalty=args.zs_independence_penalty,
        mask_salient_mean_on_background=args.mask_salient_mean_on_background,
        cell_type_categories=categories.tolist(),
        background_representation="softmax(D_bg(z))",
        salient_representation=(
            "softmax(D_sal(s)), condition observed composition"
            if args.target_mode == "observed"
            else "D_sal(s), signed composition shift"
        ),
        target_mode=args.target_mode,
        paired_target_construction=args.target_mode == "paired",
        clustering="raw decoder output -> KMeans; no scaling or PCA",
        mender=False,
        pca=False,
        one_stage=True,
        domain_supervision=False,
        cell_type_composition_supervision=True,
        postfit=False,
    )
    np.save(os.path.join(out_dir, "background_decoder_composition.npy"), background_output)
    np.save(os.path.join(out_dir, "salient_decoder_composition_shift.npy"), salient_output)
    result = {
        "method": (
            "haruka_direct_unpaired_observed_composition"
            if args.target_mode == "observed"
            else "haruka_direct_z_s_composition_decoders"
        ),
        "protocol": "raw decoder outputs -> KMeans (no MENDER, no PCA)",
        "section": str(augmented.obs[args.batch_key].iloc[0]),
        "scores": scores,
        "paired_target_construction": args.target_mode == "paired",
    }
    with open(os.path.join(out_dir, "direct_decoder_kmeans_ari.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
