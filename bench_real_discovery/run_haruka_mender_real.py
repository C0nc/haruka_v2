"""Run canonical Haruka-MENDER on an unpaired real contrastive study.

The only training label is binary arm membership.  Perturbation identity, cell
type, niche, and plaque labels are deliberately unavailable to the fit and are
reserved for downstream biological evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for module_root in (ROOT / "bench_merfish", ROOT / "src"):
    sys.path.insert(0, str(module_root))

from common import Timer, save_embeddings  # noqa: E402
from mender_composition import aggregate_mender_compositions  # noqa: E402
from run_haruka_cvi_scviva import (  # noqa: E402
    DualNicheDecoder,
    augment_with_targets,
    load_reference,
    neighbor_index,
    niche_features,
    prep_expression,
    pseudo_states,
)


def optional_key(value: str | None) -> str | None:
    if value is None or value.lower() in {"none", "null", ""}:
        return None
    return value


def validate_batch(obs: pd.DataFrame, batch_key: str | None, condition_key: str) -> None:
    if batch_key is None:
        return
    table = pd.crosstab(obs[batch_key].astype(str), obs[condition_key].astype(str))
    bad = table.index[(table > 0).sum(axis=1) < 2].astype(str).tolist()
    if bad:
        raise ValueError(
            f"batch_key={batch_key!r} is condition-confounded for: {bad[:8]}"
        )


def fit_reference(adata: ad.AnnData, args, batch_key: str | None):
    import scvi
    from scvi.external import ContrastiveVI

    prepared = prep_expression(adata)
    target = (
        prepared.obs[args.condition_key].astype(str).to_numpy()
        == args.target_label
    )
    scvi.settings.seed = args.seed
    ContrastiveVI.setup_anndata(prepared, layer="counts", batch_key=batch_key)
    model = ContrastiveVI(
        prepared,
        n_me=0,
        n_hidden=256,
        n_background_latent=args.n_latent,
        n_salient_latent=args.n_latent,
        use_observed_lib_size=True,
        wasserstein_penalty=0.2,
        zs_independence_penalty=args.zs_independence_penalty,
        batch_injection_mode=args.batch_injection_mode,
        gene_likelihood=args.gene_likelihood,
        n_spatial_latent=0,
        niche_context=False,
    )
    model.train(
        background_indices=np.flatnonzero(~target),
        target_indices=np.flatnonzero(target),
        max_epochs=args.reference_epochs,
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
        prepared, representation_kind="background"
    )
    s = model.get_latent_representation(prepared, representation_kind="salient")
    return z.astype(np.float32), s.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--out-root", default="bench_real_discovery/results_haruka_mender"
    )
    parser.add_argument("--initial-emb-dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-epochs", type=int, default=75)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-latent", type=int, default=32)
    parser.add_argument("--n-bg-states", type=int, default=12)
    parser.add_argument("--n-sal-states", type=int, default=8)
    parser.add_argument("--k-niche", type=int, default=20)
    parser.add_argument("--spatial-weight", type=float, default=10.0)
    parser.add_argument(
        "--zs-independence-penalty", type=float, default=2.0,
        help="Directional z <- s disentanglement weight (Haruka default: 2.0)",
    )
    parser.add_argument(
        "--batch-injection-mode",
        choices=["all", "salient_encoder", "decoder_only"],
        default="salient_encoder",
        help=(
            "Batch routing mode (Haruka default: salient encoder only; "
            "decoder_only matches scVI-style decoder conditioning with "
            "batch-free biological encoders)"
        ),
    )
    parser.add_argument("--mender-scale", type=int, default=6)
    parser.add_argument("--mender-radius", type=float, default=15.0)
    parser.add_argument(
        "--mender-neighbors",
        type=int,
        default=15,
        help="Adapt each graph radius to this many first-ring neighbors; <=0 uses --mender-radius",
    )
    parser.add_argument("--gene-likelihood", choices=["poisson", "normal"], required=True)
    parser.add_argument("--batch-key", default="none")
    parser.add_argument("--sample-key", default="graph_id")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--target-label", default="disease")
    args = parser.parse_args()

    import scvi
    from scvi.external import ContrastiveVI

    torch.set_float32_matmul_precision("high")
    batch_key = optional_key(args.batch_key)
    data_path = Path(args.data).resolve()
    adata = ad.read_h5ad(data_path)
    required = {args.sample_key, args.condition_key}
    if batch_key is not None:
        required.add(batch_key)
    missing = sorted(required.difference(adata.obs.columns))
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")
    if "counts" not in adata.layers:
        raise KeyError("Haruka requires expression in adata.layers['counts']")
    if "spatial" not in adata.obsm:
        raise KeyError("Haruka-MENDER requires adata.obsm['spatial']")
    if args.batch_injection_mode in {"all", "decoder_only"}:
        validate_batch(adata.obs, batch_key, args.condition_key)

    target_mask = (
        adata.obs[args.condition_key].astype(str).to_numpy() == args.target_label
    )
    background_idx = np.flatnonzero(~target_mask)
    target_idx = np.flatnonzero(target_mask)
    if not len(background_idx) or not len(target_idx):
        raise ValueError("ContrastiveVI requires non-empty target and background arms")

    out = Path(args.out_root) / args.study / f"haruka_mender_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    contract = {
        "study": args.study,
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "background_cells": int(len(background_idx)),
        "target_cells": int(len(target_idx)),
        "batch_key": batch_key,
        "sample_key": args.sample_key,
        "gene_likelihood": args.gene_likelihood,
        "training_labels": [args.condition_key],
        "evaluation_only_labels": [
            "perturbation", "guide_name", "cell_type", "niche",
            "plaque_tau", "plaque_abeta", "plaque_any",
        ],
    }
    print(json.dumps(contract, indent=2), flush=True)

    with Timer() as timer:
        if args.initial_emb_dir:
            z_ref, s_ref = load_reference(adata, args.initial_emb_dir)
            reference_source = str(Path(args.initial_emb_dir).resolve())
        else:
            z_ref, s_ref = fit_reference(adata, args, batch_key)
            reference_source = f"internal ContrastiveVI ({args.reference_epochs} epochs)"

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
        sal_comp[~target_mask] = 0.0
        sal_act[~target_mask] = 0.0
        targets = np.concatenate([bg_comp, bg_act, sal_comp, sal_act], axis=1)
        augmented = augment_with_targets(adata, targets)

        scvi.settings.seed = args.seed
        ContrastiveVI.setup_anndata(
            augmented, layer="counts", batch_key=batch_key
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
        batch_injection_mode=args.batch_injection_mode,
            gene_likelihood=args.gene_likelihood,
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
            max_epochs=args.epochs,
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
        prediction = []
        model.module.eval()
        with torch.no_grad():
            for left in range(0, len(z), 8192):
                latent = torch.as_tensor(
                    np.concatenate([z[left:left + 8192], s[left:left + 8192]], axis=1),
                    device=device,
                )
                prediction.append(model.module.decoder_me(latent, None).cpu().numpy())
        prediction = np.concatenate(prediction).astype(np.float32)
        bg_decoder = prediction[:, :bg_out]
        sal_decoder = prediction[:, bg_out:]
        sal_decoder[~target_mask] = 0.0
        adaptive_neighbors = args.mender_neighbors if args.mender_neighbors > 0 else None
        spatial, salient_spatial, radius_diagnostics = aggregate_mender_compositions(
            augmented.obsm["spatial"],
            augmented.obs[args.sample_key].astype(str).to_numpy(),
            bg_decoder[:, :args.n_bg_states],
            sal_decoder[:, :args.n_sal_states],
            scale=args.mender_scale,
            radius=args.mender_radius,
            radius_neighbors=adaptive_neighbors,
            return_diagnostics=True,
        )

    (out / "mender_radius_diagnostics.json").write_text(
        json.dumps(radius_diagnostics, indent=2) + "\n"
    )
    np.save(out / "background_composition.npy", bg_decoder[:, :args.n_bg_states])
    np.save(out / "salient_composition.npy", sal_decoder[:, :args.n_sal_states])

    save_embeddings(
        str(out),
        background=z,
        spatial=spatial,
        salient=s,
        salient_spatial=salient_spatial,
        obs_names=augmented.obs_names.to_numpy(),
        method="haruka_mender_real",
        runtime_s=timer.seconds,
        reference_source=reference_source,
        seed=args.seed,
        epochs=args.epochs,
        n_latent=args.n_latent,
        n_bg_states=args.n_bg_states,
        n_sal_states=args.n_sal_states,
        k_niche=args.k_niche,
        spatial_weight=args.spatial_weight,
        zs_independence_penalty=args.zs_independence_penalty,
        batch_injection_mode=args.batch_injection_mode,
        mender_scale=args.mender_scale,
        mender_radius=args.mender_radius,
        mender_neighbors=adaptive_neighbors,
        mender_radius_mode="cell_local_kth_neighbor",
        gene_likelihood=args.gene_likelihood,
        batch_key=batch_key,
        sample_key=args.sample_key,
        binary_arm_membership_only=True,
        all_samples_joint=True,
        holdout=None,
        supervised_labels=False,
        postfit=False,
        paired_loss=False,
        simvi_components=False,
        gat_components=False,
        cellcharter_components=False,
        spatial_readout=(
            "MENDER soft pseudo-state composition in non-overlapping rings; "
            "cell-local radius calibrated by kth-neighbor distance"
        ),
    )


if __name__ == "__main__":
    main()
