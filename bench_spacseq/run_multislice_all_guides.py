"""Train multi-slice Haruka with all supported perturbations as condition."""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench_merfish"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from batch_invariance import install_four_slot_batch_mmd  # noqa: E402
from common import Timer, save_embeddings  # noqa: E402
from run_haruka import extract_all_cells  # noqa: E402
from run_haruka_ownership import (  # noqa: E402
    BACKGROUND_MODULES,
    freeze_background,
    maximum_background_change,
    snapshot_background,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="bench_spacseq/results_multislice")
    parser.add_argument("--tag", default="all_guides_batchmmd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1_epochs", type=int, default=150)
    parser.add_argument("--stage2_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--k_spatial", type=int, default=8)
    parser.add_argument("--batch_mmd_weight", type=float, default=10.0)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--salient_latent_dim", type=int, default=32)
    parser.add_argument("--n_spatial_latent", type=int, default=8)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--model_n_layers", type=int, default=2)
    parser.add_argument("--n_attention_heads", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=0)
    args = parser.parse_args()

    from haruka_spatial import HarukaSpatial

    adata = ad.read_h5ad(args.data)
    required = {"condition", "perturbation", "slice_id", "graph_id"}
    missing = sorted(required - set(adata.obs))
    if missing:
        raise KeyError(f"missing dataset columns: {missing}")
    targets = sorted(
        set(adata.obs.perturbation.astype(str)) - {"NT"}
    )
    if len(targets) < 2:
        raise ValueError("multi-perturbation training requires at least two targets")
    cross = adata.obs.groupby(
        ["slice_id", "perturbation"], observed=True
    ).size().unstack(fill_value=0)
    if (cross == 0).any().any():
        raise ValueError("every retained perturbation must occur in every slice")

    data_tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(
        args.out_root, data_tag, f"haruka_{args.tag}_seed{args.seed}"
    )
    h = HarukaSpatial(
        adata, cov_gene_num=0, seed=args.seed,
        k_spatial=args.k_spatial, library_key="graph_id",
    )
    h.prepare_spatial_graph()
    h.setup_model(
        seed=args.seed,
        latent_dim=args.latent_dim,
        salient_latent_dim=args.salient_latent_dim,
        n_spatial_latent=args.n_spatial_latent,
        n_hidden=args.n_hidden,
        n_layers=args.model_n_layers,
        n_attention_heads=args.n_attention_heads,
        spatial_attach="both",
        gene_likelihood="poisson",
        me_weight=0.0,
        me_reduction="sum",
        reg_to_use="mi",
        lam_mi=5.0,
        kl_spatial_weight=0.01,
        spatial_encoder="gat",
        condition_key="condition",
        condition_label="disease",
        # Slice is injected into z/s encoders, neighbor re-encoding, and decoder.
        batch_key="slice_id",
        mae_epochs=90,
        wasserstein_penalty=0.0,
        spatial_invariance_penalty=0.0,
        background_mmd_penalty=0.0,
        mask_salient_mean_on_background=False,
        paired_slot_reg_weight=0.0,
        background_representation="private",
        salient_within_penalty=0.0,
        spatial_pair_penalty=1.0,
        salient_prior_components=len(targets),
        salient_assignment_penalty=1.0,
        salient_assignment_temperature=0.5,
    )
    module = h.model.module
    install_four_slot_batch_mmd(module, args.batch_mmd_weight)

    with Timer() as timer:
        module.background_only_training = True
        print(
            f"[stage 1] all NT controls across 3 slices; "
            f"four-slot batch MMD={args.batch_mmd_weight:g}; "
            f"{args.stage1_epochs} epochs",
            flush=True,
        )
        h.train(max_epochs=args.stage1_epochs, batch_size=args.batch_size)

        print(
            f"[stage 2] {len(targets)} perturbation classes; frozen background; "
            f"{args.stage2_epochs} epochs",
            flush=True,
        )
        module.train()
        frozen, freeze_handles = freeze_background(module)
        before = snapshot_background(module)
        module.background_only_training = False
        module.mae_epochs = 0
        h.train(max_epochs=args.stage2_epochs, batch_size=args.batch_size)
        max_change = maximum_background_change(module, before)
        if max_change != 0.0:
            raise RuntimeError(
                f"frozen background state changed by {max_change:.3e}"
            )
        print(f"[verify] frozen background max change={max_change:.1f}")

    reps = extract_all_cells(h, n_layer=args.n_layer)
    save_embeddings(
        out_dir,
        background=reps["background"],
        spatial=reps["spatial"],
        salient=reps["salient"],
        salient_spatial=reps["salient_spatial"],
        obs_names=adata.obs_names.to_numpy(),
        method=f"haruka_{args.tag}",
        single_rep=False,
        runtime_s=timer.seconds,
        targets=targets,
        n_perturbations=len(targets),
        batch_key="slice_id",
        graph_library_key="graph_id",
        encoder_batch_injection=True,
        neighbor_batch_injection=True,
        decoder_batch_injection=True,
        four_slot_batch_mmd_weight=args.batch_mmd_weight,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        control_only_pretraining=True,
        frozen_background_modules=list(BACKGROUND_MODULES),
        frozen_parameter_count=int(frozen),
        frozen_state_max_change=float(max_change),
        salient_prior_components=len(targets),
        latent_dim=args.latent_dim,
        salient_latent_dim=args.salient_latent_dim,
        n_spatial_latent=args.n_spatial_latent,
        n_hidden=args.n_hidden,
        model_n_layers=args.model_n_layers,
        n_attention_heads=args.n_attention_heads,
        k_spatial=args.k_spatial,
        n_layer=args.n_layer,
        note=(
            "all supported perturbations are the target arm; all NT contexts are "
            "background; graphs are isolated by slice x perturbation; slice is "
            "injected as an encoder/neighbor/decoder covariate and removed from "
            "all four posterior slots by pairwise MMD"
        ),
    )


if __name__ == "__main__":
    main()
