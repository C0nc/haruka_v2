"""Control-pretrained latent ownership schedule for the aligned 2x2 benchmark.

Stage 1 exposes the model only to control tensors and optimizes the background
reconstruction through z/z_sp. Stage 2 either freezes the background encoders
(including their BatchNorm state) or adapts them with distribution alignment, then
trains the salient pair on the full contrastive loss. The decoder remains trainable
so it can express nonlinear background-salient fusion.
"""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from common import Timer, save_embeddings  # noqa: E402

BACKGROUND_MODULES = ("z_encoder", "gat_mean", "gat_var")


def freeze_background(module):
    """Freeze background parameters and keep normalization buffers immutable."""
    handles = []
    frozen = 0

    def force_eval(submodule, _inputs):
        submodule.eval()

    for name in BACKGROUND_MODULES:
        submodule = getattr(module, name, None)
        if submodule is None:
            print(f"   [freeze] {name}: absent, skipped", flush=True)
            continue
        submodule.eval()
        handles.append(submodule.register_forward_pre_hook(force_eval))
        for parameter in submodule.parameters():
            parameter.requires_grad_(False)
            frozen += parameter.numel()
        print(f"   [freeze] {name}: parameters and running state frozen", flush=True)
    return frozen, handles


def snapshot_background(module):
    snapshot = {}
    for name in BACKGROUND_MODULES:
        submodule = getattr(module, name, None)
        if submodule is not None:
            for key, value in submodule.state_dict().items():
                snapshot[f"{name}.{key}"] = value.detach().cpu().clone()
    return snapshot


def maximum_background_change(module, before):
    after = snapshot_background(module)
    return max(
        (after[key] - value).abs().max().item()
        for key, value in before.items()
    ) if before else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_root", default="results_ownership_fusion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1_epochs", type=int, default=250)
    parser.add_argument("--stage2_epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--k_spatial", type=int, default=8)
    parser.add_argument("--n_spatial_latent", type=int, default=8)
    parser.add_argument("--n_attention_heads", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--salient_within_penalty", type=float, default=1.0)
    parser.add_argument("--spatial_pair_penalty", type=float, default=0.0)
    parser.add_argument("--paired_slot_reg_weight", type=float, default=1.0)
    parser.add_argument("--spatial_invariance_penalty", type=float, default=0.5)
    parser.add_argument("--background_mmd_penalty", type=float, default=0.0)
    parser.add_argument("--salient_prior_components", type=int, default=0)
    parser.add_argument("--salient_assignment_penalty", type=float, default=0.0)
    parser.add_argument("--salient_assignment_temperature", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--salient_latent_dim", type=int, default=32)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--model_n_layers", type=int, default=1)
    parser.add_argument(
        "--stage2_background_mode", choices=("frozen", "adaptive"), default="frozen",
        help="'adaptive' lets z/z_sp respond to the nonlinear MMD condition-matching "
             "loss; 'frozen' reproduces the strict ownership experiment.",
    )
    parser.add_argument("--tag", default="ownership_fusion")
    args = parser.parse_args()

    from haruka_spatial import HarukaSpatial
    from run_haruka import extract_all_cells

    adata = ad.read_h5ad(args.data)
    data_tag = os.path.basename(args.data).removesuffix(".h5ad")
    out_dir = os.path.join(
        args.out_root, data_tag, f"haruka_{args.tag}_seed{args.seed}"
    )

    h = HarukaSpatial(
        adata, cov_gene_num=0, seed=args.seed,
        k_spatial=args.k_spatial, library_key="slice_id",
    )
    h.prepare_spatial_graph()
    h.setup_model(
        seed=args.seed, latent_dim=args.latent_dim,
        salient_latent_dim=args.salient_latent_dim,
        n_spatial_latent=args.n_spatial_latent, n_hidden=args.n_hidden,
        n_layers=args.model_n_layers, n_attention_heads=args.n_attention_heads,
        spatial_attach="both",
        gene_likelihood="poisson", me_weight=0.0, me_reduction="sum",
        reg_to_use="mi", lam_mi=5.0, kl_spatial_weight=0.01,
        spatial_encoder="gat", condition_key="condition",
        condition_label="disease", batch_key="timepoint", mae_epochs=90,
        wasserstein_penalty=0.0,
        spatial_invariance_penalty=args.spatial_invariance_penalty,
        background_mmd_penalty=args.background_mmd_penalty,
        mask_salient_mean_on_background=False,
        paired_slot_reg_weight=args.paired_slot_reg_weight,
        paired_slot_reg_start_epoch=0,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_within_penalty=args.salient_within_penalty,
        spatial_pair_penalty=args.spatial_pair_penalty,
        salient_prior_components=args.salient_prior_components,
        salient_assignment_penalty=args.salient_assignment_penalty,
        salient_assignment_temperature=args.salient_assignment_temperature,
    )

    with Timer() as timer:
        module = h.model.module
        module.background_only_training = True
        print(
            f"[stage 1] controls only: background ownership, "
            f"{args.stage1_epochs} epochs",
            flush=True,
        )
        h.train(max_epochs=args.stage1_epochs, batch_size=args.batch_size)

        print(
            f"\n[stage 2] {args.stage2_background_mode} background: "
            f"salient residual/fusion, "
            f"{args.stage2_epochs} epochs",
            flush=True,
        )
        # A completed Lightning fit leaves submodules in evaluation mode. Stage 2
        # is a new fit, so explicitly restore training mode before selectively
        # locking only the background modules in the frozen variant.
        module.train()
        if args.stage2_background_mode == "frozen":
            frozen, _freeze_handles = freeze_background(module)
            before = snapshot_background(module)
        else:
            frozen, _freeze_handles, before = 0, [], None
            print(
                f"   [adaptive] background encoders train with MMD weight "
                f"{args.background_mmd_penalty:g}", flush=True,
            )
        module.background_only_training = False
        module.mae_epochs = 0
        module.paired_slot_reg_start_epoch = 0
        module.paired_slot_reg_ramp_epochs = 10
        h.train(max_epochs=args.stage2_epochs, batch_size=args.batch_size)

        if before is not None:
            max_change = maximum_background_change(module, before)
            if max_change != 0.0:
                raise RuntimeError(
                    "background ownership violation: frozen state changed by "
                    f"{max_change:.3e}"
                )
            print(f"   frozen background state verified (max change={max_change:.1f})")
        else:
            max_change = None

    reps = extract_all_cells(h, n_layer=args.n_layer)
    save_embeddings(
        out_dir,
        background=reps["background"], salient=reps["salient"],
        spatial=reps.get("spatial"),
        salient_spatial=reps.get("salient_spatial"),
        obs_names=adata.obs_names.to_numpy(), method=f"haruka_{args.tag}",
        single_rep=False, runtime_s=timer.seconds, n_layer=args.n_layer,
        stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
        control_only_pretraining=True,
        treated_rows_seen_in_stage1_encoder=False,
        frozen_background_modules=(
            list(BACKGROUND_MODULES)
            if args.stage2_background_mode == "frozen"
            else []
        ),
        frozen_parameter_count=int(frozen),
        frozen_state_max_change=(float(max_change) if max_change is not None else None),
        stage2_background_mode=args.stage2_background_mode,
        background_mmd_penalty=args.background_mmd_penalty,
        decoder_trainable_in_stage2=True,
        salient_within_penalty=args.salient_within_penalty,
        spatial_pair_penalty=args.spatial_pair_penalty,
        paired_slot_reg_weight=args.paired_slot_reg_weight,
        spatial_invariance_penalty=args.spatial_invariance_penalty,
        salient_prior_components=args.salient_prior_components,
        salient_assignment_penalty=args.salient_assignment_penalty,
        salient_assignment_temperature=args.salient_assignment_temperature,
        latent_dim=args.latent_dim,
        salient_latent_dim=args.salient_latent_dim,
        n_hidden=args.n_hidden, model_n_layers=args.model_n_layers,
        n_spatial_latent=args.n_spatial_latent,
        n_attention_heads=args.n_attention_heads,
        k_spatial=args.k_spatial,
        note=(
            "controls establish z/z_sp ownership; "
            + (
                "frozen background encoders force"
                if args.stage2_background_mode == "frozen"
                else "MMD-aligned adaptive background encoders direct"
            )
            + " the treated reconstruction residual into s/s_sp while the "
              "decoder learns nonlinear fusion"
        ),
    )


if __name__ == "__main__":
    main()
