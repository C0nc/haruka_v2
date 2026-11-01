"""Two-stage training: joint, then FREEZE the background and decouple the salient pair.

WHY THIS AND NOT ANOTHER PENALTY. Only one slot pair currently carries an explicit
independence constraint -- `I(z; z_sp)` via `lam_mi=5` -- and it is the only pair
that decouples (background -> gt_niche = 0.017). The salient pair has no such term
by default and does not decouple (salient -> gt_sal_domain = 0.721).

Switching `salient_within_penalty` on ALONE does not fix it, and the reason is
structural rather than a matter of weight. When `z` is pushed off the background
domain, that information has nowhere to go but `z_sp`, where it belongs -- the
background pair is a CLOSED system. When `s` is pushed off the salient domain there
is a THIRD slot available, and it goes there: measured, `salient -> gt_sal_domain`
fell 0.721 -> 0.526 while `spatial -> gt_sal_domain` rose 0.738 -> 0.958. The
penalty worked; the redundancy simply relocated.

Freezing the background after a joint warm-up closes that escape route. `z` and
`z_sp` are fixed, so the salient penalty cannot push anything into them and has to
remove the redundancy rather than move it. This is a stronger version of
`spatial_invariance_penalty`, which only DISCOURAGES the escape; freezing
FORECLOSES it.

Stage 1 trains everything with the salient penalty off, so the background pair
reaches its usual quality. Stage 2 freezes `z_encoder`, `gat_mean`, `gat_var`
(and optionally the decoder) and turns the penalty on. The salient encoder, the
salient GAT and the library/dispersion terms stay trainable.

NOTE the decoder is trainable in stage 2 by default. Leaving it free lets the
salient branch re-fit once its input distribution changes; freezing it as well
would additionally stop the decoder compensating for a rescaled `s`. Which is
better has NOT been measured -- `--freeze_decoder` exposes the alternative so the
two can be compared.

    python run_haruka_twostage.py --data data_easy/simv4_g0.8_r0.8_p1_d1_s0.h5ad \
        --out_root results_twostage --stage1_epochs 350 --stage2_epochs 150 \
        --salient_within_penalty 1.0
"""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import numpy as np
import scanpy as sc

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from common import Timer, save_embeddings  # noqa: E402

#: submodules that carry the BACKGROUND decomposition and are frozen in stage 2
BACKGROUND_MODULES = ("z_encoder", "gat_mean", "gat_var")


def freeze(module, names, verbose=True):
    n_frozen = 0
    for nm in names:
        sub = getattr(module, nm, None)
        if sub is None:
            if verbose:
                print(f"   [freeze] {nm}: absent, skipped")
            continue
        for p in sub.parameters():
            p.requires_grad_(False)
            n_frozen += p.numel()
        if verbose:
            print(f"   [freeze] {nm}: frozen")
    return n_frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_root", default="results_twostage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1_epochs", type=int, default=350)
    ap.add_argument("--stage2_epochs", type=int, default=150)
    ap.add_argument("--salient_within_penalty", type=float, default=1.0)
    ap.add_argument("--freeze_decoder", action="store_true")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--k_spatial", type=int, default=8)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--tag", default="twostage")
    args = ap.parse_args()

    from haruka_spatial import HarukaSpatial

    adata = ad.read_h5ad(args.data)
    tag = os.path.basename(args.data).replace(".h5ad", "")
    out_dir = os.path.join(args.out_root, tag, f"haruka_{args.tag}_seed{args.seed}")

    h = HarukaSpatial(adata, cov_gene_num=0, seed=args.seed,
                      k_spatial=args.k_spatial, library_key="slice_id")
    h.prepare_spatial_graph()
    # Stage-1 kwargs mirror `both_sal8_timebatch`, the current best configuration,
    # with the salient penalty OFF so the background pair forms first.
    h.setup_model(
        seed=args.seed, latent_dim=32, salient_latent_dim=8, n_spatial_latent=8,
        n_hidden=256, spatial_attach="both", gene_likelihood="poisson",
        me_weight=0.0, me_reduction="sum", reg_to_use="mi", lam_mi=5.0,
        kl_spatial_weight=0.01, spatial_encoder="gat",
        condition_key="condition", condition_label="disease",
        batch_key="timepoint", mae_epochs=90,
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_within_penalty=0.0,
    )

    with Timer() as t:
        print(f"[stage 1] joint training, {args.stage1_epochs} epochs, "
              f"salient_within_penalty=0", flush=True)
        h.train(max_epochs=args.stage1_epochs, batch_size=args.batch_size)

        mod = h.model.module
        print(f"\n[stage 2] freezing background, {args.stage2_epochs} epochs, "
              f"salient_within_penalty={args.salient_within_penalty}", flush=True)
        names = list(BACKGROUND_MODULES) + (["decoder"] if args.freeze_decoder else [])
        n = freeze(mod, names)
        print(f"   frozen parameters: {n:,}")
        mod.salient_within_penalty = float(args.salient_within_penalty)
        # The MAE corruption schedule is an absolute epoch count and stage 2 restarts
        # the counter; leaving it on would re-corrupt the input for a branch that is
        # now supposed to be refining, not exploring.
        mod.mae_epochs = 0
        h.train(max_epochs=args.stage2_epochs, batch_size=args.batch_size)

    from run_haruka import extract_all_cells
    reps = extract_all_cells(h, n_layer=args.n_layer)
    save_embeddings(
        out_dir,
        background=reps["background"], salient=reps["salient"],
        spatial=reps.get("spatial"), salient_spatial=reps.get("salient_spatial"),
        obs_names=adata.obs_names.to_numpy(), method=f"haruka_{args.tag}",
        single_rep=False, runtime_s=t.seconds, n_layer=args.n_layer,
        stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
        salient_within_penalty=args.salient_within_penalty,
        froze_decoder=bool(args.freeze_decoder), k_spatial=args.k_spatial,
        note="stage 2 freezes z_encoder/gat_mean/gat_var so the salient "
             "independence penalty cannot displace the domain into z_sp",
    )


if __name__ == "__main__":
    main()
