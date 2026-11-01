"""Haruka v2 spatial (conda env `haruka2`: cellcharter env + the contrastivevi patch).

Variants, matching the README's own naming so the numbers here are comparable to
the ones reported on the synthetic lattice:

    vanilla     n_spatial_latent=0                            two latents, no space
    background  n_spatial_latent=16, spatial_attach=background the recommended config
    both        n_spatial_latent=16, spatial_attach=both       the 2x2
    salient     n_spatial_latent=16, spatial_attach=salient
    niche       niche_context=True                             the set-attention design

ONE DELIBERATE DEPARTURE FROM `HarukaSpatial.extract_rep`. That method extracts
representations for the TARGET ARM ONLY (`output_adata = input_adata[target_indices]`).
That is enough to cluster the lesion, but it makes the background latent's most
important property untestable: whether condition has leaked into it. You cannot
measure `ARI(background, condition)` on a set of cells that are all one condition.
So representations are pulled for ALL cells here, and the cellcharter neighbourhood
aggregation is applied to the full object -- the same two calls `extract_rep` makes,
on a superset of the rows. Clustering is left to the scorer so that every method
goes through one identical PCA-50 + full-covariance GMM.

The benchmark runner is pure GAT. Its default is the validated low-width 2x2 GAT
with matched timepoint batching (`both_sal8_timebatch`). Legacy COVET code remains
importable for old library clients, but it is not exposed here and its dependency
stack is not loaded.

    python run_haruka.py --data data/sim_d0.6_s0.h5ad --variants background both vanilla
"""

from __future__ import annotations

import argparse
import os
import sys

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from common import Timer, save_embeddings  # noqa: E402
from orthogonalize_haruka import residualize  # noqa: E402

VARIANTS = {
    "vanilla": dict(n_spatial_latent=0),
    "background": dict(n_spatial_latent=16, spatial_attach="background"),
    "both": dict(n_spatial_latent=16, spatial_attach="both"),
    "salient": dict(n_spatial_latent=16, spatial_attach="salient"),
    "niche": dict(niche_context=True),

    # --- SIMVI-faithful loss ablations on the 2x2 --------------------------------
    # SIMVI's spatial branch is regularised by exactly two terms: a down-weighted KL
    # on the spatial latent (`kl_gatweight`, here `kl_spatial_weight=0.01`) and a
    # mutual-information penalty (`lam_mi`). haruka adds two more that SIMVI has no
    # counterpart for, and both are candidates for the latent inversion that makes
    # `both` the only method with a negative assignment margin:
    #
    #   spatial_invariance_penalty=0.5  the code calls it "the symmetry breaker that
    #                                   stops z_sp from swallowing the lesion niche
    #                                   and starving s_sp"
    #   me_weight (COVET head)          a second reconstruction target on the
    #                                   scenvi COVET block; SIMVI has one decoder
    #
    # NOTE the GAT input needs no change to match SIMVI: `_spatial_pool` already
    # feeds it `encoder(log1p(neighbour counts))[:, -n_spatial:]`, i.e. the encoder
    # posterior mean, which is exactly SIMVI's `q_m[:, -n_spatial:]`. Raw expression
    # reaches the GAT nowhere; only the (inactive) `setattn` path reads it directly.
    "both_simviloss": dict(n_spatial_latent=16, spatial_attach="both",
                           spatial_invariance_penalty=0.0),
    "both_simviloss_nome": dict(n_spatial_latent=16, spatial_attach="both",
                                spatial_invariance_penalty=0.0, me_weight=0.0),

    # --- FIXES for the `both` latent role-swap ----------------------------------
    # DIAGNOSIS. Under `background` only `z` has a spatial partner, so encoding
    # smooth high-variance structure (anatomy) in `z` is strictly cheaper than in
    # `s`: the decoder gets `z` AND its denoised copy `z_sp`. That structural
    # asymmetry is what assigns anatomy to the background latent.
    #
    # `both` gives `s` its own partner `s_sp`, so the two branches become
    # interchangeable to the reconstruction loss and the tie-break vanishes.
    # Measured: R^2(s -> arch) 0.603 vs R^2(z -> arch) 0.090, with `s_sp` 87%
    # predictable from the neighbourhood mean of `s`.
    #
    # The only thing that SHOULD break the tie on biological grounds is that
    # anatomy exists in BOTH arms, so a properly condition-specific `s` could not
    # carry it. That constraint is not holding: ||s|| on control cells is 2.0-2.7
    # against 3.0-4.0 on treated -- roughly a 25% shrink, not a zeroing. The
    # Wasserstein term runs at its default 0.2 and is simply too weak.
    #
    # Two ways to restore it, both leaving the spatial branch untouched:
    "both_wass5": dict(n_spatial_latent=16, spatial_attach="both",
                       wasserstein_penalty=5.0),
    "both_maskmean": dict(n_spatial_latent=16, spatial_attach="both",
                          mask_salient_mean_on_background=True),
    "background_maskmean": dict(n_spatial_latent=16, spatial_attach="background",
                                mask_salient_mean_on_background=True),

    # --- MECHANISM ABLATIONS for the anatomy leak into `salient` -----------------
    # The leak appears the moment ANY spatial branch exists -- dependence(s -> arch)
    # goes 0.214 (vanilla) -> 0.991 (background) -> 0.998 (both) -- even though
    # `spatial_attach="background"` never touches the salient encoder in the forward
    # pass. So the cause is in the learning dynamics.
    #
    # `_independence_penalty` is `_pair_penalty(z, z_sp.detach())`: it requires the
    # BACKGROUND intrinsic latent to be independent of `z_sp`, which is pooled
    # neighbours' z, i.e. SMOOTHED ANATOMY. Anatomy is smooth, so the penalty
    # squeezes anatomy out of `z` -- and the decoder still has to reconstruct it.
    # `z` is the only latent under that constraint; `s` is unconstrained apart from
    # a Wasserstein term measured at ~25% shrinkage on controls. The salient latent
    # therefore becomes the unconstrained reservoir for exactly what the penalty
    # displaces.
    #
    # `bg_nomi` removes the squeeze; `bg_wass5` strengthens the block on `s`.
    # If the account holds, both should cut dependence(s -> arch) sharply.
    "bg_nomi": dict(n_spatial_latent=16, spatial_attach="background", lam_mi=0.0),
    "bg_wass5": dict(n_spatial_latent=16, spatial_attach="background",
                     wasserstein_penalty=5.0),

    # The 2x2 with the leak driver removed. `lam_mi=0` cut dependence(s -> arch)
    # from 0.991 to 0.229 on `background`; this asks whether it also rescues the
    # configuration that actually has a salient spatial latent. Until the kwarg
    # merge was fixed, `lam_mi` could not be overridden at all, so this
    # combination had never been runnable.
    "both_nomi": dict(n_spatial_latent=16, spatial_attach="both", lam_mi=0.0),

    # The configuration the v3 design specifies: SIMVI's squeeze applied to BOTH
    # intrinsic latents (I(z;z_sp) via lam_mi, I(s;s_sp) via salient_within_penalty)
    # plus a contrastive constraint that actually bites.
    "both_sym": dict(n_spatial_latent=16, spatial_attach="both",
                     salient_within_penalty=5.0),
    "both_sym_mask": dict(n_spatial_latent=16, spatial_attach="both",
                          salient_within_penalty=5.0,
                          mask_salient_mean_on_background=True),

    # --- OPTIMISATION SWEEP on the v3 (representable) benchmark -----------------
    # Baseline `both` leaves headroom in two places: sal->state 0.889 (vs 0.929 for
    # `background`) and leak 0.723. The spatial quadrants are already saturated
    # (0.996 / 0.995, at or above the isolated-GAT ceilings), so there is nothing
    # to win there.
    #
    # PRETRAINING RATIO. `mae_epochs=None` auto-resolves to 0.2*max_epochs = 30/150.
    # SIMVI runs 75/100 -- nearly 4x our ratio. The phase randomly permutes a
    # `permutation_rate` fraction of genes across cells, which breaks the cell-gene
    # association so the spatial branch cannot recover a cell's OWN expression from
    # its own row and pass it off as neighbour-induced. More of it should sharpen
    # exactly the intrinsic/spatial split we are measuring.
    "both_mae75": dict(n_spatial_latent=16, spatial_attach="both", mae_epochs=112),
    "both_mae50": dict(n_spatial_latent=16, spatial_attach="both", mae_epochs=75),
    # attack the leak directly with the hard contrastive zeroing
    "both_mask": dict(n_spatial_latent=16, spatial_attach="both",
                      mask_salient_mean_on_background=True),
    # wider spatial latents: 16 dims for a 3-class domain may be fine, but the
    # salient intrinsic is only 4-dim in the generator and s is 32 -- check capacity
    "both_wide": dict(n_spatial_latent=32, spatial_attach="both"),

    # --- ITERATION 2: refine around the mae=75 optimum, and combine ---------------
    # Iteration 1 found pretraining NON-MONOTONE: mae 30 -> sal>state 0.889,
    # mae 75 -> 0.956 (best), mae 112 -> 0.954 but leak 0.723 -> 0.799. So SIMVI's
    # own 75% ratio overshoots on this data and ~50% is the optimum. Refine around it.
    #
    # Width and pretraining moved DIFFERENT metrics -- mae the diagonal, width the
    # leak (0.723 -> 0.589 with no diagonal gain) -- so the combination is the
    # obvious next candidate rather than either alone.
    "both_mae45": dict(n_spatial_latent=16, spatial_attach="both", mae_epochs=45),
    "both_mae60": dict(n_spatial_latent=16, spatial_attach="both", mae_epochs=60),
    "both_mae90": dict(n_spatial_latent=16, spatial_attach="both", mae_epochs=90),
    "both_mae75_wide": dict(n_spatial_latent=32, spatial_attach="both", mae_epochs=75),
    # permutation_rate is the other knob of the same phase: how much of the gene
    # panel is shuffled per step. Never varied from its 0.5 default.
    "both_mae75_perm8": dict(n_spatial_latent=16, spatial_attach="both",
                             mae_epochs=75, permutation_rate=0.8),

    # --- ITERATION 3: what actually drives the anatomy leak ----------------------
    # Three candidates were proposed: Wasserstein, hidden capacity, symmetry.
    # Two are already answered and BOTH went the wrong way --
    #   symmetry  I(s;s_sp)          leak 0.723 -> 0.910
    #   hard Wasserstein mask@1000   leak 0.723 -> 0.922
    # while the only thing that ever reduced it was MORE capacity elsewhere
    # (n_spatial_latent 16->32, leak -> 0.589). That points at capacity, not
    # constraint -- and specifically at `s` being over-provisioned: it gets 32 dims
    # for a BINARY factor that the generator draws in 4.
    # SALIENT CAPACITY. Hypothesis: the leak is an over-provisioned salient latent
    # -- 32 dims to carry a BINARY state, so there is room left over for anatomy.
    #
    # These cannot be a clean one-factor sweep, because the module requires
    # `n_spatial_latent <= min(background, salient)` (the spatial branch reads the
    # trailing dims of the salient latent). Shrinking salient to 8 therefore forces
    # spatial from 16 down to 8 as well, confounding "less salient capacity" with
    # "less spatial capacity". `both_sp8` is the control that isolates them: it
    # moves n_spatial_latent to 8 while leaving salient at the default 32, so
    #     both_sal8 vs both_sp8  = the effect of salient width alone
    #     both_sp8  vs both_mae60 = the effect of spatial width alone
    # All three sit on mae_epochs=60, the best configuration so far, so each
    # differs from that reference by one lever.
    "both_sal8": dict(n_spatial_latent=8, spatial_attach="both",
                      mae_epochs=60, salient_latent_dim=8),
    # Dataset-contract diagnostic: slice_id is condition-confounded in the 2x2
    # simulator (T1_control and T1_disease are distinct slices). Supplying it as
    # a decoder batch covariate lets the model explain the entire treatment arm as
    # batch. timepoint is the actual matched-pair batch: every level contains both
    # conditions, matching CINEMA-OT's per-timepoint transport scope.
    # Current raw-only default. Stage 1 uses MAE for 90 epochs; stage 2 ramps a
    # differentiable ridge-predictability penalty over 10 epochs. The public
    # background representation excludes the eight internal GAT-carrier dimensions.
    # A controlled capacity screen selects n_hidden=256 without widening the
    # latents. Across explicit model seeds 0/1/2 its raw assignment margin is
    # 0.546 +/- 0.024 versus raw CINEMA-OT 0.492 on the fixed v5 dataset.
    "both_sal8_timebatch": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
    ),
    # Matched MAE ablation for the corrected pure-GAT benchmark. Keep these as
    # explicit integers: mae_epochs=None is the automatic 20% schedule, not a
    # disabled control, so only mae0 is a true no-corruption comparison.
    # --- LONGER TRAINING on the identifiable (data_easy) dataset ----------------
    # The staged schedule is in ABSOLUTE epochs (mae 90, paired_slot_reg start 90
    # of max_epochs=150), so raising max_epochs alone changes the balance: the
    # penalty would run for 210 of 300 epochs instead of 50 of 150. `_fixed` keeps
    # the original absolute schedule, `_sched` rescales it to the same fractions
    # (60% MAE, penalty on for the last third), so the two are separable.
    #
    # This is also the direct test of whether paired_slot_reg cures the degradation
    # measured before it existed: on the stress dataset salient_spatial fell 0.194
    # -> 0.068 going 150 -> 400 epochs, because nothing anchored the branch to its
    # own factor. If longer training now HELPS, the regulariser is doing that job.
    "both_sal8_timebatch_ep300_fixed": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
    ),
    "both_sal8_timebatch_ep300_sched": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=180,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=180,
        paired_slot_reg_ramp_epochs=20, background_representation="private",
    ),
    # SALIENT-QUADRANT TARGET. The measured leak on data_easy is that the two
    # salient slots carry each other: salient -> gt_sal_domain 0.721 and
    # salient_spatial -> gt_sal_intrinsic 0.585, against a structural floor of
    # ~0.14. paired_slot_reg does not touch that pair -- it constrains slots
    # against the SHARED spatial slot. `salient_within_penalty` is I(s; s_sp),
    # exactly the entangled pair, and is off by default.
    "both_sal8_timebatch_ep300_swp": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=180,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=180,
        paired_slot_reg_ramp_epochs=20, background_representation="private",
        salient_within_penalty=1.0,
    ),
    # 150 -> 300 epochs gained +0.042 margin and +0.070 on the salient diagonal, so
    # push further. `_fixed` keeps the absolute schedule (penalty on from epoch 90,
    # i.e. most of training); `_sched` rescales to the same 60%/last-third fractions.
    "both_sal8_timebatch_ep500_fixed": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
    ),
    "both_sal8_timebatch_ep500_sched": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=300,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=300,
        paired_slot_reg_ramp_epochs=30, background_representation="private",
    ),
    # MIXTURE PRIOR on the salient latent. The default N(0,I) prior is
    # structurally wrong for this factor: the generative salient state is DISCRETE
    # (2 values), and a unimodal Gaussian prior dissolves that structure. Measured
    # on data_easy -- state direction = 36.5% of salient variance and linearly
    # readable at 0.813, yet Cohen d 1.16 vs the ground truth's 1.92 and GMM ARI
    # 0.001 against a ceiling of 0.999. Ruled out first: the state/domain coupling
    # (weakening it on v5 left ARI at 0.001) and domain contamination
    # (residualising removed state along with domain, 0.813 -> 0.483).
    #
    # k=2 matches the truth; k=4 is the over-specified robustness check, since in
    # real data the number of salient states is unknown.
    "both_sal8_timebatch_sprior2": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_prior_components=2,
    ),
    "both_sal8_timebatch_sprior4": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_prior_components=4,
    ),
    # THE PAIRED CONSTRAINT. Only ONE slot pair currently has an explicit
    # independence penalty -- z vs z_sp via lam_mi=5 -- and it is the only pair that
    # decouples (background -> gt_niche = 0.017). The salient pair has
    # salient_within_penalty=0.0 by default and does not decouple
    # (salient -> gt_sal_domain = 0.721), despite identical generative structure.
    #
    # Turning that penalty on ALONE fails: the salient domain leaves `s` (0.721 ->
    # 0.526) and lands in a THIRD slot, z_sp (sp -> gt_sal_domain 0.738 -> 0.958).
    # The background pair has no such escape route, which is why it works. So the
    # salient penalty needs `spatial_invariance_penalty` raised at the same time --
    # z_sp is meant to be condition-invariant, and the salient domain exists only in
    # the treated arm, so a strong invariance term leaves it nowhere to go but s_sp.
    # Its 0.5 default is evidently too weak.
    "both_sal8_timebatch_swp_inv2": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_within_penalty=1.0, spatial_invariance_penalty=2.0,
    ),
    "both_sal8_timebatch_swp_inv5": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_within_penalty=1.0, spatial_invariance_penalty=5.0,
    ),
    # PRIVATE SALIENT CAPACITY. `_spatial_pool` reads `attach_mean[:, -n_spatial:]`,
    # so the GAT sees only the LAST n_spatial dims of a latent and the rest stays
    # private. With latent_dim=32 / n_spatial_latent=8 the background keeps 24
    # private dims; with salient_latent_dim=8 / n_spatial_latent=8 the salient keeps
    # ZERO -- the whole salient latent feeds its own GAT, so `s_sp` is a spatial
    # smoothing of `s` itself and the two cannot decouple by construction.
    #
    # That is the real asymmetry between the arms: not the generative process (a
    # fully symmetric additive K=4 dataset still leaked at 0.995) and not only the
    # missing penalty. These give the salient arm the same 24 private dims the
    # background has.
    "both_sal32_priv24": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=32, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
    ),
    "both_sal16_priv8": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=16, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
    ),
    # FULLY SYMMETRIC ARMS. Every structural difference between the two pairs
    # removed at once:
    #                        background            salient (before -> now)
    #   latent width         32                    8  -> 32
    #   dims feeding its GAT last 8                last 8 (unchanged)
    #   PRIVATE dims         24                    0  -> 24
    #   independence penalty lam_mi = 5.0          0.0 -> 5.0
    #
    # With zero private dims the salient GAT pooled the ENTIRE salient latent, so
    # `s_sp` was a spatial smoothing of `s` itself and no penalty could separate
    # them -- it could only displace the shared content into a third slot, which is
    # exactly what was measured (sp -> gt_sal_domain 0.337 -> 0.989).
    # The salient multiplier sits inside a term that is multiplied by lam_mi.
    # A value of 1.0 therefore gives both pairs effective weight 5.0; the old
    # value 5.0 accidentally weighted the salient pair 25.0.
    "both_bg_aligned": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        latent_dim=32, salient_latent_dim=32, n_hidden=256,
        batch_key="timepoint", lam_mi=5.0,
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_within_penalty=1.0,
    ),
    # Backward-compatible name for scripts requesting the intended symmetry.
    # Older result metadata still records the historical over-weighted value.
    "both_fully_symmetric": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        latent_dim=32, salient_latent_dim=32, n_hidden=256,
        batch_key="timepoint", lam_mi=5.0,
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
        salient_within_penalty=1.0,
    ),
    # Clean 2x2 objective. Intrinsic salient handling is exactly ContrastiveVI:
    # q(s|control) is evaluated only for W2(q, delta_0), while decoded control s is
    # zero. Its spatial mirror is condition-only: the salient GAT is not evaluated
    # for control centres and decoded control s_sp is zero. Apart from the standard
    # latent/spatial KLs, the only disentanglement terms are the symmetric SIMVI
    # within-pair penalties I(z;z_sp) and I(s;s_sp). No cross-spatial penalty,
    # invariance term, paired ridge term, mixture prior, or post-fit cleanup.
    "both_clean_cvi_symmetric": dict(
        n_spatial_latent=8,
        spatial_attach="both",
        mae_epochs=90,
        latent_dim=32,
        salient_latent_dim=32,
        n_hidden=256,
        batch_key="timepoint",
        wasserstein_penalty=0.2,
        lam_mi=5.0,
        salient_within_penalty=1.0,
        spatial_pair_penalty=0.0,
        spatial_invariance_penalty=0.0,
        background_mmd_penalty=0.0,
        paired_slot_reg_weight=0.0,
        salient_prior_components=0,
        salient_assignment_penalty=0.0,
        penalize_salient_spatial=False,
        mask_salient_mean_on_background=False,
        background_representation="full",
        condition_only_salient_spatial=True,
    ),
    "both_sal8_timebatch_mae0": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=0,
        salient_latent_dim=8, batch_key="timepoint",
    ),
    "both_sal8_timebatch_mae30": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=30,
        salient_latent_dim=8, batch_key="timepoint",
    ),
    "both_sal8_timebatch_mae60": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=60,
        salient_latent_dim=8, batch_key="timepoint",
    ),
    "both_sal8_timebatch_mae90": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint",
    ),
    # End-to-end replacements for post-hoc paired-slot residualization. These
    # penalize ridge-linear predictability inside training; exported embeddings
    # remain raw and require no cleanup.
    "both_sal8_timebatch_reg1": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=1.0,
    ),
    "both_sal8_timebatch_reg5": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=5.0,
    ),
    "both_sal8_timebatch_reg20": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=20.0,
    ),
    # Two-stage objective: first learn the representations under MAE, then turn
    # on separation only after corruption ends. A ten-epoch ramp avoids an abrupt
    # optimizer shock at the stage boundary.
    "both_sal8_timebatch_stage_reg1": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=1.0,
        paired_slot_reg_start_epoch=90, paired_slot_reg_ramp_epochs=10,
    ),
    "both_sal8_timebatch_stage_reg1_private": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=1.0,
        paired_slot_reg_start_epoch=90, paired_slot_reg_ramp_epochs=10,
        background_representation="private",
    ),
    # Capacity/training screen. The long variants preserve the
    # selected 60% MAE / 40% separation-stage ratio at 250 total epochs.
    "both_sal8_timebatch_stage_reg1_private_long": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=150,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=1.0,
        paired_slot_reg_start_epoch=150, paired_slot_reg_ramp_epochs=10,
        background_representation="private",
    ),
    "both_sal16_timebatch_stage_reg1_private": dict(
        latent_dim=64, n_spatial_latent=16, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=16, batch_key="timepoint", paired_slot_reg_weight=1.0,
        paired_slot_reg_start_epoch=90, paired_slot_reg_ramp_epochs=10,
        background_representation="private",
    ),
    "both_sal16_timebatch_stage_reg1_private_long": dict(
        latent_dim=64, n_spatial_latent=16, spatial_attach="both", mae_epochs=150,
        salient_latent_dim=16, batch_key="timepoint", paired_slot_reg_weight=1.0,
        paired_slot_reg_start_epoch=150, paired_slot_reg_ramp_epochs=10,
        background_representation="private",
    ),
    "both_sal8_timebatch_stage_reg1_private_hidden256": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=90,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
    ),
    "both_sal8_timebatch_stage_reg1_private_hidden256_long": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=150,
        salient_latent_dim=8, n_hidden=256, batch_key="timepoint",
        paired_slot_reg_weight=1.0, paired_slot_reg_start_epoch=150,
        paired_slot_reg_ramp_epochs=10, background_representation="private",
    ),
    "both_sal8_timebatch_stage_reg5": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=5.0,
        paired_slot_reg_start_epoch=90, paired_slot_reg_ramp_epochs=10,
    ),
    "both_sal8_timebatch_stage_reg20": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=90,
        salient_latent_dim=8, batch_key="timepoint", paired_slot_reg_weight=20.0,
        paired_slot_reg_start_epoch=90, paired_slot_reg_ramp_epochs=10,
    ),
    "both_sal8_timebatch_qfree": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=60,
        salient_latent_dim=8, batch_key="timepoint", gat_self_query=False,
    ),
    "both_sal8_slicebatch": dict(
        n_spatial_latent=8, spatial_attach="both", mae_epochs=60,
        salient_latent_dim=8, batch_key="slice_id",
    ),
    "both_sal4": dict(n_spatial_latent=4, spatial_attach="both",
                      mae_epochs=60, salient_latent_dim=4),
    "both_sp8": dict(n_spatial_latent=8, spatial_attach="both", mae_epochs=60),
    # soft Wasserstein, never tried on `both` -- only the hard weight-1000 mask was,
    # and a graded version may behave differently from an all-or-nothing one
    # THE MISSING CONSTRAINT. `s_sp` and `z_sp` are two GAT branches over the same
    # neighbourhood and no penalty ever related them, so on v5 the salient spatial
    # branch simply copied the background one: 0.933 on gt_niche, 0.068 on its own
    # gt_sal_domain. Not collapse (std 0.63, healthy) and not a starved
    # neighbourhood (k 8->32 only moved it 0.068->0.234). These add I(s_sp; z_sp).
    # Weight 5 measured WORSE than weight 1 on the fast proxy (niche leak 0.778 vs
    # 0.715, margin 0.197 vs 0.294), so the penalty is not monotone -- too much of it
    # degrades the branch rather than redirecting it. Bracketing below weight 1.
    # THE SYNTHESIS. Receptive field and the independence penalty fix DIFFERENT
    # halves of the same failure, and each is useless without the other:
    #   k=8  -> s_sp cannot extract the subtle domain (0.116) but is clean (niche 0.080)
    #   k=64 -> extraction jumps to 0.404, but the wider view reinstates the
    #           hijacking (niche 0.610), so margin barely moves
    # The penalty measured as a no-op earlier because it was tested at k=8, where
    # capacity limiting had already removed the leak it targets. At k=64 there is
    # something for it to suppress.
    "both_sal8_k64_sppen05": dict(n_spatial_latent=8, spatial_attach="both",
                                  mae_epochs=60, salient_latent_dim=8,
                                  spatial_pair_penalty=0.5),
    "both_sal8_k64_sppen1": dict(n_spatial_latent=8, spatial_attach="both",
                                 mae_epochs=60, salient_latent_dim=8,
                                 spatial_pair_penalty=1.0),
    "both_sppen05": dict(n_spatial_latent=16, spatial_attach="both",
                         mae_epochs=60, spatial_pair_penalty=0.5),
    "both_sal8_sppen1": dict(n_spatial_latent=8, spatial_attach="both", mae_epochs=60,
                             salient_latent_dim=8, spatial_pair_penalty=1.0),
    "both_sppen1": dict(n_spatial_latent=16, spatial_attach="both",
                        mae_epochs=60, spatial_pair_penalty=1.0),
    "both_sppen5": dict(n_spatial_latent=16, spatial_attach="both",
                        mae_epochs=60, spatial_pair_penalty=5.0),
    "both_sal8_sppen5": dict(n_spatial_latent=8, spatial_attach="both", mae_epochs=60,
                             salient_latent_dim=8, spatial_pair_penalty=5.0),
    "both_wass2": dict(n_spatial_latent=16, spatial_attach="both",
                       mae_epochs=60, wasserstein_penalty=2.0),
    # Hidden-width capacity is now exposed and screened by the staged private
    # variants above; these older variants remain for historical diagnostics.
    # Symmetric independence instead of no independence: constrain BOTH intrinsic
    # latents against their spatial partner, so there is no unconstrained
    # reservoir for displaced anatomy. The repo turns this off by default,
    # arguing it deletes the lesion from the salient latent; our leak numbers say
    # the trade is worth re-measuring rather than assuming.
    "both_symmi": dict(n_spatial_latent=16, spatial_attach="both",
                       penalize_salient_spatial=True),
}

SHARED = dict(
    latent_dim=32,
    gene_likelihood="poisson",
    # Pure GAT is the current benchmark method. The legacy COVET auxiliary target
    # is deliberately not exposed by this runner.
    me_weight=0.0,
    me_reduction="sum",
    reg_to_use="mi",
    lam_mi=5.0,
    kl_spatial_weight=0.01,
    spatial_encoder="gat",
    condition_key="condition",
    condition_label="disease",
    # Matched-pair batch. Unlike slice_id, every level contains both conditions.
    batch_key="timepoint",
)


def validate_batch_condition(obs, batch_key, condition_key="condition"):
    """Reject decoder batches that encode condition instead of nuisance."""
    if batch_key is None:
        return
    missing = [key for key in (batch_key, condition_key) if key not in obs]
    if missing:
        raise KeyError(f"missing batch-contract columns: {missing}")
    table = pd.crosstab(obs[batch_key], obs[condition_key], dropna=False)
    bad = table.index[(table > 0).sum(axis=1) < 2].astype(str).tolist()
    if bad:
        preview = ", ".join(bad[:6])
        raise ValueError(
            f"batch_key={batch_key!r} is condition-confounded: {len(bad)} level(s) "
            f"contain only one {condition_key!r} class ({preview}). Use the matched "
            "timepoint key or pass --allow_condition_confounded_batch only for a "
            "diagnostic ablation."
        )


def extract_all_cells(h, n_layer=3, orthogonalize_paired_slots=True,
                      orthogonalize_alpha=1e-3, return_raw=False):
    """`HarukaSpatial.extract_rep`, but over every cell instead of the target arm.

    Needed because the background latent's leak tests (vs condition, vs slice)
    are undefined on a single-condition subset.
    """
    import cellcharter as cc
    import squidpy as sq

    full = h.input_adata
    if "slice_id" in full.obs:
        full.obs["slice_id"] = full.obs["slice_id"].astype("category")

    kinds = ["background", "salient"]
    if getattr(h, "n_spatial_latent", 0) > 0:
        kinds.append("spatial")
    # The 2x2 (`spatial_attach="both"`) carries a FOURTH representation: the
    # niche structure specific to the treated arm. `HarukaSpatial.extract_rep`
    # extracts it but `cluster()` deliberately leaves it out of its spec list, so
    # it is easy to miss -- and it is exactly the representation the README
    # credits with niche recovery (8-9/12 seeds for `both` vs 1-2/12 for
    # `background`). Omitting it would benchmark the 2x2 on everything except the
    # thing it is for.
    if getattr(h.model.module, "spatial_attach", None) == "both":
        kinds.append("salient_spatial")
    for kind in kinds:
        full.obsm[f"{kind}_rep"] = h.model.get_latent_representation(
            full, representation_kind=kind
        ).astype(np.float32)

    # Match the training/generative contract in exported representations. The
    # control salient posterior remains available internally for ContrastiveVI's
    # Wasserstein term, but salient variables are absent by definition in the
    # control arm and therefore exported as exact zeros.
    if getattr(h.model.module, "condition_only_salient_spatial", False):
        is_control = np.zeros(full.n_obs, dtype=bool)
        is_control[np.asarray(h.background_indices, dtype=int)] = True
        full.obsm["salient_rep"][is_control] = 0.0
        if "salient_spatial_rep" in full.obsm:
            full.obsm["salient_spatial_rep"][is_control] = 0.0

    sq.gr.spatial_neighbors(full, library_key="slice_id", spatial_key="spatial")
    cc.gr.remove_long_links(full)

    def aggregate():
        aggregated = {}
        for kind in kinds:
            cc.gr.aggregate_neighbors(full, n_layers=n_layer, use_rep=f"{kind}_rep",
                                      out_key=f"X_cc_{kind}", sample_key="slice_id")
            aggregated[kind] = np.asarray(
                full.obsm[f"X_cc_{kind}"], dtype=np.float32
            ).copy()
        return aggregated

    # Capture raw and cleaned views from the exact same fit. Retraining a second
    # time would mix MAE effects with optimizer/GPU nondeterminism. Aggregation is
    # performed independently because the cleanup contract acts on the own-cell
    # latent before CellCharter builds its neighbourhood blocks.
    raw = aggregate() if return_raw else None

    # A 2x2 decomposition claims paired intrinsic/spatial slots, but the training
    # loss leaves their linear subspaces partly shared. Remove only the component
    # predictable from the paired GAT slot. This uses representations alone -- no
    # condition or simulator truth -- and runs before CellCharter aggregation so
    # both own and aggregated readouts obey the same contract.
    if orthogonalize_paired_slots and "spatial" in kinds:
        full.obsm["background_rep"] = residualize(
            full.obsm["background_rep"], full.obsm["spatial_rep"],
            alpha=orthogonalize_alpha,
        )
        if "salient_spatial" in kinds:
            full.obsm["salient_spatial_rep"] = residualize(
                full.obsm["salient_spatial_rep"], full.obsm["spatial_rep"],
                alpha=orthogonalize_alpha,
            )

    out = aggregate()
    return (out, raw) if return_raw else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variants", nargs="+", default=["both_sal8_timebatch"])
    ap.add_argument("--out_root", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k_spatial", type=int, default=8)
    ap.add_argument("--allow_condition_confounded_batch", action="store_true",
                    help="diagnostic only: allow a batch level containing one condition")
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument(
        "--orthogonalize-paired-slots", action=argparse.BooleanOptionalAction,
        default=False,
        help="label-free ridge residualization of background and salient-spatial "
             "slots against the shared spatial slot (diagnostic only; default: disabled)",
    )
    ap.add_argument("--orthogonalize-alpha", type=float, default=1e-3)
    ap.add_argument(
        "--raw_out_root", default=None,
        help="optional second result root for unorthogonalized representations "
             "from the same trained weights",
    )
    ap.add_argument("--save_model", action="store_true",
                    help="checkpoint the fit so representations can be re-extracted "
                         "without retraining")
    args = ap.parse_args()

    from haruka_spatial import HarukaSpatial

    tag = os.path.basename(args.data).replace("sim_", "").replace(".h5ad", "")
    base = ad.read_h5ad(args.data)
    base.layers["counts"] = base.X.copy()
    sc.pp.normalize_total(base, target_sum=1e4)
    sc.pp.log1p(base)

    for variant in args.variants:
        out_dir = os.path.join(args.out_root, tag, f"haruka_{variant}_seed{args.seed}")
        print(f"\n===== haruka:{variant} on {tag} =====", flush=True)
        adata = base.copy()

        with Timer() as t:
            h = HarukaSpatial(adata, cov_gene_num=0, seed=args.seed,
                              k_spatial=args.k_spatial, library_key="slice_id")
            h.prepare_spatial_graph()
            # Merge rather than double-splat: `**SHARED, **VARIANTS[v]` raises
            # TypeError the moment a variant overrides a SHARED key, so no ablation
            # of a shared default (lam_mi, me_weight, kl_spatial_weight, ...) was
            # expressible. Variant wins on conflict.
            kwargs = {**SHARED, **VARIANTS[variant]}
            if kwargs.get("me_weight", 0.0) != 0.0:
                raise ValueError("the benchmark runner is pure GAT; me_weight must be 0")
            if not args.allow_condition_confounded_batch:
                validate_batch_condition(
                    adata.obs, kwargs.get("batch_key"), kwargs.get("condition_key", "condition")
                )
            h.setup_model(seed=args.seed, **kwargs)
            h.train(max_epochs=args.max_epochs, batch_size=args.batch_size,
                    plan_kwargs=dict(lr=args.lr))
            extracted = extract_all_cells(
                h, n_layer=args.n_layer,
                orthogonalize_paired_slots=args.orthogonalize_paired_slots,
                orthogonalize_alpha=args.orthogonalize_alpha,
                return_raw=args.raw_out_root is not None,
            )
            if args.raw_out_root is None:
                reps, raw_reps = extracted, None
            else:
                reps, raw_reps = extracted

        pure_gat = h.cov_gene_num == 0 and kwargs.get("me_weight", 0.0) == 0.0
        if not pure_gat:
            raise RuntimeError("benchmark contract violated: expected a pure-GAT model")
        common_meta = dict(
            max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr,
            model_seed=args.seed, rng_reset_at_setup=True,
            k_spatial=args.k_spatial, cov_gene_num=h.cov_gene_num,
            n_layer=args.n_layer, shared=SHARED, variant_kwargs=VARIANTS[variant],
            resolved_kwargs=kwargs, training_layer=h.training_layer_,
            count_contract_validated=(h.training_layer_ == "counts"),
            pure_gat=True, covet_enabled=False,
            paired_slot_orthogonalization=args.orthogonalize_paired_slots,
            orthogonalize_alpha=(args.orthogonalize_alpha
                                 if args.orthogonalize_paired_slots else None),
        )
        save_embeddings(
            out_dir,
            background=reps["background"], salient=reps["salient"],
            spatial=reps.get("spatial"),
            salient_spatial=reps.get("salient_spatial"),
            obs_names=base.obs_names.to_numpy(),
            method=f"haruka_{variant}", single_rep=False, runtime_s=t.seconds,
            variant=variant, **common_meta,
        )

        if raw_reps is not None:
            raw_out_dir = os.path.join(
                args.raw_out_root, tag, f"haruka_{variant}_seed{args.seed}"
            )
            raw_meta = {
                **common_meta,
                "paired_slot_orthogonalization": False,
                "orthogonalize_alpha": None,
                "paired_view_source": out_dir,
            }
            save_embeddings(
                raw_out_dir,
                background=raw_reps["background"], salient=raw_reps["salient"],
                spatial=raw_reps.get("spatial"),
                salient_spatial=raw_reps.get("salient_spatial"),
                obs_names=base.obs_names.to_numpy(),
                method=f"haruka_{variant}_raw", single_rep=False,
                runtime_s=t.seconds, variant=variant, **raw_meta,
            )

        if args.save_model:
            h.model.save(os.path.join(out_dir, "model"), overwrite=True,
                         save_anndata=False)


if __name__ == "__main__":
    main()
