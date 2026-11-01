"""v4: v3's factor structure, but a NON-LINEAR observation model.

Same four ground-truth factors as v3 (see below), and deliberately the same
latents -- what changes is only how they reach the genes.

    v3:  cat([z, z_sp, s, s_sp]) -> tanh MLP -> softmax -> Poisson
    v4:  h_bg = tanh(cat([z, z_sp]) @ W1)
         gate = 1 + g * tanh(cat([s, s_sp]) @ Wg)
         h    = h_bg * gate          <-- multiplicative interaction
                     -> softmax -> Poisson

WHY. Measured on v3 (`diag_salient_geometry.py`):

    pairwise cosine of per-cell-type effect directions   0.795
    salient state from raw expression, linear probe      AUROC 0.988
    salient state from raw expression, non-linear probe  AUROC 0.992

That is a single shared additive perturbation, linearly readable off expression.
It is precisely the generative model CINEMA-OT assumes, and it means any method
retaining expression recovers the state for free -- which is why SIMVI scored
0.991 on the salient quadrant while leaking 0.917, and why CINEMA-OT beat most
Haruka variants. v3 could not distinguish disentanglement from absorption.

Under the gate, flipping `sal_state` changes genes by `h_bg * (gate - 1) @ W2`,
which depends on h_bg and hence on cell type: every type responds along its own
direction and no shared additive effect exists.

The gate acts in the DECODER, not on the latents, so z / z_sp / s / s_sp remain
exactly as clean and as mutually independent as in v3 and the oracle floor is
unchanged. Indexing `A_ss[cell_type, sal_state]` instead would have put cell-type
information inside the ground-truth salient factor and changed what the 2x2 means.

    python build_sim_v4.py --deltas 1.0 --seeds 0 --gate_strength 0.8

For the controlled 2x2 in which the salient pair mirrors the background pair:

    python build_sim_v4.py --align_salient_to_background --deltas 1.0 --seeds 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

D_CT, D_BGD, D_SS, D_SD = 10, 6, 4, 6

K_CT = 7           # cell types
K_BGD = 3          # background spatial domains
K_SS = 2           # salient states -- BINARY, as specified
K_SAD = 3          # salient spatial domains

DOMAIN_LS = 0.10   # spatial granularity of both domain fields
TIMEPOINTS = ["T1", "T2", "T3", "T4"]


def _stable_seed(*parts) -> int:
    """Process-independent RNG seed for a simulator slice.

    Python's built-in ``hash`` is salted per process, so the same advertised
    simulator seed previously produced different counts after a fresh launch.
    """
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2 ** 31)


def _lattice(n_side, rng):
    gx, gy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    xy = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    return xy + rng.uniform(-0.3, 0.3, xy.shape)


def _domains(xy, k, rng, grid=128, ls=DOMAIN_LS):
    """`k` contiguous spatial regions: argmax of k smooth fields."""
    fields = []
    for _ in range(k):
        f = gaussian_filter(rng.standard_normal((grid, grid)), sigma=ls * grid, mode="wrap")
        fields.append((f - f.mean()) / (f.std() + 1e-12))
    u = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
    ij = np.clip((u * (grid - 1)).round().astype(int), 0, grid - 1)
    return np.stack([f[ij[:, 1], ij[:, 0]] for f in fields], 1).argmax(1)


def _bilinear_interaction(bg_hidden, sal_hidden, coherence=0.0):
    """Strict background-by-salient interaction with tunable global coherence.

    ``coherence=0`` is the original coordinate-wise Hadamard product. Its signed
    background carrier can rotate or reverse the same salient state across cell
    types, making one ground-truth state form several background-conditional
    clusters. ``coherence=1`` replaces that carrier by the per-cell background
    RMS. The carrier is then non-negative, so state directions agree globally,
    while the effect remains non-additive and depends on the background state.

    Both endpoints, and every interpolation between them, are strict interactions:
    the result is exactly zero when either input is zero. The RMS carrier also has
    the same mean-square scale as the coordinate-wise carrier, avoiding a hidden
    effect-strength change as coherence is tuned.
    """
    if not 0.0 <= coherence <= 1.0:
        raise ValueError("bilinear_coherence must be in [0, 1]")
    bg_rms = np.sqrt(np.mean(np.square(bg_hidden), axis=1, keepdims=True))
    if coherence == 0.0:
        return bg_hidden * sal_hidden
    carrier = (1.0 - coherence) * bg_hidden + coherence * bg_rms
    carrier_rms = np.sqrt(np.mean(np.square(carrier), axis=1, keepdims=True))
    carrier = carrier * (bg_rms / np.maximum(carrier_rms, 1e-12))
    return carrier * sal_hidden


def _salient_state_probabilities(
    sal_domain,
    n_states,
    q_ss,
    p_ss,
    independent=False,
    independent_binary_rate=0.5,
):
    """Per-cell salient-state probabilities, optionally independent of domain.

    The legacy simulator deliberately made salient intrinsic composition depend on
    salient spatial domain. That is biologically plausible, but incompatible with
    an objective that penalises ``I(s, s_sp)``: the ground-truth factors themselves
    were dependent. The opt-in independent mode removes only that edge. Spatial
    domains remain smooth, while intrinsic states are sampled i.i.d. within every
    slice with an identical marginal distribution in every domain.
    """
    sal_domain = np.asarray(sal_domain, dtype=int)
    if independent:
        if n_states == 2:
            return np.full(len(sal_domain), independent_binary_rate, dtype=float)
        return np.full((len(sal_domain), n_states), 1.0 / n_states, dtype=float)
    if n_states == 2:
        return np.asarray(q_ss, dtype=float)[sal_domain]
    return np.asarray(p_ss, dtype=float)[sal_domain]


def build(delta, seed, n_side, n_genes, out_dir, sal_rates=(0.1, 0.5, 0.9),
          gate_strength=0.8, type_gate=True, delta_sp=1.0, n_timepoints=4,
          composition_shift=0.0, effect_mode='gate', n_salient_states=2,
          align_salient_to_background=False, bilinear_coherence=0.0,
          independent_salient_factors=False, independent_salient_rate=0.5):
    """Build one simulated dataset.

    align_salient_to_background is a controlled identifiability preset. It
    gives the salient pair the same number of intrinsic classes, latent widths,
    domain-conditioned class composition and effect scale as the background pair.
    Its observation path can be additive or nonlinear fusion. The only intentional
    asymmetry left is the one
    required by a contrastive experiment: salient effects are absent in controls.
    """
    if not 0.0 <= bilinear_coherence <= 1.0:
        raise ValueError("bilinear_coherence must be in [0, 1]")
    if bilinear_coherence > 0 and effect_mode != "bilinear":
        raise ValueError("bilinear_coherence is only defined for effect_mode='bilinear'")
    if not 0.0 < independent_salient_rate < 1.0:
        raise ValueError("independent_salient_rate must be in (0, 1)")
    if independent_salient_factors and align_salient_to_background:
        raise ValueError(
            "independent_salient_factors and align_salient_to_background are "
            "different simulation contracts and cannot be combined"
        )
    if align_salient_to_background:
        if not np.isclose(delta, 1.0) or not np.isclose(delta_sp, 1.0):
            raise ValueError(
                "aligned generation requires delta=1 and delta_sp=1 so salient "
                "and background effect scales match"
            )
        if not np.isclose(composition_shift, 0.0):
            raise ValueError(
                "aligned generation requires composition_shift=0 so the shared "
                "domain-conditioned composition is unchanged between arms"
            )
        # These are properties of the preset rather than extra knobs. Keeping the
        # resolution here also makes programmatic calls behave like the CLI flag.
        # Preserve an explicitly requested nonlinear fusion model. The CLI's
        # default gate remains mapped to additive for backward compatibility
        # with the original aligned benchmark.
        if effect_mode == "gate":
            effect_mode = "additive"
        n_salient_states = K_CT

    dec = np.random.default_rng(40_000 + seed)
    A_ct = dec.normal(0, 1.4, (K_CT, D_CT))
    A_bgd = dec.normal(0, 1.4, (K_BGD, D_BGD))
    # n_salient_states generalises the salient intrinsic factor from the binary
    # default to K classes. In the legacy path state 0 stays the untreated origin;
    # the aligned path instead masks the whole salient centre in controls.
    n_ss = int(n_salient_states)
    d_ss = D_CT if align_salient_to_background else D_SS
    d_sd = D_BGD if align_salient_to_background else D_SD
    if align_salient_to_background:
        # Every disease-state class has the same kind of centre as a background
        # cell type. Absence in controls is applied below as an arm mask, rather
        # than overloading class 0 to mean both "a state" and "no perturbation".
        A_ss = dec.normal(0, 1.4, (n_ss, d_ss))
        A_sd = dec.normal(0, 1.4, (K_SAD, d_sd))
    else:
        A_ss = np.vstack([np.zeros((1, d_ss)), dec.normal(0, 1.4, (n_ss - 1, d_ss))])
        A_sd = np.vstack([np.zeros((1, d_sd)), dec.normal(0, 1.4, (K_SAD - 1, d_sd))])

    # Composition of cell types PER background domain. Dirichlet(0.5) so the
    # domains differ sharply enough that a 60-neighbour pool separates them, while
    # every type still occurs everywhere (no domain is a single type, which would
    # make own-type fully determine the domain and collapse the pair).
    p_ct = dec.dirichlet(np.full(K_CT, 0.5), size=K_BGD)
    q_ss = np.asarray(sal_rates)[:K_SAD]        # P(state=1 | salient domain), K=2 only
    # SYMMETRIC MULTI-CLASS SALIENT STATE. For K != 2 the state is drawn from a
    # per-domain composition exactly as the cell type is drawn from a per-background
    # -domain composition:
    #     cell_type ~ Categorical(p_ct[bg_domain]),  p_ct ~ Dirichlet(0.5)
    #     sal_state ~ Categorical(p_ss[sal_domain]), p_ss ~ Dirichlet(0.5)
    # so the two intrinsic factors differ only in name. The Bernoulli path is kept
    # for K=2 so every existing dataset reproduces bit-for-bit (the Dirichlet is
    # only drawn when K != 2, leaving the RNG stream untouched otherwise).
    p_ss = (
        p_ct.copy()
        if align_salient_to_background
        else (dec.dirichlet(np.full(n_ss, 0.5), size=K_SAD) if n_ss != 2 else None)
    )

    # W1 now takes only the BACKGROUND factors; the salient factors enter through
    # Wg as a multiplicative gate on the resulting hidden layer.
    d_bg, d_sal = D_CT + D_BGD, d_ss + d_sd
    W1 = dec.normal(0, 1 / np.sqrt(d_bg), (d_bg, 128)); b1 = dec.normal(0, .1, 128)
    if type_gate:
        # PER-CELL-TYPE gate weights, CENTRED SO THEY AVERAGE TO ZERO ACROSS TYPES.
        # A single shared Wg leaves a shared additive component (the gate scales
        # h_bg, whose cross-type mean is non-zero), and that component is what the
        # linear probe keeps finding -- measured AUROC 0.885/0.944/0.972 at gate
        # 0.8/1.2/1.6, i.e. RISING with gate strength. Centring across types makes
        # the pooled effect cancel: averaged over cell types the perturbation has
        # no preferred direction, so it is invisible to any method that does not
        # condition on cell type, while remaining strong within each type.
        #
        # This indexes the DECODER weights by cell type. It does NOT put cell type
        # into the latent `s`, which stays a clean binary factor, so the ground
        # truth and the oracle floor are untouched.
        Wg = dec.normal(0, 1 / np.sqrt(d_sal), (K_CT, d_sal, 128))
        Wg = Wg - Wg.mean(0, keepdims=True)
    else:
        Wg = dec.normal(0, 1 / np.sqrt(d_sal), (d_sal, 128))
    d_all = d_bg + d_sal
    W1_add = dec.normal(0, 1 / np.sqrt(d_all), (d_all, 128))
    # FUSION uses separate factor projections plus an explicit Hadamard cross
    # term. This keeps the factor geometry symmetric while making the observed
    # perturbation depend on the background state rather than being additive.
    W_bg_fuse = dec.normal(0, 1 / np.sqrt(d_bg), (d_bg, 128))
    W_sal_fuse = dec.normal(0, 1 / np.sqrt(d_sal), (d_sal, 128))
    W2 = dec.normal(0, 1 / np.sqrt(128), (128, n_genes)); b2 = dec.normal(0, .1, n_genes)

    bgd_rng = np.random.default_rng(20_000 + seed)
    sad_rng = np.random.default_rng(30_000 + seed)

    # Shrink by dropping TIMEPOINTS, never by shrinking the lattice. The spatial
    # structure is generated per slice, so fewer slices leaves every spatial
    # statistic identical while cutting cells proportionally. Shrinking n_side
    # instead changes the benchmark: the 3-hop pooling window spans ~37 cells
    # regardless, so on a smaller lattice it covers a larger fraction of each
    # domain and mixes across boundaries -- measured, that dropped the
    # salient-domain ceiling from 0.640 to 0.420, i.e. it altered the very
    # quantity a fast proxy is meant to preserve.
    rows = []
    for tp in TIMEPOINTS[:n_timepoints]:
        for cond in ("control", "disease"):
            treated = cond == "disease"
            srng = np.random.default_rng(_stable_seed(seed, tp, cond))
            xy = _lattice(n_side, srng); n = len(xy)

            bg_domain = _domains(xy, K_BGD, np.random.default_rng(bgd_rng.integers(1 << 30)))

            # COMPOSITION SHIFT between arms. With shift=0 both arms draw cell types
            # from the same p_ct, so every treated cell has a look-alike control and
            # optimal-transport matching (CINEMA-OT) recovers the effect almost for
            # free: the match is same-cell-type, so the differencing is already
            # conditioned on cell type and the cell-type-specific gate costs it
            # nothing. That is a property of the SIMULATION, not of the method's
            # ability, and real perturbations do change composition.
            #
            # With shift>0 the treated arm's composition is tilted, so a treated cell
            # may have no same-type counterpart to match against and the matched
            # difference mixes a genuine perturbation effect with a composition
            # difference. Methods that MODEL the effect are unaffected; methods that
            # match-and-difference degrade. Applied only to the treated arm, and only
            # to the composition -- the per-cell salient state and both domain fields
            # are untouched, so all four ground-truth factors keep their meaning.
            p_use = p_ct
            if treated and composition_shift > 0:
                tilt = np.linspace(1.0 - composition_shift, 1.0 + composition_shift, K_CT)
                p_use = p_ct * tilt[None, :]
                p_use = p_use / p_use.sum(1, keepdims=True)
            cell_type = np.array([srng.choice(K_CT, p=p_use[d]) for d in bg_domain])

            sal_domain_cf = _domains(xy, K_SAD, np.random.default_rng(sad_rng.integers(1 << 30)))
            state_probs = _salient_state_probabilities(
                sal_domain_cf,
                n_ss,
                q_ss,
                p_ss,
                independent=independent_salient_factors,
                independent_binary_rate=independent_salient_rate,
            )
            if n_ss == 2:
                sal_state_cf = (srng.random(n) < state_probs).astype(int)
            else:
                sal_state_cf = np.array(
                    [srng.choice(n_ss, p=prob) for prob in state_probs]
                )

            # counterfactual labels retained on controls so the FP probes are defined
            sal_state = np.where(treated, sal_state_cf, 0)
            sal_domain = np.where(treated, sal_domain_cf, 0)

            z = A_ct[cell_type] + srng.normal(0, .35, (n, D_CT))
            z_sp = A_bgd[bg_domain] + srng.normal(0, .35, (n, D_BGD))
            if align_salient_to_background:
                # Disease uses the same centre-plus-noise construction as z/z_sp;
                # controls retain only posterior-scale noise around an absent effect.
                s_center = A_ss[sal_state_cf] if treated else np.zeros((n, d_ss))
                s_sp_center = A_sd[sal_domain_cf] if treated else np.zeros((n, d_sd))
            else:
                s_center = A_ss[sal_state]
                s_sp_center = A_sd[sal_domain]
            s = delta * s_center + srng.normal(0, .35, (n, d_ss))
            # `delta_sp` scales the salient SPATIAL factor independently of the
            # salient intrinsic one. Without it the two were locked together by
            # `delta`, and the salient_spatial quadrant sat at a ceiling of 0.985:
            # sal_domain reaches the genes by TWO routes -- indirectly by setting
            # P(sal_state=1), and DIRECTLY through this term feeding the gate.
            # Tightening `sal_rates` only weakens the first (state-pooling oracle
            # 0.957 -> 0.724) and left the ceiling at 0.985, because this direct
            # route stamps the domain into expression on its own (raw expr ->
            # sal_domain = 0.73, pooled = 0.99). Lowering delta_sp is what actually
            # makes the quadrant discriminate.
            s_sp = delta * delta_sp * s_sp_center + srng.normal(0, .35, (n, d_sd))

            # In the independent benchmark, absence in controls is exact in the
            # observation model, matching ContrastiveVI's decoder contract. The
            # posterior-scale noise above remains part of the latent simulation,
            # but cannot create a hidden treatment-like effect in reference cells.
            s_obs = s
            s_sp_obs = s_sp
            if independent_salient_factors and not treated:
                s_obs = np.zeros_like(s)
                s_sp_obs = np.zeros_like(s_sp)

            # MULTIPLICATIVE GATING -- the difference from v3, and the whole point
            # of v4. v3 concatenated all four factors into one MLP, which made the
            # salient perturbation a single additive vector shared by every cell
            # type (measured: pairwise cosine 0.795 across types, and the state was
            # linearly readable off raw expression at AUROC 0.988). That is exactly
            # the model CINEMA-OT assumes, so v3 tested it on its home turf.
            #
            # Here the salient factors instead MODULATE the background
            # representation. The gene-space effect of flipping `sal_state` is
            # `h_bg * (gate - 1) @ W2`, which is a function of h_bg and therefore of
            # cell type and background domain: each cell type responds to the same
            # perturbation along its own direction. No additive, confounder-free
            # effect exists to be found.
            #
            # Crucially the modulation lives in the DECODER, not in the latents.
            # Making the response cell-type-specific the obvious way -- indexing
            # `A_ss[cell_type, sal_state]` -- would inject cell-type information
            # into the ground-truth salient factor itself, raising the oracle floor
            # and corrupting what the 2x2 means. Here z, z_sp, s, s_sp stay exactly
            # as clean and as mutually independent as in v3, so the floor is
            # unchanged and only the OBSERVATION model becomes non-linear.
            if effect_mode in ("additive", "fusion", "bilinear"):
                # v3's observation model: all four factors concatenated into one
                # MLP, so the salient perturbation is a single shared direction.
                # Kept as an option so the ladder has a rung where OUR assumptions
                # hold AND the salient-spatial quadrant is still non-trivial --
                # v3 had the first but not the second (its salsp ceiling was 0.985,
                # free to everyone), v5 has the second but adds the cell-type
                # specific gate. Neither isolates the spatial branch in the
                # well-specified case; this does.
                if effect_mode == "additive":
                    h = np.tanh(
                        np.concatenate([z, z_sp, s_obs, s_sp_obs], 1) @ W1_add + b1
                    )
                else:
                    bg_hidden = np.tanh(np.concatenate([z, z_sp], 1) @ W_bg_fuse + b1)
                    # The contrastive semantics say salient effects are absent in
                    # controls, not merely centred near zero. In bilinear mode the
                    # observation therefore receives an exact zero salient block
                    # for controls; posterior-scale latent noise must not become a
                    # hidden treatment-like effect in the reference arm.
                    sal_obs = np.concatenate([s_obs, s_sp_obs], 1)
                    if effect_mode == "bilinear" and not treated:
                        sal_obs = np.zeros((n, d_sal))
                    sal_hidden = np.tanh(sal_obs @ W_sal_fuse)
                    if effect_mode == "fusion":
                        h = np.tanh(
                            bg_hidden + sal_hidden
                            + gate_strength * bg_hidden * sal_hidden
                        )
                    else:
                        # STRICT INTERACTION ONLY: there is no additive salient
                        # main effect. Every treated response direction is a
                        # function of the background state. ``bilinear_coherence``
                        # controls whether that direction can sign-flip across
                        # backgrounds or uses a sign-preserving background carrier.
                        # Both are pure products; controls reduce to tanh(h_bg).
                        h = np.tanh(
                            bg_hidden + gate_strength * _bilinear_interaction(
                                bg_hidden, sal_hidden, bilinear_coherence
                            )
                        )
                logits = h @ W2 + b2 + srng.normal(0, .25, n_genes)
                px = np.exp(logits - logits.max(1, keepdims=True)); px /= px.sum(1, keepdims=True)
                lib = srng.lognormal(np.log(600), .3, size=n)[:, None]
                rows.append(dict(
                    counts=srng.poisson(lib * px).astype(np.float32), xy=xy,
                    obs=pd.DataFrame(dict(
                        slice_id=f"{tp}_{cond}", timepoint=tp, condition=cond,
                        gt_arch=cell_type.astype(str),
                        gt_niche=bg_domain.astype(str),
                        gt_sal_intrinsic=sal_state_cf.astype(str),
                        gt_sal_domain=sal_domain_cf.astype(str),
                    ))))
                continue
            h_bg = np.tanh(np.concatenate([z, z_sp], 1) @ W1 + b1)
            sal_cat = np.concatenate([s_obs, s_sp_obs], 1)
            drive = (np.einsum("nd,ndk->nk", sal_cat, Wg[cell_type]) if type_gate
                     else sal_cat @ Wg)
            gate = 1.0 + gate_strength * np.tanh(drive)
            h = h_bg * gate
            logits = h @ W2 + b2 + srng.normal(0, .25, n_genes)   # per-slice nuisance
            px = np.exp(logits - logits.max(1, keepdims=True)); px /= px.sum(1, keepdims=True)
            lib = srng.lognormal(np.log(600), .3, size=n)[:, None]

            rows.append(dict(
                counts=srng.poisson(lib * px).astype(np.float32), xy=xy,
                obs=pd.DataFrame(dict(
                    slice_id=f"{tp}_{cond}", timepoint=tp, condition=cond,
                    gt_arch=cell_type.astype(str),          # background x intrinsic
                    gt_niche=bg_domain.astype(str),         # background x spatial
                    gt_sal_intrinsic=sal_state_cf.astype(str),   # salient x intrinsic
                    gt_sal_domain=sal_domain_cf.astype(str),     # salient x spatial
                ))))

    X = np.vstack([r["counts"] for r in rows])
    obs = pd.concat([r["obs"] for r in rows], ignore_index=True)
    obs.index = obs.index.astype(str)
    for c in obs.columns:
        obs[c] = pd.Categorical(obs[c])

    out = ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                     var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    out.obsm["spatial"] = np.vstack([r["xy"] for r in rows]).astype(np.float32)
    out.layers["counts"] = out.X.copy()
    out.uns["sim2x2"] = dict(
        delta=float(delta), seed=int(seed),
        source=("v4_background_aligned" if align_salient_to_background
                else "v4_multiplicative_gate"),
        n_domains=(K_SAD if align_salient_to_background else K_SAD - 1),
        n_cells_per_slice=int(n_side ** 2),
        k_classes=dict(arch=K_CT, niche=K_BGD, sal_intr=n_ss, sal_dom=K_SAD),
        n_salient_states=int(n_ss),
        sal_rates=([] if align_salient_to_background else list(map(float, q_ss))),
        sal_rates_used=(not align_salient_to_background
                        and not independent_salient_factors),
        independent_salient_factors=bool(independent_salient_factors),
        independent_salient_rate=float(independent_salient_rate),
        shared_domain_class_composition=(p_ct.astype(np.float32)
                                         if align_salient_to_background
                                         else np.empty((0, 0), dtype=np.float32)),
        domain_ls=DOMAIN_LS,
        gate_strength=float(gate_strength), type_gate=bool(type_gate),
        n_timepoints=int(n_timepoints), composition_shift=float(composition_shift),
        effect_mode=str(effect_mode),
        interaction_only=bool(effect_mode == "bilinear"),
        salient_observation_exact_zero_in_controls=bool(
            effect_mode == "bilinear" or independent_salient_factors
        ),
        bilinear_coherence=float(bilinear_coherence),
        bilinear_carrier=("mixed_coordinate_and_background_rms"
                          if bilinear_coherence > 0 else "coordinate_wise"),
        align_salient_to_background=bool(align_salient_to_background),
        salient_composition_matches_background=bool(align_salient_to_background),
        latent_dims=dict(background_intrinsic=D_CT, background_spatial=D_BGD,
                         salient_intrinsic=d_ss, salient_spatial=d_sd),
        delta_sp=float(delta_sp),
        builder_revision="2026-08-09-independent-strong-salient-spatial-v1",
        rng_seed_scheme="blake2b-v1",
        truth_types={"gt_arch": "categorical", "gt_niche": "categorical",
                     "gt_sal_intrinsic": "categorical", "gt_sal_domain": "categorical"},
        quadrants=dict(background_intrinsic="gt_arch", background_spatial="gt_niche",
                       salient_intrinsic="gt_sal_intrinsic",
                       salient_spatial="gt_sal_domain"),
        note=(
            "salient intrinsic is i.i.d. and independent of salient spatial domain; "
            "the spatial factor is recoverable only through its direct smooth-domain "
            "effect"
            if independent_salient_factors else
            "intrinsic = per-cell categorical drawn at a region-specific RATE; "
            "spatial = that region. Pooling neighbours' intrinsic state estimates "
            "the rate and therefore the region, which is exactly what "
            "GAT(neighbours' q_m) computes."
        ),
    )
    os.makedirs(out_dir, exist_ok=True)
    spread = float(np.max(q_ss) - np.min(q_ss))
    stem = (
        "simv4_alignbg"
        if align_salient_to_background
        else (
            f"simv4_g{gate_strength:g}_indsp"
            if independent_salient_factors
            else f"simv4_g{gate_strength:g}_r{spread:g}"
        )
    )
    path = os.path.join(
        out_dir,
        f"{stem}_p{delta_sp:g}"
        f"{'' if composition_shift == 0 else f'_c{composition_shift:g}'}"
        f"{'_add' if effect_mode == 'additive' else (f'_fusion{gate_strength:g}' if effect_mode == 'fusion' else (f'_bilinear{gate_strength:g}' if effect_mode == 'bilinear' else ''))}"
        f"{f'_coh{bilinear_coherence:g}' if effect_mode == 'bilinear' and bilinear_coherence > 0 else ''}"
        f"{'' if n_ss == 2 else f'_k{n_ss}'}"
        f"{'' if n_timepoints == 4 else f'_t{n_timepoints}'}_d{delta:g}_s{seed}.h5ad")
    out.write_h5ad(path, compression="gzip")
    print(json.dumps(dict(path=path, n_obs=int(out.n_obs), delta=delta, seed=seed,
                          median_library=float(np.median(X.sum(1))),
                          cell_type_by_domain_maxdiff=float(np.abs(p_ct[0] - p_ct[1]).max())),
                     indent=2))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_side", type=int, default=95)
    ap.add_argument("--n_genes", type=int, default=300)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--gate_strength", type=float, default=0.8,
                    help="amplitude of the multiplicative salient gate")
    ap.add_argument("--sal_rates", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                    help="P(sal_state=1) per salient domain. The DEFAULT (0.1 0.5 "
                         "0.9) makes the salient_spatial quadrant trivial: pooling "
                         "~40 neighbours estimates the rate to SD~0.08 while the "
                         "rates sit 0.4 apart, so every method scored 0.87-0.99 "
                         "there regardless of its per-cell signal. Tighter rates "
                         "(0.35 0.5 0.65) put the gap at ~2 SD instead of ~5.")
    ap.add_argument("--n_salient_states", type=int, default=2,
                    help="number of salient intrinsic states. K!=2 draws them "
                         "from a per-domain Dirichlet composition, matching how "
                         "cell types are drawn per background domain.")
    ap.add_argument("--effect_mode", choices=["gate", "additive", "fusion", "bilinear"], default="gate",
                    help="'additive' uses a shared concatenated MLP; 'fusion' "
                         "adds background, salient, and multiplicative terms; "
                         "'bilinear' is strictly interaction-only with no additive "
                         "salient main effect and exact zero salient effect in controls.")
    ap.add_argument(
        "--bilinear_coherence", type=float, default=0.0,
        help="for bilinear mode, interpolate [0,1] from the original signed "
             "coordinate-wise carrier to a sign-preserving background-RMS "
             "carrier. Both remain strict, non-additive interactions; larger "
             "values make salient states more globally clusterable.",
    )
    ap.add_argument("--composition_shift", type=float, default=0.0,
                    help="tilt the treated arm's cell-type composition in "
                         "[0,1). Breaks the same-cell-type OT matching that "
                         "lets match-and-difference methods bypass the "
                         "cell-type-specific response.")
    ap.add_argument("--n_timepoints", type=int, default=4,
                    help="use fewer slices for a fast proxy; preserves all "
                         "spatial statistics, unlike shrinking --n_side")
    ap.add_argument("--delta_sp", type=float, default=1.0,
                    help="scale on the salient SPATIAL factor, independent of "
                         "delta. Lower makes the salient_spatial quadrant harder.")
    ap.add_argument(
        "--independent_salient_factors",
        action="store_true",
        help="sample salient intrinsic state i.i.d. with the same marginal in every "
             "salient spatial domain; also makes the salient observation exactly zero "
             "in controls",
    )
    ap.add_argument(
        "--independent_salient_rate",
        type=float,
        default=0.5,
        help="P(salient intrinsic state=1) in independent binary mode",
    )
    ap.add_argument("--shared_gate", action="store_true",
                    help="use ONE gate for all cell types (leaves a shared "
                         "additive component; the default is per-type, centred)")
    ap.add_argument(
        "--align_salient_to_background", action="store_true",
        help="controlled symmetry preset: use 7 salient states, 10/6 salient "
             "ground-truth dimensions, the exact background domain-conditioned "
             "class composition, and matched factor scales. Defaults to the additive "
             "path; pass --effect_mode fusion for nonlinear fusion. Requires "
             "delta=delta_sp=1 and composition_shift=0.",
    )
    args = ap.parse_args()
    for s in args.seeds:
        for d in args.deltas:
            build(d, s, args.n_side, args.n_genes, args.out_dir,
                  sal_rates=tuple(args.sal_rates),
                  gate_strength=args.gate_strength,
                  type_gate=not args.shared_gate, delta_sp=args.delta_sp,
                  n_timepoints=args.n_timepoints,
                  composition_shift=args.composition_shift,
                  effect_mode=args.effect_mode,
                  n_salient_states=args.n_salient_states,
                  align_salient_to_background=args.align_salient_to_background,
                  bilinear_coherence=args.bilinear_coherence,
                  independent_salient_factors=args.independent_salient_factors,
                  independent_salient_rate=args.independent_salient_rate)


if __name__ == "__main__":
    main()
