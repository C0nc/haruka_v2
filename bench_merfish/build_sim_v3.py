"""Generative process matched to what the 2x2 architecture can actually represent.

THE DESIGN (user's spec). Both pairs have the same shape: the INTRINSIC factor is a
per-cell categorical draw whose RATE varies by region, and the SPATIAL factor is
that region.

    1. bg_domain(x)   spatial regions
    2. cell_type_i  ~ Categorical(p[bg_domain_i])      i.i.d. GIVEN the domain
    3. sal_domain(x)  spatial regions, independent field
    4. sal_state_i  ~ Bernoulli(q[sal_domain_i])       treated arm only, binary

    z    = A_ct [cell_type]           background x intrinsic   -> gt_arch
    z_sp = A_bgd[bg_domain]           background x spatial     -> gt_niche
    s    = A_ss [sal_state]  * delta  salient    x intrinsic   -> gt_sal_intrinsic
    s_sp = A_sd [sal_domain] * delta  salient    x spatial     -> gt_sal_domain

    latent = cat([z, z_sp, s, s_sp]) -> tanh MLP -> softmax -> Poisson(library * px)

WHY THIS SHAPE AND NOT THE PREVIOUS ONES. The architecture computes
`z_sp = GAT(neighbours' q_m)` and `s_sp = GAT(neighbours' q_m_salient)`, so a
spatial factor is only representable if it is a function of the NEIGHBOURS'
INTRINSIC state. Two earlier designs failed for opposite reasons:

  * independent spatial factors (v1): neighbours' intrinsic state carries no
    information about them at all. Measured `salient_spatial -> gt_sal_domain`
    = 0.000, while `haruka_vanilla` -- which has no spatial branch -- recovered
    the same factor at 0.69-0.73.
  * spatial factor = residualised neighbourhood composition (v2): representable,
    but deliberately orthogonal to the parent, so what remained was a small
    boundary-only residual with an information ceiling of R^2 0.31-0.60.

Here the domain is what SETS the local composition, so pooling neighbours
estimates it directly and sharply: with 60 neighbours the composition is
estimated to ~1/sqrt(60), while a cell's OWN type is a single draw from the same
distribution. That gap is what makes self-exclusion meaningful -- the diagnostic
is `z_sp -> bg_domain` >> `z -> bg_domain`, NOT `z -> bg_domain` ~ 0. Own type is
*supposed* to be weakly informative about its own region; it is one sample.

    python build_sim_v3.py --deltas 1.0 --seeds 0
"""

from __future__ import annotations

import argparse
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


def build(delta, seed, n_side, n_genes, out_dir, sal_rates=(0.1, 0.5, 0.9)):
    dec = np.random.default_rng(40_000 + seed)
    A_ct = dec.normal(0, 1.4, (K_CT, D_CT))
    A_bgd = dec.normal(0, 1.4, (K_BGD, D_BGD))
    A_ss = np.vstack([np.zeros((1, D_SS)), dec.normal(0, 1.4, (K_SS - 1, D_SS))])
    A_sd = np.vstack([np.zeros((1, D_SD)), dec.normal(0, 1.4, (K_SAD - 1, D_SD))])

    # Composition of cell types PER background domain. Dirichlet(0.5) so the
    # domains differ sharply enough that a 60-neighbour pool separates them, while
    # every type still occurs everywhere (no domain is a single type, which would
    # make own-type fully determine the domain and collapse the pair).
    p_ct = dec.dirichlet(np.full(K_CT, 0.5), size=K_BGD)
    q_ss = np.asarray(sal_rates)[:K_SAD]        # P(state=1 | salient domain)

    d_lat = D_CT + D_BGD + D_SS + D_SD
    W1 = dec.normal(0, 1 / np.sqrt(d_lat), (d_lat, 128)); b1 = dec.normal(0, .1, 128)
    W2 = dec.normal(0, 1 / np.sqrt(128), (128, n_genes)); b2 = dec.normal(0, .1, n_genes)

    bgd_rng = np.random.default_rng(20_000 + seed)
    sad_rng = np.random.default_rng(30_000 + seed)

    rows = []
    for tp in TIMEPOINTS:
        for cond in ("control", "disease"):
            treated = cond == "disease"
            srng = np.random.default_rng(abs(hash((seed, tp, cond))) % (2 ** 31))
            xy = _lattice(n_side, srng); n = len(xy)

            bg_domain = _domains(xy, K_BGD, np.random.default_rng(bgd_rng.integers(1 << 30)))
            cell_type = np.array([srng.choice(K_CT, p=p_ct[d]) for d in bg_domain])

            sal_domain_cf = _domains(xy, K_SAD, np.random.default_rng(sad_rng.integers(1 << 30)))
            sal_state_cf = (srng.random(n) < q_ss[sal_domain_cf]).astype(int)

            # counterfactual labels retained on controls so the FP probes are defined
            sal_state = np.where(treated, sal_state_cf, 0)
            sal_domain = np.where(treated, sal_domain_cf, 0)

            z = A_ct[cell_type] + srng.normal(0, .35, (n, D_CT))
            z_sp = A_bgd[bg_domain] + srng.normal(0, .35, (n, D_BGD))
            s = delta * A_ss[sal_state] + srng.normal(0, .35, (n, D_SS))
            s_sp = delta * A_sd[sal_domain] + srng.normal(0, .35, (n, D_SD))

            h = np.tanh(np.concatenate([z, z_sp, s, s_sp], 1) @ W1 + b1)
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
        delta=float(delta), seed=int(seed), source="v3_rate_by_region",
        n_domains=K_SAD - 1, n_cells_per_slice=int(n_side ** 2),
        k_classes=dict(arch=K_CT, niche=K_BGD, sal_intr=K_SS, sal_dom=K_SAD),
        sal_rates=list(map(float, q_ss)), domain_ls=DOMAIN_LS,
        quadrants=dict(background_intrinsic="gt_arch", background_spatial="gt_niche",
                       salient_intrinsic="gt_sal_intrinsic",
                       salient_spatial="gt_sal_domain"),
        note="intrinsic = per-cell categorical drawn at a region-specific RATE; "
             "spatial = that region. Pooling neighbours' intrinsic state estimates "
             "the rate and therefore the region, which is exactly what "
             "GAT(neighbours' q_m) computes.",
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"simv3_d{delta:g}_s{seed}.h5ad")
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
    args = ap.parse_args()
    for s in args.seeds:
        for d in args.deltas:
            build(d, s, args.n_side, args.n_genes, args.out_dir)


if __name__ == "__main__":
    main()
