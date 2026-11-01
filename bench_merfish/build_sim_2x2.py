"""A benchmark shaped like Haruka's 2x2, not like a single salient domain.

WHY THIS REPLACES `build_sim.py`. `spatial_attach="both"` gives the model FOUR
latents, and they are a 2x2 over two independent questions -- is this variation
shared across conditions or specific to the treated arm, and does it come from
the cell itself or from its neighbourhood:

                       intrinsic (own expression)   spatial (neighbours)
    background   |     `background`                 `spatial`
    salient      |     `salient`                    `salient_spatial`

The first build supplied ground truth for only two of those four cells, and it put
a SPATIAL domain in the `salient` slot -- which under Haruka's own semantics is
the wrong slot, because `salient` is the condition-specific *cell-intrinsic*
latent and `salient_spatial` is the condition-specific *niche*. A method was being
asked to find a niche factor with its intrinsic latent.

So this module injects one identifiable factor per quadrant:

  gt_arch          background x intrinsic  -- REAL: the atlas's 7 anatomical
                   subregions. Not injected, not ours to design.

  gt_niche         background x spatial    -- sparse "sender" cells scattered
                   i.i.d. broadcast one of two ligand classes and express a
                   ligand gene block. A receiver's niche is the Gaussian-kernel
                   weighted ligand balance over its neighbours WITH ITSELF
                   EXCLUDED, and that drives a response program. Present in BOTH
                   arms, so it belongs to background. Because self is excluded and
                   sender marks are i.i.d. and drawn from a disjoint RNG stream, a
                   cell's own row is independent of its own niche -- the factor is
                   recoverable from neighbours and provably not from self. This is
                   `haruka_sim.py`'s NICHE-CONFOUND construction, kept because it
                   is the only part of the original simulator with a proof.

  gt_sal_intrinsic salient x intrinsic     -- a discrete cell state assigned
                   i.i.d. IN SPACE in the treated arm only, driving its own
                   program. Spatially i.i.d. on purpose: neighbourhood smoothing
                   cannot help here and actively hurts. The first build had no
                   such factor, which is why every metric rewarded smoothing and
                   the most heavily smoothed methods had the highest null floors.

  gt_sal_domain    salient x spatial       -- the implanted spatial domain from
                   `build_sim.py`, unchanged: smooth field, graded edge, balanced
                   within `subregion x celltype` so anatomy cannot predict it.

A method that scores well on the diagonal of the representation-by-truth matrix
and near zero off it has done what the 2x2 claims. A method with one
representation can at best cover one cell of it, and that is the comparison.

    python build_sim_2x2.py --deltas 0 0.6 1.2 2.4 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from build_sim import CACHE, PAIRS, design_domains, load_slices

#: relative weights of the four factors, so one `delta` scales them together and
#: "delta" means the same thing in every quadrant.
#: `niche` is the RECEIVER'S OWN RESPONSE to its neighbourhood, and it defaults to
#: ZERO on purpose. This is `haruka_sim.py`'s headline regime (rho = 0) and the
#: reason its independence theorem holds: with a response term, the niche is
#: written into the receiver's own row and a purely intrinsic latent can read it
#: off directly. The first version of this build set it to 1.0 and `self_check_2x2`
#: failed exactly there -- `gt_niche` was recoverable from a cell's own expression
#: at AUROC 0.862 when it must sit at chance. With the response off, the niche
#: lives only in the neighbours' ligand marks, so aggregating neighbours is the
#: only way to see it, which is the whole point of asking a spatial latent for it.
#: Set `--niche_response` above zero for the adversarial arm where the theorem is
#: deliberately broken.
FACTOR_WEIGHTS = dict(niche=0.0, sal_intrinsic=1.0, sal_domain=1.0)

PROGRAM_SIZE = 30
LIGAND_BLOCK = 20          # genes marking each ligand class
N_LIGAND_CLASSES = 2

#: The sender's mark SCALES WITH DELTA, like every other factor. It used to be a
#: fixed 2.0 on the reasoning that a marker is not a response -- but that made
#: `delta` mean something different in this quadrant than in the others: the
#: salient programs grew with delta while the ligand marks stood still, so the
#: niche got progressively swamped. Measured directly: seed 0 recovered the niche
#: at AUROC 0.652 at delta=1.2 and 0.502 at delta=2.4, on the same construction.
#: Scaling it keeps all four quadrants commensurate, which is what `delta` is for.
LIGAND_WEIGHT = 2.0

#: Sender density and pooling width are set so that the niche is a WELL-ESTIMATED
#: local average rather than shot noise. The first attempt used P_SENDER=0.15,
#: K_NICHE=30 and a Gaussian bandwidth taken from the 7th neighbour distance --
#: which decays so fast that only the nearest ~7 cells carry weight, so roughly
#: ONE sender determined each cell's score. `self_check_2x2` measured the
#: consequence: R^2 of the true score on the neighbour-mean of the ligand genes was
#: 0.04, and the label was unrecoverable from neighbours (AUROC 0.512) even though
#: the information was formally present. A factor no aggregator can estimate is not
#: a fair thing to ask the `spatial` latent for.
P_SENDER = 0.35
K_NICHE = 100              # neighbours pooled for the niche (~35 senders)
N_SAL_STATES = 3           # cell states for the intrinsic salient factor


def build_niche(adata, seed: int, k: int = K_NICHE):
    """Sender/receiver niche with the cell itself excluded.

    Returns the continuous niche score, the sender mask, the ligand class, and a
    tercile label. Self-exclusion is what makes the factor unrecoverable from a
    cell's own expression -- the property the `spatial` latent is supposed to be
    the only one able to see.
    """
    rng = np.random.default_rng(7000 + seed)
    n = adata.n_obs
    is_sender = rng.random(n) < P_SENDER
    ligand = np.where(is_sender, rng.integers(0, N_LIGAND_CLASSES, size=n), -1)

    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    sl = adata.obs["slice_id"].astype(str).to_numpy()
    score = np.zeros(n)
    for s in np.unique(sl):
        m = np.where(sl == s)[0]
        tree = cKDTree(xy[m])
        dist, idx = tree.query(xy[m], k=k + 1)
        # drop self (column 0) -- this is the whole construction
        dist, idx = dist[:, 1:], idx[:, 1:]
        # Bandwidth from the MIDPOINT of the window, not its 7th element, so the
        # kernel decays gently across the whole neighbourhood and all ~k cells
        # contribute. A tight kernel makes the score depend on one or two cells.
        bandwidth = np.median(dist[:, k // 2])
        w = np.exp(-0.5 * (dist / max(bandwidth, 1e-9)) ** 2)
        lg = ligand[m][idx]
        pos = (w * (lg == 0)).sum(1)
        neg = (w * (lg == 1)).sum(1)
        score[m] = (pos - neg) / np.maximum(pos + neg, 1e-9)

    label = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    return dict(score=score, is_sender=is_sender, ligand=ligand, label=label)


def build_programs(n_genes: int, seed: int, n_domains: int):
    """Disjoint gene blocks: ligand markers, niche response, salient state,
    salient domains. Disjoint so that a representation cannot pick up one factor
    through a gene it shares with another."""
    rng = np.random.default_rng(8000 + seed)
    order = rng.permutation(n_genes)
    cur = 0

    def take(k):
        nonlocal cur
        idx = order[cur:cur + k]
        cur += k
        return idx

    def prog(idx):
        w = rng.choice([-1.0, 1.0], size=len(idx)) * rng.uniform(0.6, 1.4, size=len(idx))
        v = np.zeros(n_genes)
        v[idx] = w / (np.sqrt((w ** 2).mean()) + 1e-12)
        return v

    ligand_genes = [take(LIGAND_BLOCK) for _ in range(N_LIGAND_CLASSES)]
    P = dict(
        ligand=[np.eye(n_genes)[g].sum(0) for g in ligand_genes],
        niche=prog(take(PROGRAM_SIZE)),
        sal_intrinsic=np.stack([prog(take(PROGRAM_SIZE)) for _ in range(N_SAL_STATES - 1)]),
        sal_domain=np.stack([prog(take(PROGRAM_SIZE)) for _ in range(n_domains)]),
    )
    P["_n_used"] = int(cur)
    return P


def build(delta: float, seed: int, n_timepoints: int, n_cells: int,
          out_dir: str = "data", cache: str = CACHE, niche_response: float | None = None):
    adata, counts = load_slices(n_timepoints, n_cells, seed, cache=cache)
    n_genes = adata.n_vars
    is_disease = (adata.obs["condition"] == "disease").to_numpy()

    dom = design_domains(adata, seed=seed)          # salient x spatial
    n_domains = dom["soft"].shape[1]
    niche = build_niche(adata, seed)                # background x spatial
    P = build_programs(n_genes, seed, n_domains)

    # Assigned to EVERY cell, but only acted on in the treated arm. Control cells
    # therefore carry a counterfactual state label, which is what makes the
    # false-positive probe defined -- with `0` on all controls the label has one
    # class and the probe returns NaN, i.e. no test at all.
    rng = np.random.default_rng(9000 + seed)
    sal_state = rng.integers(0, N_SAL_STATES, size=adata.n_obs)

    log_mult = np.zeros_like(counts)

    # --- ligand markers: senders express their class block, in BOTH arms. This is
    # what carries the niche into a NEIGHBOUR's aggregated expression. It is a
    # marker, not a response, so it is on whenever delta is, at fixed strength.
    if delta > 0:
        for c in range(N_LIGAND_CLASSES):
            m = niche["ligand"] == c
            log_mult[m] += delta * LIGAND_WEIGHT * P["ligand"][c]

        # --- background x spatial. Default weight 0: the niche is carried ONLY by
        # the neighbours' ligand marks, never by the receiver's own row.
        w_niche = FACTOR_WEIGHTS["niche"] if niche_response is None else niche_response
        if w_niche:
            log_mult += (delta * w_niche
                         * niche["score"][:, None] * P["niche"][None, :])

        # --- salient x intrinsic: treated arm, spatially i.i.d. cell state
        for s in range(1, N_SAL_STATES):
            m = (sal_state == s) & is_disease
            log_mult[m] += delta * FACTOR_WEIGHTS["sal_intrinsic"] * P["sal_intrinsic"][s - 1]

        # --- salient x spatial: the implanted domain, treated arm, graded edge
        eff = dom["soft"] * is_disease[:, None]
        log_mult += delta * FACTOR_WEIGHTS["sal_domain"] * (eff @ P["sal_domain"])

    lam = counts * np.exp(log_mult)
    new_counts = np.random.default_rng(4000 + seed).poisson(lam).astype(np.float32)

    out = ad.AnnData(X=sp.csr_matrix(new_counts), obs=adata.obs.copy(), var=adata.var.copy())
    out.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    out.layers["counts"] = out.X.copy()

    # --- the four ground truths, one per quadrant of the 2x2 -------------------
    out.obs["gt_arch"] = pd.Categorical(adata.obs["subregion"].astype(str))
    out.obs["gt_niche"] = pd.Categorical(niche["label"])
    out.obs["gt_sal_intrinsic"] = pd.Categorical(sal_state)
    out.obs["gt_sal_domain"] = pd.Categorical(dom["labels"])
    out.obs["niche_score"] = niche["score"]
    out.obs["is_sender"] = niche["is_sender"]
    # kept for the false-positive arm: on control cells these are the
    # COUNTERFACTUAL labels, i.e. the geometry with no effect applied
    out.obs["gt_sal_domain_cf"] = pd.Categorical(dom["labels"])

    out.uns["sim2x2"] = dict(
        delta=float(delta), seed=int(seed), n_timepoints=int(n_timepoints),
        n_cells_per_slice=int(n_cells), n_domains=int(n_domains),
        domain_anatomy_nmi=dom["nmi"],
        prevalence_spread_across_subregions=dom["prevalence_spread_across_subregions"],
        program_size=PROGRAM_SIZE, genes_used=int(P["_n_used"]),
        n_sal_states=int(N_SAL_STATES), p_sender=P_SENDER, k_niche=K_NICHE,
        factor_weights=dict(FACTOR_WEIGHTS,
                            niche=(FACTOR_WEIGHTS["niche"] if niche_response is None
                                   else niche_response)),
        quadrants=dict(background_intrinsic="gt_arch", background_spatial="gt_niche",
                       salient_intrinsic="gt_sal_intrinsic",
                       salient_spatial="gt_sal_domain"),
        pairs=[list(p) for p in PAIRS[:n_timepoints]],
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sim2x2_d{delta:g}_s{seed}.h5ad")
    out.write_h5ad(path, compression="gzip")

    summary = dict(path=path, n_obs=int(out.n_obs), n_vars=int(out.n_vars),
                   **{k: v for k, v in out.uns["sim2x2"].items() if k != "pairs"})
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.6, 1.2, 2.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_timepoints", type=int, default=4)
    ap.add_argument("--n_cells", type=int, default=9000)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--niche_response", type=float, default=None,
                    help="receiver response to its own niche; >0 deliberately breaks the "
                         "independence theorem (adversarial arm)")
    args = ap.parse_args()
    for s in args.seeds:
        for d in args.deltas:
            build(d, s, args.n_timepoints, args.n_cells,
                  out_dir=args.out_dir, cache=args.cache,
                  niche_response=args.niche_response)


if __name__ == "__main__":
    main()
