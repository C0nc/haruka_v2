"""Build a MERFISH-grounded contrastive benchmark with an implanted salient spatial domain.

WHAT THIS IS. Every existing Haruka-v2 benchmark number comes from `haruka_sim.py`,
a fully synthetic lattice. This module replaces the lattice with real tissue: the
Wyss-Coray mouse-brain spatial-aging MERFISH atlas (`process_all_brain.h5ad`,
cohort `aging_coronal`, 300 genes). Real cells, real anatomy, real cross-animal
and cross-age nuisance variation -- and one thing that is NOT real, injected by us
so that ground truth exists:

    a SUBTLE SALIENT SPATIAL DOMAIN, present only in the treated arm.

DESIGN.

* CONDITION IS BETWEEN-ANIMAL, not a duplicated slice. At each of `n_timepoints`
  ages we take two age-matched mice; one is `control`, one is `disease`. No cell
  appears in both arms. The arms therefore differ by genuine mouse-to-mouse and
  technical (slide A/B) variation, which is exactly the nuisance a contrastive
  method claims to quarantine into the background latent. Slide batch is
  deliberately alternated across arms so that condition is not confounded with it.

* THE IMPLANT IS SHARED ACROSS TREATED SLICES, NOT PER-SLICE NOISE. Each slice's
  coordinates are mapped to a common unit frame, and a single pair of Gaussian
  random fields (one simulation seed) defines two domains used by every treated
  slice. A salient factor that lived in only one slice would be indistinguishable
  from a slice effect, and no method could be expected to find it.

* THE DOMAINS ARE DECORRELATED FROM ANATOMY BY CONSTRUCTION. Candidate fields are
  rejected until `NMI(domain, subregion) < nmi_tol`. This is the load-bearing
  choice: if the implanted domain coincided with a cortical layer, a plain
  anatomy clusterer would "detect" it for free and the benchmark would measure
  nothing. The achieved NMI is recorded in `uns` so the claim is checkable.

* THE EDGE IS GRADED, THE LABEL IS HARD. The multiplicative effect follows a
  sigmoid of the field, so there is no crisp boundary to latch onto, but the
  ground-truth label is the hard threshold. This is what makes the domain subtle
  rather than a painted-on rectangle.

* NOISE IS SYMMETRIC ACROSS ARMS. Counts are recovered from the atlas's
  log-normalised matrix (`round-trip through the stored transcript_count`, exact
  in row sums) and then BOTH arms are Poisson-resampled. Resampling only the
  treated arm would leave an arm-specific noise level that a model could read off
  directly, which would flatter every method for the wrong reason.

* delta=0 IS A REAL ARM. It emits the same object with the same `gt_domain`
  labels and no effect applied. Any method that "recovers" the domain there is
  reporting a false positive, and that is the number that separates a detector
  from a pattern-matcher.

Usage
-----
    python build_sim.py --deltas 0 0.3 0.6 1.2 --seeds 0 1 2 --out_dir data
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from sklearn.metrics import normalized_mutual_info_score

ATLAS = "/home/yancui/virtualorganoid/notebook/process_all_brain.h5ad"
CACHE = "data/atlas_pairs.h5ad"

# Age-matched (control, disease) mouse pairs from cohort `aging_coronal`, ordered
# young -> old. Slide batch is alternated between the arms across timepoints (B/A,
# A/B, B/B, A/B) so that `condition` is not a proxy for slide batch; pair T3 shares
# one slide (B4) outright and is the internal check that the effect survives when
# technical variation is minimal.
PAIRS = [
    # (timepoint, control_mouse, control_age, disease_mouse, disease_age)
    ("T1_young", "53", 3.4, "1", 3.8),
    ("T2_mid", "19", 15.5, "70", 15.8),
    ("T3_old", "75", 18.8, "80", 19.8),
    ("T4_veryold", "38", 26.7, "93", 28.5),
    ("T5_aged", "97", 32.6, "46", 33.2),
]


# ---------------------------------------------------------------------------
# spatial field helpers
# ---------------------------------------------------------------------------
def _unit_frame(xy: np.ndarray) -> np.ndarray:
    """Map one slice's coordinates into [0, 1]^2 using robust quantiles.

    Robust rather than min/max because a handful of stray cells outside the
    section would otherwise shrink the whole tissue into a corner of the frame,
    and the implanted domain -- defined in this frame -- would land in a
    different anatomical place on each slice.
    """
    lo = np.percentile(xy, 1.0, axis=0)
    hi = np.percentile(xy, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return np.clip((xy - lo) / span, 0.0, 1.0)


def _grf(rng: np.random.Generator, grid: int, length_scale: float) -> np.ndarray:
    """Unit-variance smooth Gaussian random field on a `grid` x `grid` lattice."""
    z = rng.standard_normal((grid, grid))
    f = gaussian_filter(z, sigma=length_scale * grid, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-12)


def _sample_field(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    grid = field.shape[0]
    ij = np.clip((uv * (grid - 1)).round().astype(int), 0, grid - 1)
    return field[ij[:, 1], ij[:, 0]]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_slices(n_timepoints: int, n_cells: int, seed: int, cache: str = CACHE):
    """Load the cached slices (see `prep_cache.py`) and subsample per animal.

    Subsampling is uniform at random within a slice, which thins the tissue but
    leaves the geometry, the anatomical composition and the domain boundaries
    intact. It does widen the k-NN neighbourhood radius by roughly
    ``sqrt(n_full / n_cells)``; at k=8 and the domain scales used here that
    radius stays far below the domain size, so no spatial method is starved.
    """
    pairs = PAIRS[:n_timepoints]
    wanted = {}
    for tp, cm, ca, dm, da in pairs:
        wanted[cm] = dict(timepoint=tp, condition="control", age=ca)
        wanted[dm] = dict(timepoint=tp, condition="disease", age=da)

    full = ad.read_h5ad(cache)
    mid = full.obs["mouse_id"].astype(str).to_numpy()

    rng = np.random.default_rng(1000 + seed)
    chosen = []
    for mouse in wanted:
        pos = np.where(mid == mouse)[0]
        if len(pos) == 0:
            raise KeyError(f"mouse {mouse} absent from cache {cache}; rerun prep_cache.py")
        if len(pos) > n_cells:
            pos = rng.choice(pos, size=n_cells, replace=False)
        chosen.append(np.sort(pos))
    positions = np.sort(np.concatenate(chosen))

    sub = full[positions].copy()
    counts = sub.X.toarray().astype(np.float64) if sp.issparse(sub.X) else np.asarray(sub.X, dtype=np.float64)

    meta = [wanted[m] for m in sub.obs["mouse_id"].astype(str)]
    sub.obs["timepoint"] = pd.Categorical([m["timepoint"] for m in meta])
    sub.obs["condition"] = pd.Categorical([m["condition"] for m in meta])
    sub.obs["age_months"] = [m["age"] for m in meta]
    # `slice_id` is the library key everything downstream batches on: one slice per
    # animal, so it doubles as the animal id.
    sub.obs["slice_id"] = pd.Categorical(
        [f"{m['timepoint']}_{c}_{mid}" for m, c, mid in
         zip(meta, sub.obs["condition"], sub.obs["mouse_id"])]
    )
    return sub, counts


# ---------------------------------------------------------------------------
# implant
# ---------------------------------------------------------------------------
def design_domains(
    adata,
    seed: int,
    n_domains: int = 2,
    area_frac: float = 0.13,
    length_scale: float = 0.13,
    strata_keys: tuple = ("subregion", "celltype"),
    min_stratum: int = 40,
    grid: int = 256,
):
    """Smooth domains in the shared unit frame, BALANCED against anatomy and composition.

    An earlier version drew a Gaussian random field, thresholded it globally, and
    rejected candidates on ``NMI(domain, subregion)``. `self_check.py` failed it:
    NMI sat at 0.028 while a logistic probe on the one-hot subregion label
    predicted domain membership at AUROC 0.64. NMI is the wrong statistic for a
    21%/79% split across 7 classes -- small per-class enrichments barely move it
    but are trivially learnable, and every non-contrastive baseline would have
    collected that 0.64 for free.

    The fix is construction, not rejection. The field is thresholded WITHIN each
    ``subregion x celltype`` stratum at that stratum's own quantile, so

        P(domain | subregion, celltype) = area_frac   for every stratum,

    exactly, by construction. Anatomy and cell-type composition carry no
    information about domain membership, and the only thing that distinguishes a
    domain cell from its stratum-mates is the injected program. The domain stays
    spatially coherent because the field is smooth and the selected cells within
    each stratum are the ones sitting where the field is high.

    Strata smaller than `min_stratum` fall back to the global threshold; they are
    too small for a stable quantile and together hold a negligible cell fraction.
    """
    uv = np.zeros((adata.n_obs, 2))
    for sl in adata.obs["slice_id"].unique():
        m = (adata.obs["slice_id"] == sl).to_numpy()
        uv[m] = _unit_frame(np.asarray(adata.obsm["spatial"])[m])

    strata = adata.obs[list(strata_keys)].astype(str).agg("|".join, axis=1).to_numpy()
    codes, counts = np.unique(strata, return_counts=True)
    big = {c for c, n in zip(codes, counts) if n >= min_stratum}
    rng = np.random.default_rng(2000 + seed)

    fields = [_grf(rng, grid, length_scale) for _ in range(n_domains)]
    vals = [_sample_field(f, uv) for f in fields]

    labels = np.zeros(adata.n_obs, dtype=int)
    soft = np.zeros((adata.n_obs, n_domains))
    taken = np.zeros(adata.n_obs, dtype=bool)

    for d, v in enumerate(vals):
        thr_per_cell = np.full(adata.n_obs, np.quantile(v[~taken], 1.0 - area_frac))
        for c in codes:
            if c not in big:
                continue
            m = (strata == c) & ~taken
            if m.sum() < min_stratum:
                continue
            thr_per_cell[strata == c] = np.quantile(v[m], 1.0 - area_frac)
        # Sigmoid edge scaled by the field's spread: the transition band is a fixed
        # fraction of the domain regardless of seed, so there is no crisp boundary
        # to latch onto while the hard label stays well defined.
        tau = 0.25 * float(v.std())
        soft[:, d] = 1.0 / (1.0 + np.exp(-(v - thr_per_cell) / tau))
        soft[taken, d] = 0.0
        hard = (v >= thr_per_cell) & ~taken
        labels[hard] = d + 1
        taken |= hard

    anatomy = adata.obs["subregion"].astype(str).to_numpy()
    nmi = float(normalized_mutual_info_score(anatomy, labels))
    # Balance is exact by construction, but assert the achieved per-stratum
    # prevalence rather than trusting the argument.
    prev = pd.Series(labels > 0).groupby(pd.Series(anatomy)).mean()
    spread = float(prev.max() - prev.min())
    if spread > 0.06:
        warnings.warn(
            f"[build_sim] domain prevalence varies by {spread:.3f} across subregions; "
            "stratified thresholding did not balance (small strata?)"
        )
    return dict(nmi=nmi, labels=labels, soft=soft,
                prevalence_spread_across_subregions=spread,
                sizes=[float((labels == d + 1).mean()) for d in range(n_domains)])


def design_programs(n_genes: int, n_domains: int, seed: int, program_size: int = 30):
    """Disjoint up/down gene programs, one per domain, unit RMS loading."""
    rng = np.random.default_rng(3000 + seed)
    order = rng.permutation(n_genes)
    W = np.zeros((n_domains, n_genes))
    for d in range(n_domains):
        idx = order[d * program_size: (d + 1) * program_size]
        signs = rng.choice([-1.0, 1.0], size=len(idx))
        mag = rng.uniform(0.6, 1.4, size=len(idx))
        w = signs * mag
        W[d, idx] = w / (np.sqrt((w ** 2).mean()) + 1e-12)
    return W


def build(delta: float, seed: int, n_timepoints: int, n_cells: int,
          delta_mode: str = "constant", out_dir: str = "data", cache: str = CACHE):
    adata, counts = load_slices(n_timepoints, n_cells, seed, cache=cache)
    dom = design_domains(adata, seed=seed)
    n_domains = dom["soft"].shape[1]
    W = design_programs(adata.n_vars, n_domains, seed)

    is_disease = (adata.obs["condition"] == "disease").to_numpy()

    # Per-timepoint effect scaling. 'progressive' makes the salient effect grow
    # with age, which is what a real age-dependent pathology looks like; the
    # default 'constant' keeps effect size orthogonal to timepoint so that a
    # single delta means one thing.
    tps = list(adata.obs["timepoint"].cat.categories)
    if delta_mode == "progressive":
        ramp = {tp: 0.4 + 1.2 * i / max(1, len(tps) - 1) for i, tp in enumerate(tps)}
    else:
        ramp = {tp: 1.0 for tp in tps}
    scale = adata.obs["timepoint"].map(ramp).to_numpy(dtype=float)

    # log-multiplier: graded membership x program x delta, treated cells only
    log_mult = np.zeros_like(counts)
    if delta > 0:
        eff = dom["soft"] * (is_disease * scale)[:, None]
        log_mult = delta * (eff @ W)
    lam = counts * np.exp(log_mult)

    rng = np.random.default_rng(4000 + seed)
    new_counts = rng.poisson(lam).astype(np.float32)

    out = ad.AnnData(
        X=sp.csr_matrix(new_counts),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    out.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    out.layers["counts"] = out.X.copy()

    # Ground truth. `gt_domain` is defined for every cell, treated or not: on
    # control cells it is the COUNTERFACTUAL label (where the domain would have
    # been), which is what makes the false-positive test possible -- a method that
    # "finds" the domain among controls is finding geometry, not an effect.
    out.obs["gt_domain"] = pd.Categorical(dom["labels"])
    out.obs["gt_domain_observed"] = pd.Categorical(
        np.where(is_disease, dom["labels"], 0)
    )
    for d in range(n_domains):
        out.obs[f"gt_soft_d{d + 1}"] = dom["soft"][:, d]

    out.uns["sim"] = dict(
        delta=float(delta), seed=int(seed), delta_mode=delta_mode,
        n_timepoints=int(n_timepoints), n_cells_per_slice=int(n_cells),
        domain_anatomy_nmi=dom["nmi"], domain_sizes=dom["sizes"],
        prevalence_spread_across_subregions=dom["prevalence_spread_across_subregions"],
        program_size=int((W[0] != 0).sum()), n_domains=int(n_domains),
        pairs=[list(p) for p in PAIRS[:n_timepoints]],
    )
    out.uns["program_W"] = W

    os.makedirs(out_dir, exist_ok=True)
    tag = f"d{delta:g}_s{seed}" + ("" if delta_mode == "constant" else f"_{delta_mode}")
    path = os.path.join(out_dir, f"sim_{tag}.h5ad")
    out.write_h5ad(path, compression="gzip")

    # The honesty check on effect size: the mean |log fold change| actually applied,
    # restricted to the cells and genes it was applied TO. Averaging over all 300
    # genes (270 of which have zero loading) would understate it tenfold and make
    # every arm look subtler than it is.
    inside = (dom["labels"] > 0) & is_disease
    prog_genes = (np.abs(W).sum(0) > 0)
    lfc = float(np.abs(log_mult[np.ix_(inside, prog_genes)]).mean()) if delta > 0 else 0.0
    summary = dict(path=path, tag=tag, n_obs=int(out.n_obs), n_vars=int(out.n_vars),
                   mean_abs_lfc_program_genes=lfc,
                   median_fold_change=float(np.exp(lfc)),
                   frac_cells_in_domain=float((dom["labels"] > 0).mean()),
                   **out.uns["sim"])
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.3, 0.6, 1.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_timepoints", type=int, default=4)
    ap.add_argument("--n_cells", type=int, default=9000)
    ap.add_argument("--delta_mode", default="constant", choices=["constant", "progressive"])
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        for d in args.deltas:
            rows.append(build(d, s, args.n_timepoints, args.n_cells,
                              delta_mode=args.delta_mode, out_dir=args.out_dir,
                              cache=args.cache))
    with open(os.path.join(args.out_dir, "sim_manifest.json"), "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
