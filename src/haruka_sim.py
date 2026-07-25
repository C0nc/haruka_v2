"""NICHE-CONFOUND: a spatial-niche benchmark simulator for Haruka's spatial branch.

The question this simulator exists to answer is narrow and falsifiable: does the
SIMVI-style neighbour-attention branch recover a factor that a non-spatial model
*provably cannot*, and does it refrain from inventing one when none exists?

Everything here follows from one construction. A sparse set of "sender" cells is
scattered i.i.d. across the lattice and each sender broadcasts a ligand class.
A cell's NICHE ``v_i`` is a Gaussian-kernel-weighted average of its neighbours'
ligand marks with **the cell itself excluded** (``w_ii = 0``). In the headline
regime the cell does not respond to its own dose at all (``rho = 0``), so the
niche is written into the rows of a cell's NEIGHBOURS and never into its own.

    THEOREM (headline regime, rho = 0).  For every cell i, sender or receiver,
    x_i is a measurable function of (y_i, u_i, t_i, c_i, s_i, L_i, ell_i,
    slice_i, eps_i).  v_i = f({(s_j, L_j, p_j) : j != i}) with self excluded.
    The sender marks are i.i.d. across sites and drawn from an RNG stream
    disjoint from the one generating (y, u, t, c, ell), and (s_i, L_i) is
    independent of {(s_j, L_j)}_{j != i}.
        =>  x_i _||_ v_i exactly, and the Bayes-optimal R^2(x_i -> v_i) is 0
            for a model of ANY capacity.

Three consequences carry the whole benchmark:

1. The theorem holds **with the spatial-autocorrelation confounder ON**. Cell
   identity ``y_i`` is drawn from Gaussian random fields at range ``xi_u``, so a
   cell's neighbours strongly resemble it for reasons that have nothing to do
   with interaction. That does not couple ``x_i`` to ``v_i``, because the sender
   field is independent of the identity field. This is why an i.i.d.-in-space
   intrinsic factor is the *easy* end of a sweep here and not the default: without
   the confounder, any neighbourhood aggregator wins for free.
2. No cells need to be excluded from scoring. The sender / receiver split is
   reported as a DIAGNOSTIC -- a gap between them means something leaked.
3. Any positive ``R^2(own row -> v)`` is measured probe overfitting, and is
   reported as the empirical floor rather than suppressed.

Two exactness caveats, stated rather than hidden:

* Ambient contamination is drawn from the MODEL-IMPLIED slice profile
  ``softmax(beta_0 + delta_slice)``, not the empirical slice mean. The empirical
  mean is a function of every cell in the slice and would make ``x_i`` depend
  (at order 1/n) on other cells' data. Using the model-implied profile keeps the
  theorem exact. ``ambient_local_frac > 0`` convolves counts with a neighbour
  kernel and therefore manufactures niche signal out of nothing; it is
  adversarial-only, and any run that sets it is refused entry to the null arms.
* ``sender_field='clustered'`` makes sender occupancy spatially structured, so a
  cell's own sender status predicts its local sender density. The exact-zero
  guarantee is explicitly WITHDRAWN for that arm and it is likewise barred from
  the nulls.

Anti-target. ``gt_decoy`` is a spatially smooth field that drives the cell's OWN
expression cell-autonomously, with no cell-to-cell transfer. It has the same
spatial signature as a niche effect and is not one. A spatial latent that scores
high on the decoy while scoring low on the niche has been fooled by smoothness,
and that is a FAILURE regardless of every other number.

Usage
-----
    from haruka_sim import simulate_spatial_contrast, self_check

    adata = simulate_spatial_contrast(preset="small", seed=0)           # ~4.2k cells
    adata = simulate_spatial_contrast(preset="main", seed=0)            # ~30k cells
    rna, atac = simulate_spatial_contrast(preset="dev", with_atac=True)

    report = self_check(seed=0, verbose=True)   # validates the SIMULATOR itself

``self_check`` is the load-bearing test: it fits spatially blocked cross-validated
ridge probes predicting the true niche from (a) the cell's OWN expression and
(b) the MEAN OF ITS NEIGHBOURS' expression, and asserts own ~ 0 while neighbour
is clearly positive; and asserts that BOTH collapse to ~0 under the null
(``niche_strength=0``). If that fails the simulator is broken and every downstream
benchmark number is worthless.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

__all__ = [
    "simulate_spatial_contrast",
    "simulate_niche_confound",
    "self_check",
    "PRESETS",
    "GENE_BLOCK_FRACTIONS",
    "NULL_ARMS",
    "build_knn_index",
    "neighbor_mean",
    "gen_kernel_mean",
    "neighbor_mask_density",
    "frac_cross_slice_neighbors",
    "NBR_IDX_KEY",
    "NBR_MASK_KEY",
    "permute_graph_within_slice",
    "shuffle_truth",
    "resample_truth",
    "audit_independence",
    "TRUTH_KEYS",
]

# ---------------------------------------------------------------------------
# configuration tables
# ---------------------------------------------------------------------------

#: cells/slice = R * C * (1 - p_hole); both slices are forced to EXACTLY this many
#: cells because ``ContrastiveVAE._get_min_batch_size`` truncates the larger arm
#: every step, so unequal slices silently waste data (harness gotcha G6).
PRESETS: dict[str, dict[str, int]] = {
    "small": dict(R=48, C=48, n_genes=1600),    # ~4.2k cells total -- the default
    "dev": dict(R=64, C=64, n_genes=1600),      # ~7.5k
    "main": dict(R=128, C=128, n_genes=1600),   # ~30k, matches ARC-seq scale
    "scale": dict(R=224, C=224, n_genes=1600),  # ~92k, runtime / memory arm
}

#: Disjoint gene blocks. Fractions are the spec's G=1600 sizes divided by 1600, so
#: ``n_genes`` can be varied while the programme balance is preserved.
GENE_BLOCK_FRACTIONS: dict[str, float] = {
    "intrinsic": 320 / 1600,
    "alias": 240 / 1600,
    "ligand": 200 / 1600,
    "response": 120 / 1600,
    "salient": 200 / 1600,
    "lesion": 80 / 1600,
    "decoy": 120 / 1600,
    "housekeeping": 320 / 1600,
}

#: Generator-side null arms. N3 (permuted graph), N4 (shuffled truth) and
#: N5 (resampled truth) are fit-/score-time and have their own helpers.
NULL_ARMS: dict[str, dict[str, Any]] = {
    # N0 -- niche written nowhere; both arms must sit at the floor.
    "no_niche": dict(niche_strength=0.0, rho=0.0),
    # N1 -- THE KILLER. Exquisite spatial structure, zero interaction. Any rich
    # coherent "niche" recovered here is an artefact of smoothness.
    "mirage": dict(niche_strength=0.0, rho=0.0, xi_u=8.0),
    # N2 -- does the branch absorb a cell-autonomous smooth field?
    "decoy_only": dict(niche_strength=0.0, rho=0.0, lam_c=2.0),
}

#: obsm keys holding the continuous ground-truth factor blocks.
TRUTH_KEYS: dict[str, str] = {
    "F_int": "gt_intrinsic",
    "F_niche": "gt_niche",
    "F_sal": "gt_salient",
    "F_decoy": "gt_decoy",
}

NBR_IDX_KEY = "spatial_nbr_idx"
NBR_MASK_KEY = "spatial_nbr_mask"

#: Above this many (cell x gene) entries, the dense oracle blocks ``gt_eta`` and
#: ``gt_mu`` are not stored unless explicitly requested. 60M float32 = 240 MB each.
_AUTO_STORE_ENTRIES = 60_000_000

_ZI_BINS = 4096
_ZI_MAX = 14.0  # log1p(mu) can never plausibly exceed this at these depths

CODE_VERSION = "haruka_sim/1.0"


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softmax_rows(eta):
    """Row-wise softmax, in place on a float32 copy."""
    e = np.asarray(eta, dtype=np.float32)
    e = e - e.max(axis=1, keepdims=True)
    np.exp(e, out=e)
    e /= e.sum(axis=1, keepdims=True)
    return e


def _standardise_fields(f):
    """Standardise each field to mean 0 / sd 1 over its own lattice."""
    f = np.asarray(f, dtype=np.float64)
    m = f.reshape(f.shape[0], -1).mean(axis=1)[:, None, None]
    s = f.reshape(f.shape[0], -1).std(axis=1)[:, None, None]
    s = np.where(s < 1e-12, 1.0, s)
    return (f - m) / s


def _gauss_fields(rng, R, C, xi, n_fields):
    """``n_fields`` standardised Gaussian random fields at correlation range ``xi``.

    Drawn i.i.d. on a lattice padded by ``P = ceil(4*xi)`` on every side, smoothed
    with ``mode='wrap'``, then cropped back to the interior. The padding is what
    removes the boundary artefact: wrapping a field whose margin is itself fresh
    noise leaves the retained interior with the correct stationary covariance,
    where wrapping the bare R x C lattice would make opposite edges correlated.

    ``xi <= 0`` short-circuits to i.i.d. noise, which is the ``xi_u = 0`` end of
    the confounder sweep.
    """
    if xi <= 0:
        return _standardise_fields(rng.standard_normal((n_fields, R, C)))
    P = int(math.ceil(4.0 * xi))
    eps = rng.standard_normal((n_fields, R + 2 * P, C + 2 * P))
    out = np.empty((n_fields, R, C), dtype=np.float64)
    for k in range(n_fields):
        out[k] = gaussian_filter(eps[k], sigma=xi, mode="wrap")[P : P + R, P : P + C]
    return _standardise_fields(out)


def _scale_programme(loading, contrib, target_sd):
    """Rescale a programme's loadings so its per-gene contribution has ``target_sd``.

    The spec says "row-scaled to a target per-programme variance". Scaling every
    row to exactly the same variance would flatten gene-to-gene heterogeneity,
    which the count model needs, so a SINGLE scalar per programme is applied such
    that the mean per-gene variance of the contribution hits the target. The
    programme scale is then exactly interpretable and ``lam_*`` multiplies a
    unit-variance block, while individual genes still differ in how strongly they
    load. The realised scalar is recorded in ``uns['gt_loading_scales']``.
    """
    v = float(np.mean(np.var(contrib, axis=0)))
    if not np.isfinite(v) or v <= 0:
        return loading, 1.0
    s = float(target_sd) / math.sqrt(v)
    return loading * s, s


# ---------------------------------------------------------------------------
# spatial graphs
# ---------------------------------------------------------------------------


def _niche_kernel(coords, xi_v):
    """Row-normalised Gaussian kernel with the diagonal removed.

    ``w_ij = exp(-d_ij^2 / (2 xi_v^2))`` for ``0 < d_ij <= 3 xi_v``, ``w_ii = 0``,
    rows normalised to sum 1. Self-exclusion is the single load-bearing property:
    it is what makes the niche unreadable from the cell's own row.

    Returned as CSR so the 92k-cell arm stays inside a few hundred MB -- a dense
    kernel there would be 34 GB.
    """
    n = coords.shape[0]
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=3.0 * xi_v, output_type="ndarray")
    if pairs.shape[0] == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64), np.zeros(n, dtype=bool)
    d2 = ((coords[pairs[:, 0]] - coords[pairs[:, 1]]) ** 2).sum(1)
    w = np.exp(-d2 / (2.0 * xi_v * xi_v))
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    vals = np.concatenate([w, w])
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W.setdiag(0.0)
    W.eliminate_zeros()
    rs = np.asarray(W.sum(axis=1)).ravel()
    isolated = rs <= 0
    rs = np.where(isolated, 1.0, rs)
    Wn = sparse.diags(1.0 / rs) @ W
    return Wn.tocsr(), isolated


def _kernel_to_padded(W, width):
    """CSR kernel -> ``(idx, w)`` padded to ``width``, so the true graph is inspectable.

    ``width`` is passed in rather than derived per slice: slices have different
    maximum degrees (holes fall differently), and padding each to its own maximum
    produces arrays that cannot be concatenated into one ``obsm`` block.
    """
    W = W.tocsr()
    n = W.shape[0]
    width = max(int(width), 1)
    idx = np.zeros((n, width), dtype=np.int32)
    wt = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        a, b = W.indptr[i], W.indptr[i + 1]
        ii, ww = W.indices[a:b], W.data[a:b]
        if ii.size > width:  # keep the heaviest edges
            keep = np.argpartition(ww, -width)[-width:]
            ii, ww = ii[keep], ww[keep]
        idx[i, : ii.size] = ii
        wt[i, : ww.size] = ww
        if ii.size == 0:
            idx[i, 0] = i  # padded slots point at self and carry weight 0
    return idx, wt


def build_knn_index(
    adata,
    k=10,
    spatial_key="spatial",
    library_key="slice_id",
    remove_long_links=True,
    distance_percentile=99.0,
):
    """Self-contained k-NN index matching ``haruka_spatial.build_neighbor_index``.

    Same semantics, no scvi / scenvi / cellcharter import, so the simulator and
    its self-check stay lightweight:

    * ``idx`` holds GLOBAL row positions into ``adata``;
    * a cell is never its own neighbour;
    * neighbours never cross ``library_key``;
    * padded slots point at the cell itself and are marked invalid in ``mask``;
    * long links are trimmed at ``distance_percentile`` PER SECTION, so a sparse
      section is not thresholded against a dense one's spacing.

    This is the DELIBERATELY MISSPECIFIED fitting graph: generation uses a
    continuous Gaussian kernel at a per-seed randomised range, fitting uses k-NN.
    For the model run itself prefer ``haruka_spatial.build_neighbor_index`` so
    that the graph under test is literally the fork's own.
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n = coords.shape[0]
    idx = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, k))
    mask = np.zeros((n, k), dtype=np.float32)

    if library_key is None:
        groups = [np.arange(n)]
    else:
        lab = np.asarray(adata.obs[library_key]).astype(str)
        groups = [np.flatnonzero(lab == u) for u in pd.unique(lab)]

    for g in groups:
        if g.size <= 1:
            continue
        kk = int(min(k, g.size - 1))
        tree = cKDTree(coords[g])
        d, j = tree.query(coords[g], k=kk + 1)
        d, j = np.atleast_2d(d)[:, 1:], np.atleast_2d(j)[:, 1:]  # drop self
        valid = np.ones_like(d, dtype=bool)
        if remove_long_links and d.size:
            thr = np.percentile(d, distance_percentile)
            valid = d <= thr
        idx[g[:, None], np.arange(kk)[None, :]] = g[j]
        mask[g[:, None], np.arange(kk)[None, :]] = valid.astype(np.float32)
        # padded slots (kk < k, or trimmed edges) point at self, weight 0
        bad = ~valid
        if bad.any():
            r, c = np.nonzero(bad)
            idx[g[r], c] = g[r]

    adata.obsm[NBR_IDX_KEY] = idx.astype(np.int64)
    adata.obsm[NBR_MASK_KEY] = mask
    return idx, mask


def neighbor_mask_density(adata, mask_key=NBR_MASK_KEY):
    """Realised fraction of valid neighbour slots -- a collapsed graph is visible."""
    return float(np.asarray(adata.obsm[mask_key]).mean())


def frac_cross_slice_neighbors(adata, idx_key=NBR_IDX_KEY, library_key="slice_id"):
    """Tripwire T2: must be EXACTLY 0. Cheap, and catches the bug class early."""
    idx = np.asarray(adata.obsm[idx_key])
    mask = np.asarray(adata.obsm[NBR_MASK_KEY]) > 0.5
    lab = np.asarray(adata.obs[library_key]).astype(str)
    cross = (lab[idx] != lab[:, None]) & mask
    denom = max(int(mask.sum()), 1)
    return float(cross.sum() / denom)


def neighbor_mean(X, idx, mask=None, weights=None):
    """Mean of a cell's neighbours' rows of ``X``.

    ``mask`` (validity) and ``weights`` (kernel) may both be given; where both are
    absent this is a plain uniform mean over ``idx``. Cells with no valid
    neighbour get a zero row.
    """
    X = np.asarray(X, dtype=np.float32)
    idx = np.asarray(idx)
    n, k = idx.shape
    w = np.ones((n, k), dtype=np.float32) if weights is None else np.asarray(weights, np.float32)
    if mask is not None:
        w = w * (np.asarray(mask, np.float32) > 0.5)
    denom = w.sum(1, keepdims=True)
    denom[denom <= 0] = 1.0
    out = np.zeros((n, X.shape[1]), dtype=np.float32)
    step = max(1, int(2**22 // max(X.shape[1], 1)))
    for a in range(0, n, step):
        b = min(a + step, n)
        out[a:b] = np.einsum("nk,nkg->ng", w[a:b], X[idx[a:b]], optimize=True)
    out /= denom
    return out


def gen_kernel_mean(adata, X):
    """Neighbour mean under the TRUE generative kernel (baseline arm (e), the ceiling).

    Separates "does the mechanism work" from "is the fitted graph right". If this
    is low, the arm is DEAD and must be discarded rather than reported as a model
    failure.
    """
    return neighbor_mean(X, adata.obsm["gt_gen_nbr_idx"], weights=adata.obsm["gt_gen_nbr_w"])


# ---------------------------------------------------------------------------
# the simulator
# ---------------------------------------------------------------------------


def simulate_spatial_contrast(
    preset: str = "small",
    seed: int = 0,
    *,
    # ---- geometry -------------------------------------------------------
    R: int | None = None,
    C: int | None = None,
    p_hole: float = 0.08,
    geometry: str = "lattice",
    spot_pitch: float = 1.0,
    n_rep: int = 1,
    # ---- genes ----------------------------------------------------------
    n_genes: int | None = None,
    # ---- factor 1: intrinsic (the confounder) ---------------------------
    K_type: int = 6,
    xi_u: float = 4.0,
    d_u: int = 4,
    sigma_u: float = 0.5,
    lam_int: float = 1.5,
    # ---- factor 2: niche (the target) -----------------------------------
    d_v: int = 3,
    p_send: float = 0.12,
    xi_v: float | None = None,
    niche_strength: float = 1.0,
    reception: str = "additive",
    kappa_R: float = 0.5,
    sender_field: str = "iid",
    rho: float = 0.0,
    pi_alias: float = 0.0,
    lam_v: float = 1.0,
    cross_slice_tripwire: bool = True,
    # ---- factor 3: salient ----------------------------------------------
    d_t: int = 3,
    sigma_t: float = 0.6,
    lam_sal: float = 1.5,
    lesion_radius_frac: float = 0.18,
    salient_control_leak: float = 0.0,
    lam_les: float = 1.0,
    # ---- factor 4: decoy (the anti-target) ------------------------------
    d_c: int = 2,
    lam_c: float = 0.6,
    xi_c: float | None = None,
    # ---- count model ----------------------------------------------------
    lib_size: float = 3000.0,
    lib_sigma: float = 0.35,
    theta_mu: float = 2.0,
    theta_sigma: float = 0.6,
    zi_target: float = 0.80,
    ambient_frac: float = 0.02,
    ambient_local_frac: float = 0.0,
    depth_confound: bool = False,
    tau_slice: float = 0.15,
    prog_sd: float = 1.0,
    # ---- ATAC -----------------------------------------------------------
    with_atac: bool = False,
    n_peaks: int = 2000,
    d_m: int = 2,
    atac_reassign_frac: float = 0.2,
    atac_mean: float = 0.2,
    # Poisson at mean 0.2, spread compositionally, is already ~0.90 zeros on its
    # own, so a target below that would leave the per-modality dropout inert.
    atac_zi_target: float = 0.93,
    # ---- fitting graph --------------------------------------------------
    k_fit: int = 10,
    remove_long_links: bool = True,
    distance_percentile: float = 99.0,
    # ---- storage --------------------------------------------------------
    store_eta: bool | None = None,
    store_mu: bool | None = None,
    store_clean: bool = True,
    store_gen_kernel: bool = True,
    # ---- arms / audit ---------------------------------------------------
    null_arm: str | None = None,
    audit: bool = True,
    audit_max_cells: int = 5000,
    verbose: bool = False,
    **overrides,
):  # noqa: C901
    """Simulate a paired control / disease spatial-transcriptomics contrast.

    Returns an :class:`anndata.AnnData` with integer counts in ``.X`` and
    ``.layers['counts']``, or ``(rna, atac)`` when ``with_atac=True``.

    Deterministic given ``seed``: the whole draw hangs off one
    ``np.random.SeedSequence(seed)`` spawned into named, never-reused streams.
    The global ``np.random`` state is never touched.

    Key knobs
    ---------
    niche_strength
        Carrier amplitude ``lam_lig``. **Set to 0.0 for the NULL CONTROL**: the
        niche is then written nowhere and both the own-row and the neighbour-row
        probes must collapse to the floor. Also the dose-response sweep axis; the
        response must rise gradedly FROM ~0 rather than switch on.

        **The dose-response is NOT monotone over an unbounded range, and that is a
        property of the count model rather than a defect.** Counts are
        compositional (``mu = ell * softmax(eta)``), so a sender's ligand block
        competes for a fixed budget. Measured on the ``small`` preset, seed 0, with
        the true-kernel neighbour probe::

            lam_lig      0     0.25    0.5     1.0     2.0     4.0     8.0
            R^2       -0.06   -0.04   +0.38   +0.49   +0.39   +0.35   +0.39
            lig-share  0.061   0.063  0.074   0.112   0.152   0.167   0.170
            corr(L, ligand-block total, senders only)
                       0.02    0.49   0.76    0.70    0.38    0.11    0.03

        The fall is NOT counting noise: it persists on the noiseless compositional
        mean (0.89 -> 0.80 -> 0.82 -> 0.68). The mechanism is softmax saturation.
        Once a sender's ligand genes dominate its own budget the block's share
        plateaus (~0.17), and the block stops encoding the GRADED load ``L_i`` --
        it converges on whichever gene has the largest ``A_lig`` entry regardless
        of amplitude. Since ``v_i`` is a kernel-weighted average of neighbours'
        ``L_i``-weighted marks, losing ``L_i`` costs niche recoverability directly.

        Practical consequence: **sweep ``{0, 0.25, 0.5, 1.0, 2.0}`` and read
        monotonicity on ``[0, 1]``**, where the estimand is genuinely graded. The
        default 1.0 sits at the peak. Treat ``>= 2`` as a saturation arm, not as a
        stronger-signal arm.
    xi_u
        Spatial autocorrelation range of the INTRINSIC factor -- the confounder.
        ``0`` = i.i.d. cell types (easy); ``8`` = large coherent patches where a
        cell's neighbours are nearly a copy of it for reasons that are not
        interaction. Requirement across the sweep: ``Delta_niche`` must not
        degrade materially and intrinsic leakage must not grow with ``xi_u``.
    rho, pi_alias
        Own-row self-readability of the niche. ``rho=0`` (default) is the
        HEADLINE regime with Bayes ``R^2(x_i -> v_i) = 0`` exactly.
        ``rho>0, pi_alias=0`` gives the niche private marker genes (easy control);
        ``rho>0, pi_alias=1`` puts the niche's own-cell effect exactly inside the
        column space of the intrinsic loadings (hardest realistic case).
    reception
        ``'additive'`` (default) makes the niche a linear functional of
        neighbours, so a uniform neighbour mean is near-Bayes-optimal and PARITY
        with the k-NN-mean baseline is the pre-registered PASS. ``'competitive'``
        makes it a ratio, where normalised pooling is the matched inductive bias.
    null_arm
        ``'no_niche'`` (N0), ``'mirage'`` (N1), ``'decoy_only'`` (N2). Sets the
        corresponding parameters and records the arm in ``uns``.

    Notes
    -----
    ``ambient_local_frac > 0`` and ``sender_field='clustered'`` both manufacture
    niche-like signal inside a cell's own row. They are adversarial arms: the
    exact-zero guarantee is withdrawn, ``uns['headline_eligible']`` is set False,
    and combining either with a ``null_arm`` raises.
    """
    t_start = _perf()

    # ---- aliases, and a hard stop on anything unrecognised ---------------
    # The spec's signature ends in **overrides, but silently swallowing unknown
    # keywords is the wrong failure mode for a benchmark generator: a typo such as
    # `niche_strengh=0` would hand back a FULL-SIGNAL dataset labelled as a null
    # and every downstream number would be quietly wrong. Documented aliases are
    # mapped; everything else raises.
    _ALIASES = {
        "intrinsic_autocorr": "xi_u",   # the task brief's name for the confounder knob
        "lam_lig": "niche_strength",    # the spec's internal name for the carrier
        "niche_range": "xi_v",
        "decoy_range": "xi_c",
    }
    for key in list(overrides):
        if key not in _ALIASES:
            raise TypeError(
                f"simulate_spatial_contrast() got an unexpected keyword {key!r}. "
                f"Known aliases: {sorted(_ALIASES)}."
            )
        target = _ALIASES[key]
        val = overrides.pop(key)
        if target == "xi_u":
            xi_u = val
        elif target == "niche_strength":
            niche_strength = val
        elif target == "xi_v":
            xi_v = val
        elif target == "xi_c":
            xi_c = val

    # ---- resolve arms and presets ---------------------------------------
    params_requested = dict(
        preset=preset, seed=seed, R=R, C=C, p_hole=p_hole, geometry=geometry,
        n_rep=n_rep, n_genes=n_genes, K_type=K_type, xi_u=xi_u, d_u=d_u,
        sigma_u=sigma_u, lam_int=lam_int, d_v=d_v, p_send=p_send, xi_v=xi_v,
        niche_strength=niche_strength, reception=reception, kappa_R=kappa_R,
        sender_field=sender_field, rho=rho, pi_alias=pi_alias, lam_v=lam_v,
        d_t=d_t, sigma_t=sigma_t, lam_sal=lam_sal,
        lesion_radius_frac=lesion_radius_frac,
        salient_control_leak=salient_control_leak, lam_les=lam_les, d_c=d_c,
        lam_c=lam_c, xi_c=xi_c, lib_size=lib_size, lib_sigma=lib_sigma,
        theta_mu=theta_mu, theta_sigma=theta_sigma, zi_target=zi_target,
        ambient_frac=ambient_frac, ambient_local_frac=ambient_local_frac,
        depth_confound=depth_confound, tau_slice=tau_slice, prog_sd=prog_sd,
        with_atac=with_atac, n_peaks=n_peaks, k_fit=k_fit, null_arm=null_arm,
        cross_slice_tripwire=cross_slice_tripwire,
    )
    params_requested.update(overrides)

    if null_arm is not None:
        if null_arm not in NULL_ARMS:
            raise ValueError(f"unknown null_arm {null_arm!r}; choose from {sorted(NULL_ARMS)}")
        if ambient_local_frac > 0 or sender_field != "iid":
            raise ValueError(
                f"null_arm={null_arm!r} refused: ambient_local_frac>0 and "
                "sender_field='clustered' both manufacture niche-like signal in a "
                "cell's own row and would invalidate the null."
            )
        arm = NULL_ARMS[null_arm]
        niche_strength = arm.get("niche_strength", niche_strength)
        rho = arm.get("rho", rho)
        xi_u = arm.get("xi_u", xi_u)
        lam_c = arm.get("lam_c", lam_c)

    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        pr = PRESETS[preset]
        R = pr["R"] if R is None else R
        C = pr["C"] if C is None else C
        n_genes = pr["n_genes"] if n_genes is None else n_genes
    if R is None or C is None or n_genes is None:
        raise ValueError("give a preset, or all of R, C and n_genes")
    if reception not in ("additive", "competitive"):
        raise ValueError("reception must be 'additive' or 'competitive'")
    if sender_field not in ("iid", "clustered"):
        raise ValueError("sender_field must be 'iid' or 'clustered'")
    if geometry not in ("lattice", "jittered"):
        raise ValueError("geometry must be 'lattice' or 'jittered'")

    lam_lig = float(niche_strength)
    headline_eligible = (ambient_local_frac <= 0) and (sender_field == "iid")

    # ---- RNG streams ----------------------------------------------------
    # Named once, never reused across factors. Spawning 10 leaves the first 8
    # bit-identical to the spec's 8-stream layout, because SeedSequence children
    # are produced in a fixed order.
    ss = np.random.SeedSequence(seed)
    st = ss.spawn(10)
    (ss_intrinsic, ss_sender, ss_salient, ss_decoy, ss_library,
     ss_counts, ss_atac, ss_dropout, ss_load, ss_geom) = st

    rng_load = np.random.default_rng(ss_load)
    rng_lib = np.random.default_rng(ss_library)
    rng_counts = np.random.default_rng(ss_counts)
    rng_drop = np.random.default_rng(ss_dropout)

    n_slices = 2 * int(n_rep)
    sub_intr = ss_intrinsic.spawn(n_slices)
    sub_send = ss_sender.spawn(n_slices)
    sub_sal = ss_salient.spawn(n_slices)
    sub_decoy = ss_decoy.spawn(n_slices)
    sub_geom = ss_geom.spawn(n_slices)

    # ---- shared parameters (drawn ONCE, identical across slices) --------
    if xi_v is None:
        xi_v = float(rng_load.uniform(1.2, 2.0))  # randomised per seed, stored in uns
    xi_v = float(xi_v)
    xi_c = float(xi_v) if xi_c is None else float(xi_c)
    xi_les = 4.0 * xi_v

    G = int(n_genes)
    blocks = _gene_blocks(G, rng_load)
    mu_type = rng_load.standard_normal((K_type, d_u)) / math.sqrt(d_u)
    b_dir = rng_load.standard_normal(d_t)
    b_dir /= np.linalg.norm(b_dir)
    M_alias = rng_load.standard_normal((d_u, d_v))
    beta_0 = rng_load.standard_normal(G)
    theta_g = np.exp(math.log(theta_mu) + theta_sigma * rng_load.standard_normal(G))
    delta_slice = tau_slice * rng_load.standard_normal((n_slices, G))

    # ---- geometry -------------------------------------------------------
    n_keep = int(round(R * C * (1.0 - p_hole)))
    if n_keep < 16:
        raise ValueError("lattice too small after holes")
    slice_meta = []
    for si in range(n_slices):
        cond = "control" if si < n_rep else "disease"
        rep = si % n_rep
        name = f"{cond}_s{rep + 1}"
        rgeom = np.random.default_rng(sub_geom[si])
        keep = np.sort(rgeom.choice(R * C, size=n_keep, replace=False))
        rr, cc = np.divmod(keep, C)
        xy = np.stack([cc, rr], axis=1).astype(np.float64)
        if geometry == "jittered":
            xy = xy + rgeom.uniform(-0.35, 0.35, size=xy.shape)
        xy *= spot_pitch
        slice_meta.append(dict(name=name, cond=cond, rep=rep, keep=keep, rr=rr, cc=cc, xy=xy))

    n_total = n_slices * n_keep
    if verbose:
        print(f"[haruka_sim] {n_slices} slices x {n_keep} cells = {n_total}, G={G}, "
              f"xi_u={xi_u}, xi_v={xi_v:.3f}, lam_lig={lam_lig}")

    # ---- factor 1: intrinsic (spatially autocorrelated) -----------------
    y_all, u_all = [], []
    for si, sm in enumerate(slice_meta):
        r = np.random.default_rng(sub_intr[si])
        f = _gauss_fields(r, R, C, xi_u, K_type)              # (K, R, C)
        flat = f.reshape(K_type, -1)[:, sm["keep"]]
        y = flat.argmax(axis=0).astype(np.int32)
        u = mu_type[y] + sigma_u * r.standard_normal((n_keep, d_u))
        y_all.append(y)
        u_all.append(u)
    y_i = np.concatenate(y_all)
    u_i = np.concatenate(u_all).astype(np.float64)

    # ---- factor 2: sender marks (i.i.d.) and niche ----------------------
    # i.i.d. sender occupancy is load-bearing. Any spatially clustered occupancy
    # makes s_i predict the local sender density and voids the theorem.
    s_full = np.zeros((n_slices, R * C), dtype=np.int8)
    L_full = np.zeros((n_slices, R * C), dtype=np.float64)
    for rep in range(n_rep):
        ci, di = rep, n_rep + rep
        rc = np.random.default_rng(sub_send[ci])
        rd = np.random.default_rng(sub_send[di])
        if sender_field == "clustered":
            # ADVERSARIAL ARM -- exact-zero guarantee explicitly withdrawn.
            fc = _gauss_fields(rc, R, C, 2.0 * xi_v, 1)[0].ravel()
            occ_c = fc >= np.quantile(fc, 1.0 - p_send)
            fd = _gauss_fields(rd, R, C, 2.0 * xi_v, 1)[0].ravel()
            occ_d = fd >= np.quantile(fd, 1.0 - p_send)
        else:
            occ_c = rc.random(R * C) < p_send
            if cross_slice_tripwire:
                # T1: disease occupancy ANTI-correlated with control at matched
                # (r, c). Each field stays marginally i.i.d. and independent of
                # (u, t, c) within its own slice, but if the fitted graph leaks
                # across slices, pooled neighbourhoods mix v_i with -v_i' and the
                # niche R^2 collapses by a large, specific amount.
                p_dis = p_send * (1.0 - 0.9 * occ_c.astype(np.float64))
            else:
                p_dis = np.full(R * C, p_send)
            occ_d = rd.random(R * C) < p_dis
        for si, r, occ in ((ci, rc, occ_c), (di, rd, occ_d)):
            cls = r.integers(1, d_v + 1, size=R * C)
            s_full[si] = np.where(occ, cls, 0).astype(np.int8)
            L_full[si] = np.where(occ, r.gamma(shape=4.0, scale=0.25, size=R * C), 0.0)

    s_all, L_all, v_all, kernels, isolated_all = [], [], [], [], []
    for si, sm in enumerate(slice_meta):
        s_i = s_full[si][sm["keep"]].astype(np.int32)
        L_i = L_full[si][sm["keep"]]
        ell_mark = np.zeros((n_keep, d_v))
        nz = s_i > 0
        ell_mark[nz, s_i[nz] - 1] = L_i[nz]

        W, isolated = _niche_kernel(sm["xy"], xi_v)
        a = W @ ell_mark                                   # self excluded by w_ii=0
        if reception == "competitive":
            vt = a / (kappa_R + a.sum(axis=1, keepdims=True))
        else:
            vt = a
        s_all.append(s_i)
        L_all.append(L_i)
        v_all.append(vt)
        isolated_all.append(isolated)
        if store_gen_kernel:
            kernels.append(W)

    gen_idx, gen_w = [], []
    if store_gen_kernel and kernels:
        # one width for every slice, or the blocks will not concatenate
        width = max(int(np.diff(W_.tocsr().indptr).max()) for W_ in kernels)
        for si, W_ in enumerate(kernels):
            gi, gw = _kernel_to_padded(W_, width)
            gen_idx.append(gi + si * n_keep)  # global row positions into adata
            gen_w.append(gw)

    s_i = np.concatenate(s_all)
    L_i = np.concatenate(L_all)
    v_raw = np.concatenate(v_all)
    v_mean, v_sd = v_raw.mean(0), v_raw.std(0)
    v_sd = np.where(v_sd < 1e-12, 1.0, v_sd)
    v_i = (v_raw - v_mean) / v_sd                          # column-standardised

    # ---- factor 3: salient (+ lesion niche) -----------------------------
    t_i = np.zeros((n_total, d_t))
    dose = np.zeros(n_total)
    u_les = np.zeros(n_total)
    for si, sm in enumerate(slice_meta):
        lo, hi = si * n_keep, (si + 1) * n_keep
        r = np.random.default_rng(sub_sal[si])
        nu = r.standard_normal((n_keep, d_t))
        if sm["cond"] == "disease":
            p0 = np.array([r.uniform(0.25 * C, 0.75 * C), r.uniform(0.25 * R, 0.75 * R)])
            r0 = lesion_radius_frac * R
            a_i = np.exp(-((sm["xy"] / spot_pitch - p0) ** 2).sum(1) / (2.0 * r0 * r0))
            # nu is the NON-SPATIAL salient axis and is mandatory: without it the
            # salient factor is a pure function of position, and the salient and
            # spatial latents can trade places undetected.
            t_i[lo:hi] = a_i[:, None] * b_dir[None, :] + sigma_t * nu
            dose[lo:hi] = a_i
            emit = (a_i > 0.5).astype(np.float64)
            u_les[lo:hi] = _diffuse_on_lattice(emit, sm, R, C, xi_les)
        else:
            # HONEST ARM: leak>0 breaks the exact match between the salient truth
            # and ContrastiveVI's hard zeroing of s on background cells.
            t_i[lo:hi] = salient_control_leak * sigma_t * nu
    if u_les.std() > 1e-12:
        dmask = np.repeat([sm["cond"] == "disease" for sm in slice_meta], n_keep)
        if dmask.any() and u_les[dmask].std() > 1e-12:
            u_les[dmask] = (u_les[dmask] - u_les[dmask].mean()) / u_les[dmask].std()

    # ---- factor 4: decoy (anti-target) ----------------------------------
    c_all = []
    for si, sm in enumerate(slice_meta):
        r = np.random.default_rng(sub_decoy[si])
        f = _gauss_fields(r, R, C, xi_c, d_c)
        c_all.append(f.reshape(d_c, -1)[:, sm["keep"]].T)
    c_i = np.concatenate(c_all)

    # ---- library size (independent of every factor AND of slice) --------
    # Load-bearing: use_observed_lib_size=True reads log(x.sum(1)) and slice is
    # the batch covariate, so any depth-factor coupling would be absorbed by a
    # path that has nothing to do with the science.
    ell = np.exp(math.log(lib_size) + lib_sigma * rng_lib.standard_normal(n_total))
    slice_of = np.repeat(np.arange(n_slices), n_keep)
    is_disease = np.repeat([sm["cond"] == "disease" for sm in slice_meta], n_keep)
    if depth_confound:
        ell = np.where(is_disease, ell * 0.85, ell)
    if n_rep > 1:
        ell = ell * np.exp(0.1 * rng_lib.standard_normal(n_slices))[slice_of]

    # ---- loadings, programme-scaled -------------------------------------
    A_int = rng_load.standard_normal((K_type, blocks["intrinsic"].size))
    B = rng_load.standard_normal((blocks["intrinsic"].size, d_u))
    B_alias = rng_load.standard_normal((blocks["alias"].size, d_u))
    A_lig = np.vstack([
        np.zeros((1, blocks["ligand"].size)),
        rng_load.standard_normal((d_v, blocks["ligand"].size)),
    ])
    Cc = rng_load.standard_normal((blocks["response"].size, d_v))
    E = rng_load.standard_normal((blocks["salient"].size, d_t))
    A_les = rng_load.standard_normal(blocks["lesion"].size)
    D = rng_load.standard_normal((blocks["decoy"].size, d_c))

    onehot_y = np.zeros((n_total, K_type))
    onehot_y[np.arange(n_total), y_i] = 1.0
    scales = {}
    A_int, scales["A_int"] = _scale_programme(A_int, onehot_y @ A_int, prog_sd)
    B, scales["B"] = _scale_programme(B, u_i @ B.T, prog_sd)
    B_alias, scales["B_alias"] = _scale_programme(B_alias, u_i @ B_alias.T, prog_sd)
    lig_contrib = A_lig[s_i] * L_i[:, None]
    A_lig, scales["A_lig"] = _scale_programme(A_lig, lig_contrib, prog_sd)
    Cc, scales["C"] = _scale_programme(Cc, v_i @ Cc.T, prog_sd)
    E, scales["E"] = _scale_programme(E, t_i @ E.T, prog_sd)
    A_les, scales["A_les"] = _scale_programme(A_les, u_les[:, None] * A_les[None, :], prog_sd)
    D, scales["D"] = _scale_programme(D, c_i @ D.T, prog_sd)

    # the aliased niche lives INSIDE the intrinsic column space: eta depends on
    # (u_i, v_i) only through the sum u_i + rho*pi_alias*lam_v*M v_i
    u_eff = u_i + (rho * pi_alias * lam_v) * (v_i @ M_alias.T)
    resp_gain = rho * (1.0 - pi_alias) * lam_v

    def _eta_chunk(lo, hi):
        m = hi - lo
        eta = np.tile(beta_0.astype(np.float32), (m, 1))
        eta += delta_slice[slice_of[lo:hi]].astype(np.float32)
        bi = blocks["intrinsic"]
        eta[:, bi] += (lam_int * A_int[y_i[lo:hi]] + u_i[lo:hi] @ B.T).astype(np.float32)
        ba = blocks["alias"]
        eta[:, ba] += (u_eff[lo:hi] @ B_alias.T).astype(np.float32)
        if lam_lig != 0.0:
            bl = blocks["ligand"]
            eta[:, bl] += (lam_lig * A_lig[s_i[lo:hi]] * L_i[lo:hi, None]).astype(np.float32)
        if resp_gain != 0.0:
            br = blocks["response"]
            eta[:, br] += (resp_gain * (v_i[lo:hi] @ Cc.T)).astype(np.float32)
        d_here = is_disease[lo:hi]
        if d_here.any():
            bs = blocks["salient"]
            eta[np.ix_(d_here, bs)] += (lam_sal * (t_i[lo:hi][d_here] @ E.T)).astype(np.float32)
            ble = blocks["lesion"]
            eta[np.ix_(d_here, ble)] += (
                lam_les * u_les[lo:hi][d_here, None] * A_les[None, :]
            ).astype(np.float32)
        bd = blocks["decoy"]
        eta[:, bd] += (lam_c * (c_i[lo:hi] @ D.T)).astype(np.float32)
        return eta

    # ---- counts ---------------------------------------------------------
    p_ambient = _softmax_rows(beta_0[None, :] + delta_slice)  # model-implied, exact
    chunk = int(max(1, min(n_total, (1 << 22) // G)))
    want_eta = (n_total * G <= _AUTO_STORE_ENTRIES) if store_eta is None else bool(store_eta)
    want_mu = (n_total * G <= _AUTO_STORE_ENTRIES) if store_mu is None else bool(store_mu)

    counts_clean = np.zeros((n_total, G), dtype=np.int32)
    eta_store = np.zeros((n_total, G), dtype=np.float32) if want_eta else None
    mu_store = np.zeros((n_total, G), dtype=np.float32) if want_mu else None
    theta_row = theta_g[None, :]
    hist = np.zeros(_ZI_BINS, dtype=np.float64)
    edges = np.linspace(0.0, _ZI_MAX, _ZI_BINS + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    n_zero_nb = 0

    for lo in range(0, n_total, chunk):
        hi = min(lo + chunk, n_total)
        eta = _eta_chunk(lo, hi)
        mu = (ell[lo:hi, None] * _softmax_rows(eta)).astype(np.float64)
        if eta_store is not None:
            eta_store[lo:hi] = eta
        if mu_store is not None:
            mu_store[lo:hi] = mu.astype(np.float32)
        # NB(mu, theta) == Poisson(Gamma(shape=theta, scale=mu/theta)); this is the
        # same parameterisation scvi.distributions.NegativeBinomial uses.
        lam = rng_counts.gamma(shape=np.broadcast_to(theta_row, mu.shape), scale=mu / theta_row)
        xc = rng_counts.poisson(lam).astype(np.int32)
        counts_clean[lo:hi] = xc
        pos = xc > 0
        n_zero_nb += int(xc.size - pos.sum())
        if pos.any():
            hist += np.histogram(np.log1p(mu[pos]), bins=edges)[0]
        del eta, mu, lam, xc

    a0 = _solve_zi_intercept(hist, centers, n_zero_nb, n_total * G, zi_target)

    X = np.zeros((n_total, G), dtype=np.int32)
    n_zero_post_zi = 0
    for lo in range(0, n_total, chunk):
        hi = min(lo + chunk, n_total)
        if mu_store is not None:
            mu = mu_store[lo:hi].astype(np.float64)
        else:
            mu = (ell[lo:hi, None] * _softmax_rows(_eta_chunk(lo, hi))).astype(np.float64)
        pi = _sigmoid(a0 - np.log1p(mu))
        keep = rng_drop.random(mu.shape) >= pi
        x = counts_clean[lo:hi] * keep
        n_zero_post_zi += int((x == 0).sum())
        if ambient_frac > 0:
            x = _apply_ambient(x, slice_of[lo:hi], p_ambient, ambient_frac, rng_drop)
        X[lo:hi] = x
        del mu, pi, keep, x

    if ambient_local_frac > 0:
        # ADVERSARIAL ARM ONLY. Convolving counts with a neighbour kernel injects
        # neighbours' RNA into the cell's own row and manufactures niche signal
        # from nothing. Barred from every null and from the headline table.
        warnings.warn(
            "ambient_local_frac > 0 injects neighbours' counts into a cell's own row; "
            "this run is adversarial-only and is not headline-eligible.",
            stacklevel=2,
        )
        X = _apply_local_bleed(X, slice_meta, n_keep, xi_v, ambient_local_frac)

    # ---- assemble -------------------------------------------------------
    obs = _build_obs(slice_meta, n_keep, y_i, s_i, L_i, dose, v_i, u_les, ell,
                     R, C, K_type, seed)
    var = pd.DataFrame(
        {"gt_program": _program_labels(blocks, G), "theta": theta_g},
        index=[f"gene{g:05d}" for g in range(G)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X
    if store_clean:
        adata.layers["counts_clean"] = counts_clean

    adata.obsm["spatial"] = np.concatenate([sm["xy"] for sm in slice_meta]).astype(np.float32)
    adata.obsm["gt_intrinsic"] = np.hstack([onehot_y, u_i]).astype(np.float32)
    adata.obsm["gt_niche"] = v_i.astype(np.float32)
    adata.obsm["gt_salient"] = t_i.astype(np.float32)
    adata.obsm["gt_decoy"] = c_i.astype(np.float32)
    lig_load = np.zeros((n_total, d_v), dtype=np.float32)
    nz = s_i > 0
    lig_load[nz, s_i[nz] - 1] = L_i[nz]
    adata.obsm["gt_ligand_load"] = lig_load
    if eta_store is not None:
        adata.obsm["gt_eta"] = eta_store
    if mu_store is not None:
        adata.obsm["gt_mu"] = mu_store
    if store_gen_kernel:
        adata.obsm["gt_gen_nbr_idx"] = np.concatenate(gen_idx).astype(np.int32)
        adata.obsm["gt_gen_nbr_w"] = np.concatenate(gen_w).astype(np.float32)

    build_knn_index(
        adata, k=k_fit, remove_long_links=remove_long_links,
        distance_percentile=distance_percentile,
    )

    realised_zero = float((X == 0).mean())
    adata.uns["sim_params"] = {
        **params_requested,
        "code_version": CODE_VERSION,
        "resolved": dict(
            R=R, C=C, n_genes=G, n_keep=n_keep, n_slices=n_slices, n_total=n_total,
            xi_v=xi_v, xi_c=xi_c, xi_les=xi_les, lam_lig=lam_lig, xi_u=float(xi_u),
            zi_intercept=float(a0), realised_zero_frac=realised_zero,
            # The bisection targets the zero fraction of the ZERO-INFLATION step.
            # Ambient contamination runs afterwards and refills a small number of
            # zeros, so the FINAL fraction sits slightly below zi_target; both are
            # recorded rather than one being quietly reported as the other.
            realised_zero_frac_pre_ambient=float(n_zero_post_zi / (n_total * G)),
            zi_target=float(zi_target),
            realised_sender_frac=float((s_i > 0).mean()),
            realised_sender_frac_control=float((s_i[~is_disease] > 0).mean()),
            realised_sender_frac_disease=float((s_i[is_disease] > 0).mean()),
            n_isolated_cells=int(np.concatenate(isolated_all).sum()),
            mask_density=neighbor_mask_density(adata),
            frac_cross_slice_neighbors=frac_cross_slice_neighbors(adata),
            median_libsize=float(np.median(np.asarray(X.sum(1)))),
            gen_time_s=_perf() - t_start,
        ),
        "streams": ["intrinsic", "sender", "salient", "decoy", "library",
                    "counts", "atac", "dropout", "loadings", "geometry"],
    }
    adata.uns["null_arm"] = null_arm
    adata.uns["headline_eligible"] = bool(headline_eligible and ambient_local_frac <= 0)
    adata.uns["generating_kernel"] = dict(
        form="w_ij = exp(-d_ij^2 / (2 xi_v^2)) for 0 < d_ij <= 3*xi_v, w_ii = 0, rows sum to 1",
        xi_v=xi_v, cutoff=3.0 * xi_v, reception=reception, kappa_R=kappa_R,
        fitting_graph="kNN", k_fit=k_fit,
    )
    adata.uns["gt_loadings"] = dict(
        A_int=A_int, A_lig=A_lig, B=B, B_alias=B_alias, M=M_alias, C=Cc, E=E,
        A_les=A_les, D=D, beta_0=beta_0, theta=theta_g, mu_type=mu_type,
        b_dir=b_dir, delta_slice=delta_slice,
    )
    adata.uns["gt_loading_scales"] = scales
    adata.uns["gt_gene_blocks"] = {k: v.astype(np.int32) for k, v in blocks.items()}
    if audit:
        adata.uns["gt_independence_audit"] = audit_independence(
            adata, max_cells=audit_max_cells, seed=seed
        )

    if verbose:
        res = adata.uns["sim_params"]["resolved"]
        print(f"[haruka_sim] zero frac {res['realised_zero_frac']:.3f} "
              f"(target {zi_target}), median lib {res['median_libsize']:.0f}, "
              f"mask density {res['mask_density']:.3f}, "
              f"cross-slice nbrs {res['frac_cross_slice_neighbors']:.3g}, "
              f"{res['gen_time_s']:.1f}s")

    if not with_atac:
        return adata

    atac = _simulate_atac(
        adata, rng=np.random.default_rng(ss_atac), n_peaks=n_peaks, d_m=d_m,
        reassign_frac=atac_reassign_frac, atac_mean=atac_mean,
        zi_target=atac_zi_target, y_i=y_i, u_i=u_i, v_i=v_i, t_i=t_i, c_i=c_i,
        u_les=u_les, s_i=s_i, L_i=L_i, is_disease=is_disease,
        lam_int=lam_int, lam_lig=lam_lig, lam_sal=lam_sal, lam_les=lam_les,
        lam_c=lam_c, resp_gain=resp_gain, prog_sd=prog_sd, K_type=K_type,
    )
    return adata, atac


#: The merged spec names the entry point ``simulate_niche_confound``; the task
#: brief names it ``simulate_spatial_contrast``. They are the same function.
simulate_niche_confound = simulate_spatial_contrast


# ---------------------------------------------------------------------------
# generator internals
# ---------------------------------------------------------------------------


def _perf():
    import time

    return time.perf_counter()


def _gene_blocks(G, rng):
    """Disjoint, contiguous-after-shuffle index blocks, one per programme."""
    names = list(GENE_BLOCK_FRACTIONS)
    sizes = [int(round(GENE_BLOCK_FRACTIONS[k] * G)) for k in names]
    sizes[-1] = G - sum(sizes[:-1])
    if min(sizes) < 1:
        raise ValueError(f"n_genes={G} is too small for {len(names)} programmes")
    order = rng.permutation(G)
    out, a = {}, 0
    for k, s in zip(names, sizes):
        out[k] = np.sort(order[a : a + s])
        a += s
    return out


def _program_labels(blocks, G):
    lab = np.empty(G, dtype=object)
    for k, idx in blocks.items():
        lab[idx] = k
    return pd.Categorical(lab, categories=list(GENE_BLOCK_FRACTIONS))


def _diffuse_on_lattice(field_on_cells, sm, R, C, xi):
    """Self-excluded Gaussian diffusion of a per-cell field, via lattice rasterisation.

    Used for the lesion chemokine only. Its range ``xi_les = 4*xi_v`` puts ~1800
    sites inside the cutoff, so a KD-tree pair list would be ~80M pairs at the
    92k-cell arm. Rasterising onto the R x C lattice and filtering is O(RC log)
    and exact for lattice geometry (approximate under jitter, which is acceptable
    for a secondary truth that is confounded with slice by construction and is
    scored separately, never pooled into any disentanglement matrix).
    """
    num_img = np.zeros((R, C), dtype=np.float64)
    den_img = np.zeros((R, C), dtype=np.float64)
    num_img.ravel()[sm["keep"]] = field_on_cells
    den_img.ravel()[sm["keep"]] = 1.0
    num = gaussian_filter(num_img, sigma=xi, mode="constant")
    den = gaussian_filter(den_img, sigma=xi, mode="constant")
    delta = np.zeros((R, C))
    delta[R // 2, C // 2] = 1.0
    k0 = float(gaussian_filter(delta, sigma=xi, mode="constant")[R // 2, C // 2])
    num = num.ravel()[sm["keep"]] - k0 * field_on_cells
    den = den.ravel()[sm["keep"]] - k0
    return num / np.maximum(den, 1e-9)


def _solve_zi_intercept(hist, centers, n_zero_nb, n_entries, zi_target):
    """Bisect ``a0`` so the REALISED marginal zero fraction equals ``zi_target``.

    ``f(a0) = [n_zero_nb + sum_b hist[b] * sigmoid(a0 - center_b)] / n_entries`` is
    monotone increasing in ``a0``. The histogram of ``log1p(mu)`` over NB-nonzero
    entries makes each evaluation O(bins) instead of O(cells x genes), so 60
    bisection steps cost nothing even at the 148M-entry arm.
    """
    base = n_zero_nb / n_entries
    if base >= zi_target:
        warnings.warn(
            f"NB draw alone already yields {base:.3f} zeros >= zi_target={zi_target}; "
            "zero-inflation disabled for this configuration.",
            stacklevel=2,
        )
        return -40.0

    def f(a0):
        return (n_zero_nb + float(np.dot(hist, _sigmoid(a0 - centers)))) / n_entries

    lo, hi = -40.0, 40.0
    if f(hi) < zi_target:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < zi_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _apply_ambient(x, slice_ids, p_ambient, frac, rng):
    """Swap ``frac`` of each cell's counts for draws from the slice's ambient profile.

    The profile is MODEL-IMPLIED (``softmax(beta_0 + delta_slice)``), not the
    empirical slice mean. The empirical mean is a function of every cell in the
    slice, which would make ``x_i`` depend on other cells' realised data at order
    1/n and cost the theorem its exactness. GLOBAL, never local.
    """
    keep = rng.binomial(x, 1.0 - frac).astype(np.int32)
    n_removed = x.sum(axis=1) - keep.sum(axis=1)
    for si in np.unique(slice_ids):
        m = slice_ids == si
        if not m.any():
            continue
        keep[m] += rng.multinomial(n_removed[m], p_ambient[si]).astype(np.int32)
    return keep


def _apply_local_bleed(X, slice_meta, n_keep, xi_v, frac):
    """Adversarial spot bleed: move ``frac`` of counts into the neighbour kernel."""
    out = X.astype(np.float64)
    for si, sm in enumerate(slice_meta):
        lo, hi = si * n_keep, (si + 1) * n_keep
        W, _ = _niche_kernel(sm["xy"], xi_v)
        out[lo:hi] = (1.0 - frac) * out[lo:hi] + frac * (W @ out[lo:hi])
    return np.rint(out).astype(np.int32)


def _build_obs(slice_meta, n_keep, y_i, s_i, L_i, dose, v_i, u_les, ell, R, C, K_type, seed):
    from sklearn.cluster import KMeans

    n_total = len(y_i)
    slice_id = np.concatenate([np.repeat(sm["name"], n_keep) for sm in slice_meta])
    cond = np.concatenate([np.repeat(sm["cond"], n_keep) for sm in slice_meta])
    rr = np.concatenate([sm["rr"] for sm in slice_meta]).astype(np.int32)
    cc = np.concatenate([sm["cc"] for sm in slice_meta]).astype(np.int32)

    nn = np.linalg.norm(v_i, axis=1)
    nbin = (nn > np.median(nn)).astype(int)
    km = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(v_i)
    lesion = (dose > 0.5).astype(int)

    obs = pd.DataFrame(
        {
            "slice_id": pd.Categorical(slice_id),
            "condition": pd.Categorical(cond, categories=["control", "disease"]),
            "array_row": rr,
            "array_col": cc,
            "cell_type": pd.Categorical(y_i.astype(str), categories=[str(i) for i in range(K_type)]),
            "is_sender": s_i > 0,
            "sender_class": s_i.astype(np.int32),
            "sender_load": L_i.astype(np.float32),
            "gt_lesion": lesion.astype(bool),
            "gt_lesion_dose": dose.astype(np.float32),
            "gt_niche_norm": nn.astype(np.float32),
            "gt_niche_bin": nbin.astype(np.int32),
            "gt_niche_label": pd.Categorical(km.labels_.astype(str)),
            "gt_niche_lesion": u_les.astype(np.float32),
            "gt_libsize": ell.astype(np.float32),
            "gt_border_margin": (
                (rr >= 2) & (rr < R - 2) & (cc >= 2) & (cc < C - 2)
            ),
            "true_domain": pd.Categorical(
                np.char.add(np.char.add(y_i.astype(str), "_"),
                            np.char.add(nbin.astype(str), np.char.add("_", lesion.astype(str))))
            ),
        },
        index=[f"{slice_id[i]}_{i:07d}" for i in range(n_total)],
    )
    obs["cv_block"] = -1  # written by haruka_metrics.assign_cv_blocks
    return obs


def _simulate_atac(
    adata, rng, n_peaks, d_m, reassign_frac, atac_mean, zi_target,
    y_i, u_i, v_i, t_i, c_i, u_les, s_i, L_i, is_disease,
    lam_int, lam_lig, lam_sal, lam_les, lam_c, resp_gain, prog_sd, K_type,
):
    """Paired ATAC. A PLUMBING CHECK for HarukaMultiomicsSpatial, not a biological claim.

    Peak blocks mirror the gene blocks with 20% of peaks reassigned at random, so
    the modalities are correlated but not a relabelling of each other, plus a
    modality-specific factor ``m_i`` with its own peak block so the multiomic arm
    is not trivially redundant.
    """
    n = adata.n_obs
    names = list(GENE_BLOCK_FRACTIONS) + ["modality"]
    fr = np.array([GENE_BLOCK_FRACTIONS[k] for k in names[:-1]] + [0.10])
    fr = fr / fr.sum()
    sizes = np.maximum(1, np.round(fr * n_peaks).astype(int))
    sizes[-1] = n_peaks - int(sizes[:-1].sum())
    if sizes[-1] < 1:
        raise ValueError(f"n_peaks={n_peaks} is too small for {len(names)} programmes")
    # blocks mirror the gene blocks, then 20% of peaks are reassigned at random
    assign = np.repeat(np.arange(len(names)), sizes)[rng.permutation(n_peaks)]
    swap = rng.random(n_peaks) < reassign_frac
    assign[swap] = rng.integers(0, len(names), size=int(swap.sum()))
    blocks = {k: np.flatnonzero(assign == i) for i, k in enumerate(names)}

    m_i = rng.standard_normal((n, d_m))            # modality-specific factor
    eta = np.tile(rng.standard_normal(n_peaks), (n, 1))
    onehot = np.zeros((n, K_type))
    onehot[np.arange(n), y_i] = 1.0

    def _add(key, F, gain=1.0, rows=None):
        """Add ``gain * F @ L.T`` on block ``key``, programme-scaled."""
        b = blocks.get(key)
        if b is None or b.size == 0 or gain == 0.0:
            return
        L = rng.standard_normal((b.size, F.shape[1]))
        L, _ = _scale_programme(L, F @ L.T, prog_sd)
        if rows is None:
            eta[:, b] += gain * (F @ L.T)
        else:
            eta[np.ix_(rows, b)] += gain * (F[rows] @ L.T)

    _add("intrinsic", onehot, gain=lam_int)
    _add("intrinsic", u_i)
    _add("alias", u_i)
    if lam_lig != 0.0 and blocks["ligand"].size:
        b = blocks["ligand"]
        A = np.vstack([np.zeros((1, b.size)), rng.standard_normal((v_i.shape[1], b.size))])
        A, _ = _scale_programme(A, A[s_i] * L_i[:, None], prog_sd)
        eta[:, b] += lam_lig * A[s_i] * L_i[:, None]
    _add("response", v_i, gain=resp_gain)
    if is_disease.any():
        _add("salient", t_i, gain=lam_sal, rows=is_disease)
        _add("lesion", u_les[:, None], gain=lam_les, rows=is_disease)
    _add("decoy", c_i, gain=lam_c)
    _add("modality", m_i)

    rate = (atac_mean * n_peaks) * _softmax_rows(eta)
    Xc = rng.poisson(rate).astype(np.int32)
    lm = np.log1p(rate)
    pos = Xc > 0
    edges = np.linspace(0.0, _ZI_MAX, _ZI_BINS + 1)
    hist = np.histogram(lm[pos], bins=edges)[0].astype(np.float64)
    a0 = _solve_zi_intercept(hist, 0.5 * (edges[1:] + edges[:-1]),
                             int(Xc.size - pos.sum()), Xc.size, zi_target)
    # independent per-modality dropout
    X = (Xc * (rng.random(Xc.shape) >= _sigmoid(a0 - lm))).astype(np.int32)

    prog = np.empty(n_peaks, dtype=object)
    for k in names:
        prog[blocks[k]] = k
    at = ad.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=pd.DataFrame(
            {"gt_program": pd.Categorical(prog, categories=names)},
            index=[f"peak{p:05d}" for p in range(n_peaks)],
        ),
    )
    at.layers["counts"] = X
    for k in ("spatial", "gt_intrinsic", "gt_niche", "gt_salient", "gt_decoy",
              "gt_ligand_load", "gt_gen_nbr_idx", "gt_gen_nbr_w",
              NBR_IDX_KEY, NBR_MASK_KEY):
        if k in adata.obsm:
            at.obsm[k] = adata.obsm[k].copy()
    at.obsm["gt_modality"] = m_i.astype(np.float32)
    at.uns["sim_params"] = adata.uns["sim_params"]
    at.uns["null_arm"] = adata.uns["null_arm"]
    at.uns["atac_zero_frac"] = float((X == 0).mean())
    return at


# ---------------------------------------------------------------------------
# null-arm helpers (fit-time / score-time)
# ---------------------------------------------------------------------------


def permute_graph_within_slice(adata, seed=0, idx_key=NBR_IDX_KEY, library_key="slice_id"):
    """N3: shuffle the neighbour index WITHIN each slice, at fit time.

    Gives the empirical noise floor of the whole pipeline: the graph still has the
    right degree, mask density and slice confinement, and carries no geometry.
    """
    rng = np.random.default_rng(seed)
    idx = np.asarray(adata.obsm[idx_key]).copy()
    lab = np.asarray(adata.obs[library_key]).astype(str)
    for u in pd.unique(lab):
        g = np.flatnonzero(lab == u)
        idx[g] = g[rng.integers(0, g.size, size=(g.size, idx.shape[1]))]
    adata.obsm[idx_key] = idx
    adata.uns["null_arm"] = "permuted_graph"
    return idx


def shuffle_truth(adata, key="gt_niche", seed=0):
    """N4: cell-wise permutation of the truth, at score time. A harness self-test.

    Destroys the target's autocorrelation, so its floor is artificially LOW. Use
    :func:`resample_truth` for the honest null on an autocorrelated target.
    """
    rng = np.random.default_rng(seed)
    return np.asarray(adata.obsm[key])[rng.permutation(adata.n_obs)]


def resample_truth(adata, seed=0):
    """N5: a FRESH independent draw of the same niche process on the same geometry.

    Preserves the target's spatial autocorrelation, which N4 destroys, and is
    therefore the correct null for an autocorrelated target.
    """
    p = adata.uns["sim_params"]
    res = p["resolved"]
    rng = np.random.default_rng(seed + 987_654)
    xi_v = float(res["xi_v"])
    d_v = int(p.get("d_v", 3))
    p_send = float(p.get("p_send", 0.12))
    lab = np.asarray(adata.obs["slice_id"]).astype(str)
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    out = np.zeros((adata.n_obs, d_v))
    for u in pd.unique(lab):
        g = np.flatnonzero(lab == u)
        occ = rng.random(g.size) < p_send
        cls = rng.integers(1, d_v + 1, size=g.size)
        L = np.where(occ, rng.gamma(4.0, 0.25, size=g.size), 0.0)
        mark = np.zeros((g.size, d_v))
        nz = occ
        mark[nz, cls[nz] - 1] = L[nz]
        W, _ = _niche_kernel(xy[g], xi_v)
        out[g] = W @ mark
    sd = out.std(0)
    sd[sd < 1e-12] = 1.0
    return ((out - out.mean(0)) / sd).astype(np.float32)


# ---------------------------------------------------------------------------
# independence audit
# ---------------------------------------------------------------------------


def audit_independence(adata, max_cells=5000, seed=0, n_perm_null=30, hsic_cells=800):
    """Realised dCor and HSIC p-values for all 6 pairs of (int, niche, salient, decoy).

    Reported against an AUTOCORRELATION-PRESERVING tile-permutation null rather
    than a fixed absolute threshold. With ``xi_u=8`` on a 128-lattice there are
    only a few hundred effectively independent patches; dCor is non-negative and
    upward-biased at that effective n, so a fixed cut such as "dCor < 0.05" is
    either unattainable (infinite redraw loop) or vacuous.

    Pairs involving the salient factor are computed on the DISEASE slices only --
    the salient truth is identically zero in control, so pooling would measure the
    zero block rather than any dependence.

    **Read this audit at the ``main`` preset, not at ``small``.** The statistic is
    limited by the number of independent spatial patches, not by the cell count.
    At R=48 with ``xi_u=4`` there are only ~36 patches per slice, and the audit is
    then underpowered in a specific, measurable way: for the same seed and the same
    independent-by-construction pair, dCor reads 0.158 with a +0.02 excess at
    ``small`` but 0.085 with a NEGATIVE excess at ``main`` (R=128, ~256 patches),
    and the HSIC p-value moves from the permutation floor to 0.38-0.80. The
    ``small`` reading is dCor's upward bias at low effective n, not a dependence.
    Sweeping the null's tile size over {4, 8, 16, 24, 32} does not change this,
    which is what identifies effective n rather than the null's construction as
    the cause.
    """
    # Imported here rather than at module scope: haruka_metrics must never depend
    # on haruka_sim (it has to work on any latent), so the one-way lazy import is
    # what keeps the two files acyclic while sharing a single implementation.
    from haruka_metrics import distance_correlation_null, hsic_pvalue, tile_ids

    rng = np.random.default_rng(seed + 4242)
    keys = dict(TRUTH_KEYS)
    disease = np.asarray(adata.obs["condition"]).astype(str) == "disease"
    tiles_all = tile_ids(adata)

    def _sub(mask, k):
        """A RANDOM subsample of ``k`` eligible cells.

        Deliberately not the head of a sorted index array: rows are laid out slice
        by slice, so taking the first k would hand back one contiguous slice and
        make an ostensibly global audit a single-slice, spatially-clustered one --
        which also starves the tile permutation of tiles to permute.
        """
        g = np.flatnonzero(mask)
        if g.size > k:
            g = np.sort(rng.choice(g, k, replace=False))
        return g

    out = {"pairs": {}, "n_rejected_draws": 0,
           "disclosure": ("No rejection sampling is applied by default. Redrawing until "
                          "an independence threshold is met biases the ensemble toward "
                          "atypically decoupled draws; the realised statistics are "
                          "reported against a tile-permutation null instead."),
           "null": f"tile permutation, {n_perm_null} draws, 95th percentile"}
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mask = disease if ("F_sal" in (a, b)) else np.ones(adata.n_obs, bool)
            A_all = np.asarray(adata.obsm[keys[a]], np.float64)
            B_all = np.asarray(adata.obsm[keys[b]], np.float64)
            gs = _sub(mask, max_cells)
            if A_all[gs].std() < 1e-12 or B_all[gs].std() < 1e-12:
                out["pairs"][f"{a}|{b}"] = dict(dcor=0.0, dcor_null_p95=0.0,
                                                dcor_excess=0.0, hsic_p=1.0,
                                                n=int(gs.size))
                continue
            res = distance_correlation_null(A_all[gs], B_all[gs], tiles_all[gs],
                                            n_perm=n_perm_null,
                                            seed=int(rng.integers(0, 2**31 - 1)))
            gh = _sub(mask, hsic_cells)
            # tile null, not a cell null: every factor here is spatially
            # autocorrelated, and a cell-wise permutation reports p ~ 0.02 on
            # factors that are independent by construction.
            res["hsic_p"] = float(hsic_pvalue(
                A_all[gh], B_all[gh], seed=int(rng.integers(0, 2**31 - 1)),
                max_cells=hsic_cells, tiles=tiles_all[gh], n_perm=100,
            )["p"])
            out["pairs"][f"{a}|{b}"] = res
    return out


# ---------------------------------------------------------------------------
# THE SELF-CHECK -- validates the SIMULATOR, not the model
# ---------------------------------------------------------------------------


def _probe_r2_ridge(X, y, folds, tiles, alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)):
    """Blocked-CV ridge R^2, pooled over folds, centred by the TRAIN mean.

    ``alpha`` is chosen by an inner cross-validation that is ITSELF SPATIALLY
    BLOCKED (grouped on tiles). This is not a refinement -- it is load-bearing.
    With a random inner split, a cell's own neighbours land in the inner-training
    set; the neighbour-mean features are smooth, so they interpolate, a tiny alpha
    wins, and the model then extrapolates catastrophically into the
    spatially-held-out OUTER fold. Measured on the null arm, that mis-selection
    drove the neighbour probe to R^2 = -0.77 where the honest answer is ~0.
    Grouping the inner split on tiles makes alpha selection face the same
    extrapolation the outer score measures.

    A generous alpha grid matters for the same reason the probe is given PCA
    features and a proper penalty: the own-row probe must be handicapped by
    nothing, so that its zero is a property of the CONSTRUCTION and not of a
    crippled regressor.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    tiles = np.asarray(tiles)
    num = den = 0.0
    for tr, te in folds:
        if tr.size < 20 or te.size < 5:
            continue
        mx, sx = X[tr].mean(0), X[tr].std(0)
        sx[sx < 1e-12] = 1.0
        Xtr, Xte = (X[tr] - mx) / sx, (X[te] - mx) / sx
        ytr, my = y[tr], y[tr].mean(0)
        g = tiles[tr]
        n_grp = np.unique(g).size
        best, best_sc = alphas[-1], -np.inf
        if n_grp >= 3:
            inner = GroupKFold(n_splits=min(3, n_grp))
            for a in alphas:
                sc = 0.0
                for i_tr, i_va in inner.split(Xtr, ytr, groups=g):
                    m = Ridge(alpha=a, fit_intercept=True).fit(Xtr[i_tr], ytr[i_tr])
                    sc -= float(((ytr[i_va] - m.predict(Xtr[i_va])) ** 2).sum())
                if sc > best_sc:
                    best_sc, best = sc, a
        pred = Ridge(alpha=best, fit_intercept=True).fit(Xtr, ytr).predict(Xte)
        num += float(((y[te] - pred) ** 2).sum())
        den += float(((y[te] - my) ** 2).sum())
    return float(1.0 - num / den) if den > 0 else float("nan")


def self_check(
    seed=0,
    preset="small",
    n_pcs=50,
    k_probe=None,
    verbose=True,
    strict=True,
    own_tol=0.05,
    nbr_min=0.15,
    null_tol=0.05,
    **sim_kwargs,
):
    """Prove the niche factor is NOT recoverable from a cell's own expression.

    This is the load-bearing validation of the simulator itself. If it fails, the
    whole benchmark is worthless and the simulator -- not the model -- is what
    needs fixing.

    Fits spatially blocked cross-validated ridge probes predicting the true niche
    ``v`` from

      * ``own``        -- the cell's OWN log1p expression (PCA-reduced),
      * ``nbr_kernel`` -- the TRUE generative kernel mean of neighbours' expression
                          (baseline arm (e), the ceiling: if this is low the ARM IS
                          DEAD and must be discarded, not read as a model failure),
      * ``nbr_knn``    -- the deliberately misspecified k-NN mean (baseline arm (c)),

    and asserts ``own ~ 0`` while the neighbour probes are clearly positive; then
    repeats under the null (``niche_strength=0``) where ALL of them must collapse.

    Spatially blocked CV with a buffer is mandatory here, not a refinement. With
    random folds a cell's own neighbours land in the training set, and since the
    niche IS a function of neighbours the probe near-interpolates -- the own-row
    probe would then score above its theoretical zero purely via position.

    The decoy control is also reported: the anti-target IS recoverable from the
    cell's own row (it is cell-autonomous), which confirms the own-row probe has
    real power and that its zero on the niche is a property of the construction
    rather than a dead probe.

    ``k_probe`` overrides the fitting graph's ``k_fit`` for the kNN arm; leave it
    ``None`` to check the simulator at the same k the benchmark will use.
    """
    from haruka_metrics import blocked_cv_folds, tile_ids

    rep: dict[str, Any] = {"seed": seed, "preset": preset, "failures": []}
    if k_probe is not None:
        sim_kwargs.setdefault("k_fit", int(k_probe))

    for tag, ns in (("signal", sim_kwargs.pop("niche_strength", 1.0)), ("null", 0.0)):
        adata = simulate_spatial_contrast(
            preset=preset, seed=seed, niche_strength=ns, audit=False, **sim_kwargs
        )
        xi_v = float(adata.uns["sim_params"]["resolved"]["xi_v"])
        R = int(adata.uns["sim_params"]["resolved"]["R"])
        folds = blocked_cv_folds(adata, xi=xi_v, n_folds=5, n_tiles=5, R=R)
        tiles = tile_ids(adata)

        Xl = np.log1p(np.asarray(adata.layers["counts"], np.float32))
        own = _pca(Xl, n_pcs, seed)
        knn = _pca(neighbor_mean(Xl, adata.obsm[NBR_IDX_KEY], adata.obsm[NBR_MASK_KEY]),
                   n_pcs, seed)
        ker = _pca(gen_kernel_mean(adata, Xl), n_pcs, seed)

        v = np.asarray(adata.obsm["gt_niche"], np.float64)
        dec = np.asarray(adata.obsm["gt_decoy"], np.float64)
        r = {
            "own": _probe_r2_ridge(own, v, folds, tiles),
            "nbr_kernel": _probe_r2_ridge(ker, v, folds, tiles),
            "nbr_knn": _probe_r2_ridge(knn, v, folds, tiles),
            "own_decoy_control": _probe_r2_ridge(own, dec, folds, tiles),
            "xi_v": xi_v,
            "n_cells": int(adata.n_obs),
            "zero_frac": float(adata.uns["sim_params"]["resolved"]["realised_zero_frac"]),
            "sender_frac": float(adata.uns["sim_params"]["resolved"]["realised_sender_frac"]),
            "frac_cross_slice_neighbors": float(
                adata.uns["sim_params"]["resolved"]["frac_cross_slice_neighbors"]
            ),
        }
        # sender / receiver diagnostic: a gap means something leaked
        snd = np.asarray(adata.obs["is_sender"])
        for nm, m in (("sender", snd), ("receiver", ~snd)):
            keep = np.flatnonzero(m)
            sub = [(t[np.isin(t, keep)], e[np.isin(e, keep)]) for t, e in folds]
            r[f"own_{nm}"] = _probe_r2_ridge(own, v, sub, tiles)
        rep[tag] = r

        if verbose:
            print(f"\n--- self_check [{tag}]  niche_strength={ns} "
                  f"(n={r['n_cells']}, xi_v={xi_v:.2f}, zeros={r['zero_frac']:.2f}) ---")
            for k in ("own", "own_sender", "own_receiver", "nbr_kernel", "nbr_knn",
                      "own_decoy_control"):
                print(f"    R2 {k:<20s} {r[k]:+.4f}")

    s, z = rep["signal"], rep["null"]
    checks = [
        ("own-row niche R2 ~ 0 (theorem: Bayes R2 = 0)", s["own"] <= own_tol),
        ("true-kernel neighbour niche R2 clearly positive", s["nbr_kernel"] >= nbr_min),
        ("kNN neighbour niche R2 positive", s["nbr_knn"] > 0.02),
        ("neighbour >> own", s["nbr_kernel"] - max(s["own"], 0.0) >= nbr_min),
        ("own-row probe has real power (decoy recoverable)", s["own_decoy_control"] > 0.10),
        ("NULL: own-row niche R2 ~ 0", z["own"] <= null_tol),
        ("NULL: kernel-neighbour niche R2 ~ 0", z["nbr_kernel"] <= null_tol),
        ("NULL: kNN-neighbour niche R2 ~ 0", z["nbr_knn"] <= null_tol),
        ("sender/receiver own-row gap small", abs(s["own_sender"] - s["own_receiver"]) <= 0.10),
        ("no cross-slice neighbours (T2)", s["frac_cross_slice_neighbors"] == 0.0),
    ]
    rep["checks"] = {k: bool(v) for k, v in checks}
    rep["failures"] = [k for k, v in checks if not v]
    rep["passed"] = len(rep["failures"]) == 0
    if verbose:
        print("\n--- verdict ---")
        for k, v in checks:
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    if strict and not rep["passed"]:
        raise AssertionError("simulator self-check FAILED: " + "; ".join(rep["failures"]))
    return rep


def _pca(X, n_pcs, seed):
    from sklearn.decomposition import PCA

    X = np.asarray(X, np.float32)
    k = int(min(n_pcs, min(X.shape) - 1))
    if k < 1:
        return X
    return PCA(n_components=k, random_state=seed).fit_transform(X)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="small", choices=sorted(PRESETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true", help="run the simulator self-check")
    a = p.parse_args()
    if a.check:
        self_check(seed=a.seed, preset=a.preset, verbose=True)
    else:
        adata = simulate_spatial_contrast(preset=a.preset, seed=a.seed, verbose=True)
        print(adata)
