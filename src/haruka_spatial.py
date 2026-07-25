"""Spatially aware Haruka: NICHE-AWARE background and salient latents.

Haruka contrasts CONDITIONS: ``background`` is variation shared by control and
disease, ``salient`` is variation only the disease arm has. This module keeps
that decomposition exactly as it is -- two latents, same widths, same zeroing of
``s`` on control cells, same decoder -- and makes BOTH of them aware of where the
cell sits in the tissue.

The niche is modelled as a CELL SET: a cell's k spatial neighbors, optionally
including the cell itself, pooled permutation-invariantly into a fixed-width
context vector. Two such blocks are learned, one per latent, and each encoder is
handed its own::

    z = z_encoder(cat([x_own, ctx_bg]))    s = s_encoder(cat([x_own, ctx_sal]))
    latent = cat([z, s])

So the niche enters as ENCODER INPUT, never as a latent of its own. Two blocks
rather than one because the two latents want different things from the niche:
the background latent wants the structure shared across conditions (tissue
architecture), the salient latent wants the structure that separates the lesion
microenvironment.

The set-attention block is a native, heavily shrunk reimplementation of the axial
"tabular" transformer of the Arc stack model -- gene-module tokens, attention
across the members of the set, pre-norm residuals, token-wise MLP, no positional
encoding on the set axis. Its pretrained weights are neither used nor usable
(human vocabulary, mouse data) and stack is not a dependency. Stack's per-cell NB
reconstruction needs no porting either: Haruka already uses the same
``scvi.distributions.NegativeBinomial`` through ``DecoderSCVI``.

A SECOND, older design also lives here and stays reachable for comparison:
``n_spatial_latent > 0`` adds a SIMVI-style (Dong et al., Nat Commun 16:2990,
2025) third latent that splits intrinsic from interaction-induced state, with an
independence penalty, its own KL and a denoising phase. SIMVI's mechanism is
reimplemented natively inside the scvi-tools fork rather than imported: SIMVI
pins ``scvi-tools<=0.16.2`` and this fork is 1.2.2, so installing it would break
the fork. See README_SPATIAL_SIMVI.md.

Nothing here changes ``Haruka`` or ``HarukaMultiomics``: the spatial path lives
entirely in subclasses and is opt-in.
"""

import inspect
import os
import sys

import numpy as np
import anndata as ad
import scipy.sparse
import squidpy as sq
import scenvi
import scvi
import cellcharter as cc
from pytorch_lightning import seed_everything

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Haruka import Haruka
from HarukaMultiomics import ensure_spatial
from haruka_cluster import FastCluster

__all__ = [
    "build_neighbor_index",
    "HarukaSpatial",
    "HarukaPeakSpatial",
    "HarukaMultiomicsSpatial",
    "ensure_spatial",
    "NBR_IDX_KEY",
    "NBR_MASK_KEY",
]

NBR_IDX_KEY = "spatial_nbr_idx"
NBR_MASK_KEY = "spatial_nbr_mask"


def build_neighbor_index(
    adata,
    k=10,
    spatial_key="spatial",
    library_key="slice_id",
    remove_long_links=True,
    distance_percentile=99.0,
    idx_key=NBR_IDX_KEY,
    mask_key=NBR_MASK_KEY,
    tie_tol=0.02,
    degenerate_frac=0.6,
    verbose=True,
):
    """Build the within-section kNN index the spatial branch attends over.

    Writes ``(n_obs, k)`` arrays into ``adata.obsm[idx_key]`` and
    ``adata.obsm[mask_key]`` and returns ``(idx, mask)``.

    ``idx`` holds GLOBAL row positions into ``adata``. That is deliberate and it
    is what makes the design work at all: the contrastive dataloader yields
    spatially non-contiguous cells, so a cell's neighbors are essentially never
    in its own minibatch and have to be gathered from a resident bank instead.
    Build the graph on the same full object you pass to ``setup_anndata``.

    A cell is never its own neighbor, so the pooled representation is
    neighbor-induced rather than a copy of the cell. Padded slots (sections with
    fewer than k+1 cells, or edges dropped as implausibly long) point at the cell
    itself so the gather stays in range, and are marked invalid in ``mask`` so
    they contribute nothing to the attention.

    Long-link removal follows cellcharter's intent but thresholds per section
    rather than pooling every section's distances into one global percentile, so
    a sparse section is not trimmed against a dense one's spacing. Note it is
    close to vacuous on array data: on the real arcseq lattice it drops 0.1% of
    edges at k=10, so it provides none of the safeguard its name suggests.

    On a regular array the k-nearest query can cut a degenerate distance shell in
    half and pick an arbitrary subset of the tied sites, which makes the stencil
    directionally biased. ``tie_tol`` / ``degenerate_frac`` control the correction
    for that; see :meth:`scvi.external.ContrastiveVI.build_neighbor_index` for the
    measurements. The correction is gated so it is a bit-exact no-op on
    continuous coordinates.
    """
    idx, mask = scvi.external.ContrastiveVI.build_neighbor_index(
        adata,
        k=k,
        spatial_key=spatial_key,
        library_key=library_key,
        remove_long_links=remove_long_links,
        distance_percentile=distance_percentile,
        tie_tol=tie_tol,
        degenerate_frac=degenerate_frac,
        verbose=verbose,
    )
    adata.obsm[idx_key] = idx
    # float32 because scvi's ObsmField casts obsm to float32 on load regardless;
    # storing it that way keeps the inspected dtype equal to the loaded one.
    adata.obsm[mask_key] = mask.astype(np.float32)
    return idx, mask


class HarukaSpatial(Haruka):
    """Haruka with NICHE-AWARE background and salient latents.

    Adds one step to the pipeline::

        construct_cov() -> prepare_spatial_graph() -> setup_model()
                        -> train() -> extract_rep() -> cluster()

    ``prepare_spatial_graph`` is optional; ``setup_model`` builds the graph with
    this object's defaults if it has not been built yet.

    Two designs live here, and ``setup_model`` selects between them. BOTH are
    opt-in: with no spatial argument given, this class trains exactly the plain
    contrastive model its parent does.

    NICHE CONTEXT (opt in with ``niche_context=True``). Haruka's two latents are
    kept exactly as they
    are -- background and salient, same widths, same contrastive zeroing of ``s``
    on control cells, same decoder -- and BOTH are made niche-aware by handing
    each encoder its own pooled summary of the cell's spatial neighborhood as
    extra INPUT::

        ctx_bg  = background_block(x_own, x_nbr, mask)
        ctx_sal = salient_block   (x_own, x_nbr, mask)
        z = z_encoder(cat([x_own, ctx_bg]))    s = s_encoder(cat([x_own, ctx_sal]))
        latent = cat([z, s])

    The niche is modelled as a CELL SET and pooled permutation-invariantly, so it
    is read as a set rather than as an ordered list. Two separate attention
    blocks, never one: the background latent wants the niche structure that is
    shared across conditions (tissue architecture) while the salient latent wants
    the structure that separates the lesion microenvironment, and one shared
    block would have to represent a compromise. ``extract_rep`` therefore still
    produces exactly ``salient_rep`` and ``background_rep`` (plus their
    CellCharter-aggregated forms) -- there is no ``spatial_rep`` in this mode,
    because there is no third latent.

    SIMVI-STYLE THIRD LATENT (opt in with ``n_spatial_latent > 0``). Adds a
    separate niche-induced latent alongside the two, with an independence penalty
    between it and the intrinsic latents, its own KL, an annealing schedule and a
    denoising phase. It answers a different question -- separating intrinsic from
    niche-induced state -- and it is kept reachable for comparison runs. Asking
    for both designs explicitly raises rather than silently combining them.

    Leaving both unset -- ``n_spatial_latent=0``, ``niche_context`` unspecified --
    disables all spatial machinery and reproduces the parent model bit-for-bit,
    which is both the default and the clean ablation to run against.

    Parameters
    ----------
    adata
        Object carrying ``obsm['spatial']`` and the section / condition columns.
    cov_gene_num
        Number of HVGs for the COVET niche-covariance block, as in ``Haruka``.
    seed
        Default seed for clustering.
    k_spatial
        Neighbors per cell in the attention graph. SIMVI uses 10 throughout, and
        that transfers here because ``build_neighbor_index`` will not admit a
        partial distance shell: on the regular arrays that DBiT / ARC-seq /
        Visium produce, the shells hold 4, 4 and 4 sites, so ``k=10`` resolves to
        the first two complete shells (8 neighbors) rather than 8 plus two
        arbitrary of the four tied third-shell sites. Without that correction
        ``k=10`` gives 71% of interior spots on the real arcseq lattice a stencil
        with a 2.1-pitch net displacement, decided by coordinate rounding rather
        than by biology. On continuous (imaging) coordinates nothing is trimmed
        and k means exactly k.
        Note the COVET niche is a separate graph fixed at k=8 by scenvi.
    spatial_key
        ``obsm`` key holding the coordinates.
    library_key
        ``obs`` column identifying the section. Neighbors never cross sections.
    """

    def __init__(
        self,
        adata,
        cov_gene_num=32,
        seed=42,
        k_spatial=10,
        spatial_key="spatial",
        library_key="slice_id",
    ):
        super().__init__(adata, cov_gene_num=cov_gene_num, seed=seed)
        self.seed = seed
        self.k_spatial = k_spatial
        self.spatial_key = spatial_key
        self.library_key = library_key
        self.n_spatial_latent = 0
        self.niche_context = False
        self.cluster_models_ = {}
        self._graph_ready = False
        self._mae_epochs_auto = False
        self._mae_fraction = 0.2

    # -- COVET -------------------------------------------------------------

    def construct_cov(self):
        """COVET construction, with the parent's reshape corrected.

        ``Haruka.construct_cov`` reshapes the COVET block using
        ``g = self.cov_gene_num``, but ``highly_variable_genes(n_top_genes=g)``
        can return a different number of genes when dispersions tie. On the
        arcseq RNA it returns 40 for ``g=32``, and the parent then dies with
        "cannot reshape array of size 222400 into shape (1024)".

        Take the width from the returned array instead, and keep
        ``cov_gene_num`` in sync so ``n_me = cov_gene_num**2`` still describes
        the block that was actually appended. Whenever the parent succeeds the
        two agree exactly, so this does not disturb spatial-on/spatial-off
        comparisons against ``Haruka``.
        """
        g = self.cov_gene_num
        adata = self.adata.copy()
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()

        if g <= 0:
            self.input_adata = adata
            self.cov_gene_num = 0
            return

        try:
            _, covet_sqrt, _ = scenvi.utils.compute_covet(adata, g=g, batch_key="slice_id")
        except Exception as err:
            # The parent swallows this silently, which hides the fact that the model
            # then trains with no COVET block at all. Same fallback, but say so.
            print(f"[HarukaSpatial] COVET failed ({type(err).__name__}: {err}); "
                  "continuing without the COVET block")
            self.input_adata = adata
            self.cov_gene_num = 0
            return

        g_actual = int(covet_sqrt.shape[-1])
        if g_actual != g:
            print(
                f"[HarukaSpatial] COVET used {g_actual} HVGs, not the {g} requested "
                "(dispersion ties); continuing with the actual width"
            )
            self.cov_gene_num = g_actual

        new_adata = ad.AnnData(
            X=np.concatenate(
                [adata.X, covet_sqrt.reshape(-1, g_actual * g_actual)], axis=1
            ),
            obs=adata.obs,
        )
        new_adata.obsm = adata.obsm
        self.input_adata = new_adata

    # -- graph -------------------------------------------------------------

    def prepare_spatial_graph(
        self,
        k=None,
        spatial_key=None,
        library_key=None,
        remove_long_links=True,
        distance_percentile=99.0,
    ):
        """Build and register the neighbor index on the model input object.

        Runs ``construct_cov()`` first if it has not run, because the index has
        to live on the same object ``setup_anndata`` registers.
        """
        if self.input_adata is None:
            self.construct_cov()
        k = self.k_spatial if k is None else k
        spatial_key = self.spatial_key if spatial_key is None else spatial_key
        library_key = self.library_key if library_key is None else library_key

        idx, mask = build_neighbor_index(
            self.input_adata,
            k=k,
            spatial_key=spatial_key,
            library_key=library_key,
            remove_long_links=remove_long_links,
            distance_percentile=distance_percentile,
        )
        self.k_spatial = k
        self.spatial_key = spatial_key
        self.library_key = library_key
        self._graph_ready = True
        return idx, mask

    # -- model -------------------------------------------------------------

    def setup_model(
        self,
        latent_dim=256,
        gene_likelihood="poisson",
        me_weight=0.3,
        me_reduction="sum",
        batch_key="slice_id",
        wasserstein_penalty=0.2,
        condition_key="condition",
        condition_label="disease",
        seed=42,
        layer=None,
        niche_context=None,
        n_context=128,
        context_pool="setattn",
        context_shared=False,
        include_self=True,
        mask_rate=0.0,
        n_spatial_latent=0,
        spatial_attach="background",
        spatial_encoder="gat",
        n_attention_heads=1,
        setattn_tokens=16,
        setattn_token_dim=32,
        setattn_layers=2,
        setattn_heads=4,
        setattn_mlp_ratio=4,
        setattn_dropout=0.0,
        setattn_self_query=False,
        kl_spatial_weight=0.01,
        lam_mi=5.0,
        reg_to_use="mi",
        mmd_bandwidth="median",
        mi_debias=True,
        penalize_salient_spatial=False,
        spatial_invariance_penalty=0.5,
        mask_salient_mean_on_background=False,
        me_from_background_only=False,
        sw_n_proj=64,
        permutation_rate=0.5,
        mae_epochs=None,
        anneal_epochs=50,
        var_eps=1e-4,
        mmd_max_samples=512,
    ):
        """Register the data and build the model.

        Parameters beyond the parent's are all spatial controls.

        niche_context
            Make the EXISTING background and salient latents niche-aware, by giving
            each encoder its own pooled summary of the cell's neighborhood as extra
            input. No third latent, no independence penalty, no spatial KL, no
            denoising phase.

            ``None`` (default) resolves to ``False``: this is an opt-in feature and
            leaving it unset reproduces the plain contrastive model exactly. Pass
            ``True`` to select it. Passing ``True`` together with
            ``n_spatial_latent > 0`` raises rather than silently combining two
            designs.
        n_context
            Width of EACH pooled niche vector. Both encoders widen by this much; the
            library encoder still reads the cell's own expression only.
        context_pool
            How the niche set is pooled.

            ``'setattn'`` (default) is the stack-style axial set-attention block:
            the feature axis is linearly compressed into ``setattn_tokens``
            gene-module tokens of width ``setattn_token_dim``, then
            ``setattn_layers`` blocks alternate attention over the tokens within a
            set member and across the members of the set, with a pre-norm residual
            and a token-wise MLP. No positional encoding on the set axis, so it is
            permutation-invariant: the niche is read as a set.

            ``'attention'`` is the cheaper GATv2 pooling over a single linear
            projection -- one weight per member, no cross-member mixing.

            ``'mean'`` is a masked mean of a single linear projection. Run it: it is
            the ablation that says whether the attention is earning its parameters.
            If ``'mean'`` ties ``'setattn'``, it is not.
        context_shared
            ``False`` (default) builds TWO independent pooling blocks, one per
            latent, which is the point of the design. ``True`` feeds both encoders
            one shared vector and exists only as the ablation that measures what
            that compromise costs.
        include_self
            Let the cell be a member of its own niche set. ``True`` by default. The
            self-exclusion the SIMVI branch enforces exists only to keep
            "niche-induced" identifiable against "intrinsic"; this design draws no
            such line.
        mask_rate
            Stack-style rectangular masked-feature regularisation on the niche
            branch: the same random gene columns are zeroed across every member of
            the set, resampled each training step, pooler input only. ``0.0``
            (default) disables it.
        n_spatial_latent
            Width of the SIMVI-style third latent. ``0`` (default) leaves that design
            off. Cannot exceed the width of the latent named by ``spatial_attach``.
            Setting it > 0 selects the SIMVI design and, unless ``niche_context`` was
            passed explicitly, turns the niche-context design off.
        spatial_attach
            Which encoder supplies the attention features, ``'background'`` or
            ``'salient'``. Keep ``'background'``: the Wasserstein penalty drives
            the salient posterior toward zero on control cells, so attaching
            there leaves the attention reading degenerate features on half the
            data. ``'salient'`` is exposed as an ablation.
        spatial_encoder
            Which neighbor encoder computes the niche latent.

            ``'gat'`` (default) is SIMVI's GATv2 over the neighbors' shared-encoder
            posterior means.

            ``'setattn'`` is a small stack-style axial set-attention transformer
            that tokenizes the neighbors' expression itself: the feature axis is
            linearly compressed into ``setattn_tokens`` gene-module tokens of width
            ``setattn_token_dim``, then ``setattn_layers`` blocks alternate
            attention over the tokens within a neighbor and across the k neighbors,
            with a token-wise MLP. It is permutation-invariant over the neighbor set
            by construction, and at the default ``setattn_self_query=False`` the
            centre cell has no path into its own niche latent at all.

            It is SLOWER than ``'gat'``, not faster: measured on an A5000 at batch
            512, k 10, one spatial-branch forward+backward is 3.1 ms (gat) vs 49.8
            ms (setattn) at 2000 genes and 28.5 vs 99.5 at 50000 peaks. That still
            projects to ~26 min for 400 epochs over 40k cells at 2000 HVGs, so it
            stays a minutes-scale model, but pick it for the inductive bias rather
            than for throughput.

            RAISE ``kl_spatial_weight`` WHEN USING IT. The set-attention encoder is
            a much stronger predictor than the GAT, and at SIMVI's ``0.01`` it wins
            the reconstruction outright and starves the intrinsic latents, which the
            KL then collapses onto their priors. Measured by rerunning
            ``run_haruka_spatial_example.py``; kNN-probe balanced accuracy on the
            RNA arm, chance 0.500 for lesion and niche:

            ==========================  ==========  ==========  ========
            config                      bg lesion   sal lesion  sp niche
            ==========================  ==========  ==========  ========
            gat, kl 0.01 (default)           0.932       0.957     0.826
            setattn, kl 0.01                 0.500       0.510     0.771
            setattn, kl 0.01, lam_mi 0       0.616       0.616     0.780
            setattn, kl 1.0                  0.952       0.952     0.577
            setattn, sw, kl 1.0              0.961       0.952     0.589
            ==========================  ==========  ==========  ========

            Row 2 is the failure: both intrinsic latents sit at chance while the
            niche latent holds everything. Row 3 turns the independence penalty off
            entirely and recovers only a fraction of the gap (0.62 against the 0.95
            the fix reaches), so that penalty is a minor contributor rather than the
            mechanism -- it is plain capacity competition for the reconstruction.
            Rows 4-5 raise ``kl_spatial_weight`` to 1.0 and restore both intrinsic
            latents fully, at the cost of niche sharpness (0.77 -> 0.58).

            On this synthetic ``'gat'`` therefore dominates: it is the only
            configuration that keeps all three latents AND the sharpest niche, at
            1/16th the cost. ``'setattn'`` is worth reaching for when the niche
            signal is a genuine SET property -- combinatorial in which cell types
            surround a cell rather than a weighted average of them -- which is
            exactly what this synthetic's single-sender-ligand design is not. The
            default is left at ``'gat'`` accordingly.
        setattn_tokens, setattn_token_dim, setattn_layers, setattn_heads
            Shape of the set-attention encoder. Defaults (16, 32, 2, 4) are sized
            for this class's own data scale -- 20k-100k cells on one GPU, minutes
            not days. Everything except the tokenizer is a fixed 45,844 parameters
            regardless of feature count; the tokenizer is ``n_features * 512``, so
            the total is ~1.1M at 2000 genes and ~25.6M at 50000 peaks. Drop
            ``setattn_tokens`` to 8 to halve both the parameter count and the
            attention cost. For reference the model this architecture is borrowed
            from is 217M parameters.
        setattn_self_query
            Let the centre cell supply a QUERY for the pooling over its neighbors,
            reproducing GATv2's ``add_self_loops=False``. The pooled VALUES are
            neighbor-derived either way. ``False`` (default) is the stronger
            guarantee and removes the centre cell from the spatial branch entirely.
        kl_spatial_weight
            KL weight on the spatial latent against N(0, I). SIMVI's 0.01, i.e.
            100x down-weighted relative to the other latents, which is what
            keeps this latent expressive instead of posterior-collapsed.
        me_reduction
            How the COVET reconstruction error is pooled, ``'sum'`` (default here)
            or ``'mean'``. ``'mean'`` is what the stock ``Haruka`` uses and is kept
            as the module-level default for backward compatibility, but it is not
            commensurate with the gene likelihood, which sums over features: the
            COVET term's share of the objective then falls like ``1/n_features``
            and reaches 1.6e-5 on the 16253-gene arcseq RNA, i.e. it cannot
            influence the fit at all. Under ``'sum'`` the share is 0.5-14% across
            every shape from the 60-gene synthetic example to the 52550-peak ATAC.
            Set ``'mean'`` to reproduce the parent class exactly.
        lam_mi, reg_to_use
            Independence penalty between the spatial latent and the contrastive
            latents. ``'mi'`` (default) wants ``lam_mi=5``; ``'mmd'`` is scaled
            entirely differently and the two are not interchangeable at a shared
            weight.

            ``'mi'`` is the default because ``'mmd'`` has no discriminative power
            at this class's latent widths. The MMD is taken over the joint
            ``[latent_dim + n_spatial_latent]``-dimensional cloud against N(0, I),
            and that estimate is noise-dominated past ~128 dimensions: at the
            ``latent_dim=256`` default it scores an exact copy of the intrinsic
            latent identically to an independent draw, so it enforces nothing.
            Constructing with ``reg_to_use='mmd'`` above that width warns.
        mmd_bandwidth
            ``'median'`` (default) or ``'simvi'``. Only relevant when
            ``reg_to_use='mmd'``. SIMVI's ``dim**2`` bandwidth saturates the kernel
            to the constant 1 at these widths; ``'median'`` adapts to the cloud and
            is 23-86x larger in magnitude, so ``lam_mi`` does not transfer between
            them. ``'simvi'`` exists for exact reproduction of the reference model.
        mi_debias
            Subtract a shuffled-latent null from the mutual-information penalty so
            it reads ~0 under independence instead of carrying the plug-in
            estimator's finite-sample bias (7.0 nats at ``latent_dim=256`` with a
            512-row batch, 62 on a short final batch). The null is detached, so the
            gradient is unchanged and only the reported value moves. Leaving this
            on is what makes the loss comparable across ``latent_dim`` and
            ``batch_size``, and safe to drive ``early_stopping`` from.
        penalize_salient_spatial
            Also force the SALIENT latent independent of the spatial one. Off by
            default, deliberately. SIMVI penalizes only intrinsic-vs-spatial, and
            that arm is well posed. Salient-vs-spatial is not: disease structure is
            spatially organized almost by definition -- a lesion is a contiguous
            region, so a cell's neighbours are lesional exactly when the cell is,
            the spatial latent carries the lesion signal, and forcing independence
            deletes the lesion from the salient latent rather than disentangling
            anything. On the synthetic benchmark the salient/lesion probe reads
            0.529 with this on against 0.952 with it off (chance 0.5), and the
            spatial latent's niche probe is also better with it off (0.708 vs
            0.606). It was harmless before only because the MMD penalty it rode on
            delivered no dependence-carrying gradient.
        mae_epochs
            Epochs of denoising corruption at the start of training, during which
            a random ``permutation_rate`` fraction of genes is shuffled across
            cells in the encoder input only. This stops the model routing a
            cell's own expression through the spatial branch and calling it
            interaction. ``None`` resolves to 20% of ``max_epochs`` at ``train()``
            time, matching SIMVI's 80/400 ratio; pass an int to fix it.
        anneal_epochs
            Ramps the spatial KL and the independence penalty in over this many
            epochs.
        """
        if self.input_adata is None:
            self.construct_cov()

        if niche_context is None:
            # OFF unless explicitly requested. niche_context is an opt-in feature: it
            # feeds each intrinsic encoder a pooled summary of the cell's neighborhood,
            # which is a different model, not a refinement of the old one. Defaulting it
            # to `n_spatial_latent == 0` turned the plain contrastive configuration --
            # the one every previous result was produced with -- into the niche-context
            # design silently. Because a lesion is a spatially contiguous region, the
            # niche summary carries lesion identity straight into the BACKGROUND
            # encoder, and the background latent stops being condition-free: measured on
            # the synthetic example, RNA background ARI fell from 0.96 to 0.49 vs the
            # layer truth while rising from 0.04 to 0.42 vs the lesion truth.
            niche_context = False

        use_spatial = n_spatial_latent > 0
        # Both designs gather neighbor expression by global row id, so both need the
        # index registered and the bank attached.
        use_graph = use_spatial or niche_context
        if use_graph and not self._graph_ready:
            print(
                f"[HarukaSpatial] building spatial graph (k={self.k_spatial}, "
                f"library_key={self.library_key!r})"
            )
            self.prepare_spatial_graph()

        if layer is None and "counts" in self.input_adata.layers:
            layer = "counts"

        scvi.external.ContrastiveVI.setup_anndata(
            self.input_adata,
            batch_key=batch_key,
            layer=layer,
            neighbor_idx_key=NBR_IDX_KEY if use_graph else None,
            neighbor_mask_key=NBR_MASK_KEY if use_graph else None,
        )

        # The denoising phase belongs to the SIMVI design only: it exists to stop a
        # cell's own expression leaking into the SEPARATE spatial latent, and with no
        # separate latent there is nothing to protect. It is inert under
        # niche_context anyway (the module gates it on n_spatial_latent), so this
        # only keeps the auto-resolution from claiming otherwise.
        self._mae_epochs_auto = use_spatial and mae_epochs is None
        self.model = scvi.external.ContrastiveVI(
            self.input_adata,
            n_me=self.cov_gene_num * self.cov_gene_num,
            n_salient_latent=latent_dim,
            n_background_latent=latent_dim,
            use_observed_lib_size=True,
            wasserstein_penalty=wasserstein_penalty,
            gene_likelihood=gene_likelihood,
            me_weight=me_weight,
            me_reduction=me_reduction,
            n_spatial_latent=n_spatial_latent,
            spatial_attach=spatial_attach,
            spatial_encoder=spatial_encoder,
            n_attention_heads=n_attention_heads,
            setattn_tokens=setattn_tokens,
            setattn_token_dim=setattn_token_dim,
            setattn_layers=setattn_layers,
            setattn_heads=setattn_heads,
            setattn_mlp_ratio=setattn_mlp_ratio,
            setattn_dropout=setattn_dropout,
            setattn_self_query=setattn_self_query,
            kl_spatial_weight=kl_spatial_weight,
            lam_mi=lam_mi,
            reg_to_use=reg_to_use,
            mmd_bandwidth=mmd_bandwidth,
            mi_debias=mi_debias,
            penalize_salient_spatial=penalize_salient_spatial,
            spatial_invariance_penalty=spatial_invariance_penalty,
            mask_salient_mean_on_background=mask_salient_mean_on_background,
            me_from_background_only=me_from_background_only,
            sw_n_proj=sw_n_proj,
            permutation_rate=permutation_rate,
            mae_epochs=0 if mae_epochs is None else mae_epochs,
            anneal_epochs=anneal_epochs,
            var_eps=var_eps,
            mmd_max_samples=mmd_max_samples,
            niche_context=niche_context,
            n_context=n_context,
            context_pool=context_pool,
            context_shared=context_shared,
            include_self=include_self,
            mask_rate=mask_rate,
        )
        self.n_spatial_latent = n_spatial_latent
        self.niche_context = niche_context

        self.background_indices = np.where(
            self.input_adata.obs[condition_key] != condition_label
        )[0]
        self.target_indices = np.where(
            self.input_adata.obs[condition_key] == condition_label
        )[0]

    def train(self, max_epochs=200, batch_size=512, **trainer_kwargs):
        if self._mae_epochs_auto:
            # Resolved here rather than in setup_model because the denoising phase
            # is an absolute epoch count and max_epochs is not known until now. A
            # fixed default would corrupt the whole of a short run.
            self.model.module.mae_epochs = max(1, int(round(self._mae_fraction * max_epochs)))
        super().train(max_epochs=max_epochs, batch_size=batch_size, **trainer_kwargs)

    # -- representations ---------------------------------------------------

    def extract_rep(self, n_layer=3):
        """Extract salient, background and (SIMVI design only) spatial representations.

        Each is then neighborhood-aggregated by cellcharter into
        ``X_cellcharter_{salient,background,spatial}``.

        Under the default niche-context design there is no third latent, so this
        produces exactly ``salient_rep`` and ``background_rep`` and their aggregated
        forms -- and no ``spatial_rep``. Both of those representations are already
        niche-aware, because the niche entered their encoders as input.
        """
        super().extract_rep(n_layer=n_layer)
        if self.n_spatial_latent == 0:
            return

        # output_adata is the target arm only, but the neighbor indices it carries
        # are global positions into the full object and the expression bank the
        # module gathers from is the full matrix, so this resolves correctly.
        self.output_adata.obsm["spatial_rep"] = self.model.get_latent_representation(
            self.output_adata, representation_kind="spatial"
        ).astype(np.float32)

        cc.gr.aggregate_neighbors(
            self.output_adata,
            n_layers=n_layer,
            use_rep="spatial_rep",
            out_key="X_cellcharter_spatial",
            sample_key="slice_id",
        )

        # 2x2 only: the salient spatial block, i.e. niche structure specific to the
        # condition arm. Exposed as a representation but deliberately NOT added to
        # cluster()'s spec list, so the set of cluster_haruka_* columns is unchanged.
        if getattr(self.model.module, "spatial_attach", None) == "both":
            self.output_adata.obsm["salient_spatial_rep"] = (
                self.model.get_latent_representation(
                    self.output_adata, representation_kind="salient_spatial"
                ).astype(np.float32)
            )
            cc.gr.aggregate_neighbors(
                self.output_adata,
                n_layers=n_layer,
                use_rep="salient_spatial_rep",
                out_key="X_cellcharter_salient_spatial",
                sample_key="slice_id",
            )

    # -- clustering --------------------------------------------------------

    def cluster(
        self,
        salient_cluster_num=4,
        background_cluster_number=4,
        seed=42,
        spatial_cluster_num=4,
        cluster_method="kmeans",
        scale=True,
        n_pcs=50,
        **cluster_kwargs,
    ):
        """Cluster each representation with ``FastCluster``.

        Writes ``cluster_haruka_salient``, ``cluster_haruka_background`` and,
        when the spatial branch is on, ``cluster_haruka_spatial``.

        ``cluster_method='kmeans'`` is the default because it is the fastest
        option, but note the measured tradeoff: on anisotropic domains a
        scaled + PCA-reduced GMM (``cluster_method='gmm'``) scores substantially
        better at still-competitive speed. Spatial domains are exactly the
        elongated, graded structures k-means handles worst, so prefer ``'gmm'``
        when domain quality matters more than wall clock.
        """
        specs = [
            ("salient", "X_cellcharter_salient", salient_cluster_num),
            ("background", "X_cellcharter_background", background_cluster_number),
        ]
        if self.n_spatial_latent > 0 and "X_cellcharter_spatial" in self.output_adata.obsm:
            specs.append(("spatial", "X_cellcharter_spatial", spatial_cluster_num))

        for kind, rep, n_clusters in specs:
            model = FastCluster(
                n_clusters=n_clusters,
                method=cluster_method,
                random_state=seed,
                scale=scale,
                n_pcs=n_pcs,
                **cluster_kwargs,
            )
            self.output_adata.obs[f"cluster_haruka_{kind}"] = model.fit_predict(
                self.output_adata, use_rep=rep
            )
            self.cluster_models_[kind] = model


class HarukaPeakSpatial(HarukaSpatial):
    """ATAC/peak counterpart of :class:`HarukaSpatial`.

    Identical pipeline with a PeakVI-style Poisson likelihood and COVET computed
    on the peak matrix.
    """

    def __init__(self, adata, cov_peak_num=32, seed=42, **kwargs):
        super().__init__(adata, cov_gene_num=cov_peak_num, seed=seed, **kwargs)


class HarukaMultiomicsSpatial:
    """Spatial multiomics Haruka: RNA + ATAC, each with a three-way factorization.

    Mirrors :class:`HarukaMultiomics.HarukaMultiomics` and keeps its attribute
    and column naming, so downstream code can be pointed at either. The
    difference is that each modality also yields a niche-induced representation,
    and the integrated clustering runs over three embedding families rather than
    two.

    Output columns follow ``cluster_haruka_<kind>_<modality>`` with ``kind`` in
    ``{salient, background, spatial}`` and ``modality`` in ``{rna, atac, multi}``.
    """

    def __init__(
        self,
        adata_rna,
        adata_atac,
        cov_gene_num=32,
        cov_peak_num=32,
        seed=42,
        k_spatial=10,
        spatial_key="spatial",
        library_key="slice_id",
    ):
        self.adata_rna = adata_rna
        self.adata_atac = adata_atac
        self.cov_gene_num = cov_gene_num
        self.cov_peak_num = cov_peak_num
        self.seed = seed
        self.k_spatial = k_spatial
        self.spatial_key = spatial_key
        self.library_key = library_key
        self.haruka_rna = None
        self.haruka_atac = None
        self.output_rna = None
        self.output_atac = None
        self.output_multi = None

    def _run_modality(
        self,
        cls,
        adata,
        cov_num,
        gene_likelihood,
        latent_dim,
        n_layer,
        salient_cluster_num,
        background_cluster_number,
        spatial_cluster_num,
        condition_key,
        condition_label,
        batch_key,
        max_epochs,
        batch_size,
        me_weight,
        n_spatial_latent,
        spatial_attach,
        kl_spatial_weight,
        lam_mi,
        reg_to_use,
        permutation_rate,
        mae_epochs,
        anneal_epochs,
        cluster_method,
        kwargs,
    ):
        seed_everything(self.seed)
        # Split by what setup_model actually accepts. HarukaMultiomics whitelists only
        # plan_kwargs/datasplitter_kwargs for training and sends everything else to
        # setup_model, which means a plain Trainer argument like enable_progress_bar
        # raises a TypeError from setup_model instead of reaching the trainer.
        setup_names = set(inspect.signature(cls.setup_model).parameters)
        setup_kwargs = {k: v for k, v in kwargs.items() if k in setup_names}
        train_kwargs = {k: v for k, v in kwargs.items() if k not in setup_names}

        model = cls(
            adata,
            cov_num,
            seed=self.seed,
            k_spatial=self.k_spatial,
            spatial_key=self.spatial_key,
            library_key=self.library_key,
        )
        model.construct_cov()
        model.setup_model(
            latent_dim=latent_dim,
            gene_likelihood=gene_likelihood,
            batch_key=batch_key,
            condition_key=condition_key,
            condition_label=condition_label,
            me_weight=me_weight,
            n_spatial_latent=n_spatial_latent,
            spatial_attach=spatial_attach,
            kl_spatial_weight=kl_spatial_weight,
            lam_mi=lam_mi,
            reg_to_use=reg_to_use,
            permutation_rate=permutation_rate,
            mae_epochs=mae_epochs,
            anneal_epochs=anneal_epochs,
            **setup_kwargs,
        )
        model.train(max_epochs=max_epochs, batch_size=batch_size, **train_kwargs)
        model.extract_rep(n_layer=n_layer)
        model.cluster(
            salient_cluster_num=salient_cluster_num,
            background_cluster_number=background_cluster_number,
            spatial_cluster_num=spatial_cluster_num,
            seed=self.seed,
            cluster_method=cluster_method,
        )
        return model

    def run_rna(
        self,
        latent_dim=256,
        gene_likelihood="poisson",
        n_layer=3,
        salient_cluster_num=4,
        background_cluster_number=4,
        spatial_cluster_num=4,
        condition_key="condition",
        condition_label="disease",
        batch_key="slice_id",
        max_epochs=200,
        batch_size=512,
        me_weight_rna=0.3,
        n_spatial_latent=0,
        spatial_attach="background",
        kl_spatial_weight=0.01,
        lam_mi=5.0,
        reg_to_use="mi",
        permutation_rate=0.5,
        mae_epochs=None,
        anneal_epochs=50,
        cluster_method="kmeans",
        **kwargs,
    ):
        """Run the spatial pipeline on RNA.

        Any additional keyword accepted by :meth:`HarukaSpatial.setup_model` is
        forwarded to it, so the set-attention encoder is reachable from here without
        a second signature to keep in sync::

            multi.run_rna(spatial_encoder="setattn", setattn_layers=2)

        Anything it does not accept goes to the trainer instead.
        """
        self.haruka_rna = self._run_modality(
            HarukaSpatial,
            self.adata_rna,
            self.cov_gene_num,
            gene_likelihood,
            latent_dim,
            n_layer,
            salient_cluster_num,
            background_cluster_number,
            spatial_cluster_num,
            condition_key,
            condition_label,
            batch_key,
            max_epochs,
            batch_size,
            me_weight_rna,
            n_spatial_latent,
            spatial_attach,
            kl_spatial_weight,
            lam_mi,
            reg_to_use,
            permutation_rate,
            mae_epochs,
            anneal_epochs,
            cluster_method,
            kwargs,
        )
        self.output_rna = self.haruka_rna.output_adata
        return self.output_rna

    def run_atac(
        self,
        latent_dim=256,
        n_layer=3,
        salient_cluster_num=4,
        background_cluster_number=4,
        spatial_cluster_num=4,
        condition_key="condition",
        condition_label="disease",
        batch_key="slice_id",
        max_epochs=200,
        batch_size=512,
        me_weight_atac=0.3,
        n_spatial_latent=0,
        spatial_attach="background",
        kl_spatial_weight=0.01,
        lam_mi=5.0,
        reg_to_use="mi",
        permutation_rate=0.5,
        mae_epochs=None,
        anneal_epochs=50,
        cluster_method="kmeans",
        **kwargs,
    ):
        """Run the spatial pipeline on ATAC with a PeakVI-style Poisson likelihood.

        As in :meth:`run_rna`, extra keywords are routed by signature: those
        :meth:`HarukaSpatial.setup_model` accepts (``spatial_encoder``, the
        ``setattn_*`` family, ...) go there, the rest go to the trainer.
        """
        self.haruka_atac = self._run_modality(
            HarukaPeakSpatial,
            self.adata_atac,
            self.cov_peak_num,
            "poisson",
            latent_dim,
            n_layer,
            salient_cluster_num,
            background_cluster_number,
            spatial_cluster_num,
            condition_key,
            condition_label,
            batch_key,
            max_epochs,
            batch_size,
            me_weight_atac,
            n_spatial_latent,
            spatial_attach,
            kl_spatial_weight,
            lam_mi,
            reg_to_use,
            permutation_rate,
            mae_epochs,
            anneal_epochs,
            cluster_method,
            kwargs,
        )
        self.output_atac = self.haruka_atac.output_adata
        return self.output_atac

    def concat_embeddings_and_cluster(
        self,
        multi_salient_cluster_num=4,
        multi_background_cluster_number=4,
        multi_spatial_cluster_num=4,
        sample_key="slice_id",
        cluster_method="kmeans",
        seed=None,
    ):
        """Align RNA and ATAC by cell, concatenate each embedding family, cluster.

        Produces ``X_multi_salient``, ``X_multi_background`` and, when both
        modalities carried a spatial branch, ``X_multi_spatial``.
        """
        if self.output_rna is None or self.output_atac is None:
            raise RuntimeError("Run run_rna() and run_atac() first.")
        seed = self.seed if seed is None else seed

        common = self.output_rna.obs.index.intersection(self.output_atac.obs.index)
        if len(common) == 0:
            raise ValueError("No common cell IDs between RNA and ATAC output.")
        rna_idx = [self.output_rna.obs.index.get_loc(c) for c in common]
        atac_idx = [self.output_atac.obs.index.get_loc(c) for c in common]

        multi_obsm = {}
        families = []
        for kind in ("salient", "background", "spatial"):
            key = f"X_cellcharter_{kind}"
            if key not in self.output_rna.obsm or key not in self.output_atac.obsm:
                continue
            rna_block = self.output_rna.obsm[key][rna_idx]
            atac_block = self.output_atac.obsm[key][atac_idx]
            multi_obsm[f"X_cellcharter_{kind}_rna"] = rna_block
            multi_obsm[f"X_cellcharter_{kind}_atac"] = atac_block
            multi_obsm[f"X_multi_{kind}"] = np.concatenate([rna_block, atac_block], axis=1)
            families.append(kind)

        multi_obs = self.output_rna.obs.loc[common].copy()
        if "spatial" in self.output_rna.obsm:
            multi_obsm["spatial"] = self.output_rna.obsm["spatial"][rna_idx]

        self.output_multi = ad.AnnData(X=multi_obsm["X_multi_salient"], obs=multi_obs)
        self.output_multi.obsm = multi_obsm

        for kind in families:
            col = f"cluster_haruka_{kind}"
            for modality, source in (("rna", self.output_rna), ("atac", self.output_atac)):
                if col in source.obs.columns:
                    self.output_multi.obs[f"{col}_{modality}"] = source.obs.loc[
                        common, col
                    ].values

        n_clusters = {
            "salient": multi_salient_cluster_num,
            "background": multi_background_cluster_number,
            "spatial": multi_spatial_cluster_num,
        }
        self.cluster_models_ = {}
        for kind in families:
            model = FastCluster(
                n_clusters=n_clusters[kind], method=cluster_method, random_state=seed
            )
            self.output_multi.obs[f"cluster_haruka_{kind}_multi"] = model.fit_predict(
                self.output_multi, use_rep=f"X_multi_{kind}"
            )
            self.cluster_models_[kind] = model

        if "spatial" in self.output_multi.obsm and sample_key in self.output_multi.obs.columns:
            sq.gr.spatial_neighbors(
                self.output_multi, library_key=sample_key, spatial_key="spatial"
            )
            try:
                cc.gr.remove_long_links(self.output_multi)
            except Exception:
                pass

        return self.output_multi
