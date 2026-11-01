"""Model class for contrastive-VI for single cell expression data."""

from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse
import torch

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    get_max_epochs_heuristic,
    scrna_raw_counts_properties,
    use_distributed_sampler,
)
from scvi.model.base import BaseModelClass
from scvi.model.base._de_core import _de_core
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp

from ._contrastive_data_splitting import ContrastiveDataSplitter
from ._module import SPATIAL_REGISTRY_KEYS, ContrastiveVAE

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from anndata import AnnData

logger = logging.getLogger(__name__)
Number = int | float


class SpatialContrastiveTrainingPlan(TrainingPlan):
    """Training plan that tells the module which epoch it is in.

    scvi's plan forwards only ``kl_weight`` to the module, but the spatial branch also
    needs the epoch to schedule its annealing weight and to end the denoising phase.
    Harmless when the spatial branch is off: the attribute is simply never read.
    """

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.module.current_epoch_ = self.current_epoch


class ContrastiveVI(BaseModelClass):
    """contrastive variational inference :cite:p:`Weinberger23`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via
        :meth:`~scvi.model.ContrastiveVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_background_latent
        Dimensionality of the background shared latent space.
    n_salient_latent
        Dimensionality of the salient latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution.
    wasserstein_penalty
        Weight of the Wasserstein distance loss that further discourages background
        shared variations from leaking into the salient latent space.
    niche_context, n_context, context_pool, context_shared, include_self, mask_rate
        Make the EXISTING background and salient latents niche-aware by giving each
        encoder its own pooled summary of the cell's spatial neighborhood as extra INPUT,
        rather than adding a third latent. Requires ``neighbor_idx_key`` at
        :meth:`setup_anndata`, and is mutually exclusive with ``n_spatial_latent > 0``.
        See :class:`~scvi.external.contrastivevi._module.ContrastiveVAE` for the full
        description of each.

    Notes
    -----
    See further usage examples in the following tutorial:

    1. :doc:`/tutorials/notebooks/scrna/contrastiveVI_tutorial`
    """

    _module_cls = ContrastiveVAE
    _data_splitter_cls = ContrastiveDataSplitter
    _training_plan_cls = SpatialContrastiveTrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_me: int,
        n_hidden: int = 128,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        wasserstein_penalty: float = 0,
        gene_likelihood = 'nb',
        me_weight = 0.3,
        me_reduction: str = "mean",
        n_spatial_latent: int = 0,
        spatial_attach: str = "background",
        spatial_encoder: str = "gat",
        n_attention_heads: int = 1,
        gat_self_query: bool = True,
        setattn_tokens: int = 16,
        setattn_token_dim: int = 32,
        setattn_layers: int = 2,
        setattn_heads: int = 4,
        setattn_mlp_ratio: int = 4,
        setattn_dropout: float = 0.0,
        setattn_self_query: bool = False,
        kl_spatial_weight: float = 0.01,
        lam_mi: float = 5.0,
        reg_to_use: str = "mi",
        mmd_bandwidth: str = "median",
        mi_debias: bool = True,
        penalize_salient_spatial: bool = False,
        salient_within_penalty: float = 0.0,
        spatial_pair_penalty: float = 0.0,
        condition_only_salient_spatial: bool = False,
        salient_prior_components: int = 0,
        salient_assignment_penalty: float = 0.0,
        salient_assignment_temperature: float = 1.0,
        paired_slot_reg_weight: float = 0.0,
        paired_slot_reg_alpha: float = 1e-3,
        paired_slot_reg_start_epoch: int = 0,
        paired_slot_reg_ramp_epochs: int = 0,
        background_representation: str = "full",
        spatial_invariance_penalty: float = 0.5,
        background_mmd_penalty: float = 0.0,
        background_wasserstein_penalty: float = 0.0,
        zs_independence_penalty: float = 0.0,
        symmetric_zs_independence: bool = False,
        detach_target_z_for_composition: bool = False,
        batch_injection_mode: str = "all",
        mask_salient_mean_on_background: bool = False,
        me_from_background_only: bool = False,
        sw_n_proj: int = 64,
        permutation_rate: float = 0.5,
        mae_epochs: int = 0,
        anneal_epochs: int = 50,
        var_eps: float = 1e-4,
        mmd_max_samples: int = 512,
        niche_context: bool = False,
        n_context: int = 128,
        context_pool: str = "setattn",
        context_shared: bool = False,
        include_self: bool = True,
        mask_rate: float = 0.0,
    ) -> None:
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        library_log_means, library_log_vars = None, None
        if not use_observed_lib_size:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.n_input = self.summary_stats.n_vars - n_me
        
        #print(self.n_input)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars - n_me,
            n_input_full=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            wasserstein_penalty=wasserstein_penalty,
            gene_likelihood = gene_likelihood,
            me_weight = me_weight,
            me_reduction=me_reduction,
            n_spatial_latent=n_spatial_latent,
            spatial_attach=spatial_attach,
            spatial_encoder=spatial_encoder,
            n_attention_heads=n_attention_heads,
            gat_self_query=gat_self_query,
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
            salient_within_penalty=salient_within_penalty,
            spatial_pair_penalty=spatial_pair_penalty,
            condition_only_salient_spatial=condition_only_salient_spatial,
            salient_prior_components=salient_prior_components,
            salient_assignment_penalty=salient_assignment_penalty,
            salient_assignment_temperature=salient_assignment_temperature,
            paired_slot_reg_weight=paired_slot_reg_weight,
            paired_slot_reg_alpha=paired_slot_reg_alpha,
            paired_slot_reg_start_epoch=paired_slot_reg_start_epoch,
            paired_slot_reg_ramp_epochs=paired_slot_reg_ramp_epochs,
            background_representation=background_representation,
            spatial_invariance_penalty=spatial_invariance_penalty,
            background_mmd_penalty=background_mmd_penalty,
            background_wasserstein_penalty=background_wasserstein_penalty,
            zs_independence_penalty=zs_independence_penalty,
            symmetric_zs_independence=symmetric_zs_independence,
            detach_target_z_for_composition=detach_target_z_for_composition,
            batch_injection_mode=batch_injection_mode,
            mask_salient_mean_on_background=mask_salient_mean_on_background,
            me_from_background_only=me_from_background_only,
            sw_n_proj=sw_n_proj,
            permutation_rate=permutation_rate,
            mae_epochs=mae_epochs,
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
        self._model_summary_string = (
            f"ContrastiveVI Model with the following params: \nn_hidden: {n_hidden}, "
            f"n_background_latent: {n_background_latent}, n_salient_latent: {n_salient_latent}, "
            f"n_layers: {n_layers}, dropout_rate: {dropout_rate}, "
            f"use_observed_lib_size: {use_observed_lib_size}, "
            f"wasserstein_penalty: {wasserstein_penalty}, "
            f"batch_injection_mode: {batch_injection_mode}, "
            f"n_spatial_latent: {n_spatial_latent}, "
            f"spatial_encoder: {spatial_encoder if n_spatial_latent else 'off'}, "
            f"niche_context: {context_pool if niche_context else 'off'}"
            + (
                f" (n_context: {n_context}, "
                f"{'shared' if context_shared else 'two blocks'}, "
                f"include_self: {include_self})"
                if niche_context
                else ""
            )
        )
        self.init_params_ = self._get_init_params(locals())

        # Both designs gather neighbor expression by global row id, so both need the
        # resident bank.
        if n_spatial_latent > 0 or niche_context:
            self.attach_neighbor_bank()

    @staticmethod
    def build_neighbor_index(
        adata: AnnData,
        k: int = 10,
        spatial_key: str = "spatial",
        library_key: str | None = "slice_id",
        remove_long_links: bool = True,
        distance_percentile: float = 99.0,
        tie_tol: float | None = 0.02,
        degenerate_frac: float = 0.6,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the within-slice k-nearest-neighbor index the spatial branch gathers over.

        Returns ``(idx, mask)`` of shape ``(n_obs, k)``. ``idx`` holds GLOBAL row
        positions into ``adata``; padded slots point at the cell itself so a gather is
        always in range, and ``mask`` marks them invalid so they contribute nothing.

        A cell is never its own neighbor, which is what makes the pooled representation
        neighbor-induced rather than a copy of the cell.

        Degenerate (lattice) geometries
        -------------------------------
        On a regular array -- which is what DBiT / ARC-seq / Visium data is, and what
        the arcseq sections are (8485 spots on a 99 x 98 lattice) -- neighbor distances
        quantize into shells that hold 4, 4 and 4 sites. A plain k-nearest query with
        ``k=10`` therefore takes 4 + 4 and then two ARBITRARY of the four tied
        third-shell sites, and which two is decided by sub-pixel coordinate rounding
        rather than by anything biological. Measured on the real coordinates, that
        gives 71% of interior spots a stencil with a net displacement of 2.1 lattice
        pitches, and the bias is spatially autocorrelated, so it does not average out
        in any local niche readout.

        This is corrected by refusing to admit a PARTIAL shell: when the k-th neighbor
        is tied with the first excluded candidate, the tied tail is masked off and the
        stencil stops at the last complete shell. At ``k=10`` on the real lattice that
        turns "4 + 4 + 2 of 4" into "4 + 4", dropping interior asymmetry from 71.3% to
        3.7% and mean net displacement from 2.14 to 0.20 pitches -- the same floor that
        shell-closing values of k (4, 8, 12) reach natively.

        The correction is gated on measured degeneracy so that it is a BIT-EXACT no-op
        on continuous coordinates, which is the regime SIMVI's ``k=10`` was calibrated
        in. The gate statistic is the fraction of cells whose boundary distance ratio
        falls within ``tie_tol``: measured 0.992 on the real lattice against 0.289
        (scattered) and 0.295 (clustered), so the two regimes separate with a wide
        margin. Verified no-op on scattered, clustered and jittered-grid geometries,
        and on the real lattice at every shell-closing k.

        Note that ``remove_long_links`` does not and cannot substitute for this: on a
        lattice it drops only 0.1% of edges, because 20% of the k=10 entries ARE the
        third-shell links, so the 99th-percentile threshold sits above them.

        Parameters
        ----------
        adata
            Object carrying ``obsm[spatial_key]``.
        k
            Neighbors per cell. On a degenerate geometry the effective count is the
            largest complete shell at or below k, so ``k=10`` behaves as 8 on a square
            lattice and as 10 everywhere else.
        spatial_key
            Key in ``obsm`` holding the coordinates.
        library_key
            Column in ``obs`` identifying the section. Neighbors are never drawn across
            sections. ``None`` treats the whole object as one section.
        remove_long_links
            Drop implausibly long edges, thresholded per section rather than globally,
            so a sparse section is not trimmed against a dense one's spacing.
        distance_percentile
            Percentile of within-section neighbor distances used as that threshold.
        tie_tol
            Relative tolerance for calling two neighbor distances tied. ``None``
            restores the plain hard-truncating behaviour. The default sits above a
            raster's sub-pixel pitch jitter and below the generic spacing of a
            continuous point process, where consecutive kNN distances differ by
            ``sqrt((k+1)/k) - 1`` ~ 5% at k=10.
        degenerate_frac
            Fraction of a section's cells that must show a tied boundary before the
            section is treated as shell-structured.
        verbose
            Print one line per section when the correction actually fires.
        """
        from sklearn.neighbors import NearestNeighbors

        n = adata.n_obs
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        if coords.shape[0] != n:
            raise ValueError(
                f"adata.obsm[{spatial_key!r}] has {coords.shape[0]} rows but adata has {n}"
            )

        if library_key is None:
            labels = np.zeros(n, dtype=np.int64)
        else:
            if library_key not in adata.obs.columns:
                raise KeyError(
                    f"library_key {library_key!r} is not in adata.obs; refusing to build "
                    "a graph whose neighbors would cross sections."
                )
            labels = np.asarray(adata.obs[library_key])

        idx = np.zeros((n, k), dtype=np.int64)
        mask = np.zeros((n, k), dtype=bool)

        for val in pd.unique(labels):
            rows = np.flatnonzero(labels == val)
            m = rows.size
            idx[rows] = rows[:, None]
            if m < 2:
                continue
            kk = int(min(k, m - 1))
            # One neighbor beyond kk, used only as lookahead for the tie test.
            n_query = int(min(kk + 2, m))
            nn = NearestNeighbors(n_neighbors=n_query).fit(coords[rows])
            dist, loc = nn.kneighbors(coords[rows])

            # Duplicate coordinates can push a cell off column 0 of its own result, so
            # drop self by identity rather than by position.
            keep = loc != np.arange(m)[:, None]
            order = np.argsort(~keep, axis=1, kind="stable")
            avail = int(min(n_query - 1, loc.shape[1]))
            loc = np.take_along_axis(loc, order, axis=1)[:, :avail]
            dist = np.take_along_axis(dist, order, axis=1)[:, :avail]

            # Refuse a partial degenerate shell; see the class docstring.
            cut = np.full(m, kk, dtype=np.int64)
            if tie_tol is not None and avail > kk and kk >= 2:
                tied = dist[:, kk] <= dist[:, kk - 1] * (1.0 + tie_tol)
                if tied.mean() >= degenerate_frac:
                    # gaps[:, j] : a real distance jump between position j and j+1.
                    gaps = dist[:, 1 : kk + 1] > dist[:, :kk] * (1.0 + tie_tol)
                    has_gap = gaps.any(axis=1)
                    # Largest j <= kk that sits on a shell boundary. When the boundary
                    # itself is a real gap this is kk, i.e. nothing is trimmed.
                    last_gap = kk - np.argmax(gaps[:, ::-1], axis=1)
                    # A cell whose whole neighborhood is one degenerate blob has no
                    # complete shell to fall back on; keep all k rather than none.
                    cut = np.maximum(np.where(has_gap, last_gap, kk), 1)
                    if verbose and int((cut < kk).sum()):
                        print(
                            f"[build_neighbor_index] section {val!r}: lattice-degenerate "
                            f"neighbor distances; trimming the partial shell for "
                            f"{int((cut < kk).sum())}/{m} cells "
                            f"({cut.mean():.2f} of {kk} neighbors kept on average) so "
                            "the stencil stays directionally unbiased"
                        )

            loc, dist = loc[:, :kk], dist[:, :kk]
            ok = np.arange(kk)[None, :] < cut[:, None]
            if remove_long_links and dist.size:
                ok &= dist <= np.percentile(dist, distance_percentile)
                ok[:, 0] = True

            idx[np.ix_(rows, np.arange(kk))] = rows[loc]
            mask[np.ix_(rows, np.arange(kk))] = ok

        idx = np.where(mask, idx, np.arange(n, dtype=np.int64)[:, None])
        return idx.astype(np.int64), mask

    def attach_neighbor_bank(self, adata: AnnData | None = None) -> None:
        """Install the expression matrix the spatial branch gathers neighbors from.

        Called automatically at construction. It must be called again after
        :meth:`load`, because the bank is a non-persistent buffer and is deliberately
        kept out of the saved state.
        """
        if self.module.n_spatial_latent == 0 and not self.module.niche_context:
            raise ValueError(
                "attach_neighbor_bank requires n_spatial_latent > 0 or niche_context=True"
            )

        manager = (
            self.adata_manager
            if adata is None
            else self.get_anndata_manager(adata, required=True)
        )
        x = manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        if scipy.sparse.issparse(x):
            x = x.toarray()
        x = np.asarray(x)[:, : self.n_input]
        batch = np.asarray(manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).reshape(-1)

        if SPATIAL_REGISTRY_KEYS.NEIGHBOR_IDX_KEY in manager.data_registry:
            nbr = np.asarray(
                manager.get_from_registry(SPATIAL_REGISTRY_KEYS.NEIGHBOR_IDX_KEY)
            )
            # Checked once here rather than per step: neighbor indices are global row
            # positions, so a bank built from a subset would gather silently wrong rows.
            if nbr.size and int(nbr.max()) >= x.shape[0]:
                raise ValueError(
                    f"neighbor index reaches row {int(nbr.max())} but the bank has only "
                    f"{x.shape[0]} rows. Attach the bank from the full AnnData the graph "
                    "was built on."
                )
        self.module.attach_neighbor_bank(x, batch)

    @devices_dsp.dedent
    def train(
        self,
        background_indices: list[int],
        target_indices: list[int],
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        datasplitter_kwargs
            Additional keyword arguments passed into
            :class:`~scvi.dataloaders.ContrastiveDataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        plan_kwargs = plan_kwargs or {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            background_indices=background_indices,
            target_indices=target_indices,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            shuffle_set_split=shuffle_set_split,
            distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
            load_sparse_tensor=load_sparse_tensor,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        neighbor_idx_key: str | None = None,
        neighbor_mask_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        neighbor_idx_key
            Key in ``adata.obsm`` holding the ``(n_obs, k)`` neighbor index built by
            :meth:`build_neighbor_index`. Required when training with
            ``n_spatial_latent > 0`` or with ``niche_context=True``.
        neighbor_mask_key
            Key in ``adata.obsm`` holding the matching ``(n_obs, k)`` validity mask. If
            omitted, every neighbor slot is treated as valid.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # Appended only on request: ObsmField has no required=False and raises KeyError
        # on a missing key, so an unconditional append would break every existing setup.
        if neighbor_idx_key is not None:
            anndata_fields.append(
                ObsmField(SPATIAL_REGISTRY_KEYS.NEIGHBOR_IDX_KEY, neighbor_idx_key)
            )
        if neighbor_mask_key is not None:
            anndata_fields.append(
                ObsmField(SPATIAL_REGISTRY_KEYS.NEIGHBOR_MASK_KEY, neighbor_mask_key)
            )
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """Returns the background or salient latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind
            One of "background", "salient", "spatial", or "salient_spatial".
            Spatial kinds require ``n_spatial_latent > 0``. When the model uses
            ``background_representation="private"``, the background result excludes
            the trailing dimensions reserved as the internal GAT source.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = [
            "background",
            "salient",
            "spatial",
            "salient_spatial",
        ]
        if representation_kind not in available_representation_kinds:
            raise ValueError(
                f"representation_kind = {representation_kind} is not one of"
                f" {available_representation_kinds}"
            )
        if (
            representation_kind in ("spatial", "salient_spatial")
            and self.module.n_spatial_latent == 0
        ):
            raise ValueError(
                f"representation_kind={representation_kind!r} requires the model to "
                "have been constructed with n_spatial_latent > 0."
            )
        if (
            representation_kind == "salient_spatial"
            and self.module.spatial_attach != "both"
        ):
            raise ValueError(
                "representation_kind='salient_spatial' requires "
                "spatial_attach='both'; there is only one spatial latent otherwise."
            )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            extra = {}
            if self.module.n_spatial_latent > 0 or self.module.niche_context:
                # Neighbor indices are global row positions into the AnnData the bank was
                # attached from, so they stay valid even when adata here is a subset.
                extra["neighbor_idx"] = tensors[SPATIAL_REGISTRY_KEYS.NEIGHBOR_IDX_KEY]
                extra["neighbor_mask"] = tensors.get(
                    SPATIAL_REGISTRY_KEYS.NEIGHBOR_MASK_KEY, None
                )
            outputs = self.module._generic_inference(x=x, index=self.n_input, batch_index=batch_index, n_samples=1, **extra)

            if representation_kind == "background":
                latent_m = outputs["qz_m"]
                latent_sample = outputs["z"]
                if self.module.background_representation == "private":
                    # The trailing dimensions are the GAT's source features. They
                    # remain available to reconstruction and spatial inference but
                    # are not part of the public intrinsic representation.
                    stop = -self.module.n_spatial_latent
                    latent_m = latent_m[..., :stop]
                    latent_sample = latent_sample[..., :stop]
            elif representation_kind == "spatial":
                latent_m = outputs["qsp_m"]
                latent_sample = outputs["z_sp"]
            elif representation_kind == "salient_spatial":
                latent_m = outputs["qssp_m"]
                latent_sample = outputs["s_sp"]
            else:
                latent_m = outputs["qs_m"]
                latent_sample = outputs["s"]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | str = 1.0,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
    ) -> dict[str, np.ndarray | pd.DataFrame]:
        """Returns the normalized (decoded) gene expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes. This can save
            memory when working with large datasets and few genes are of interest.
        library_size
            Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            The number of random samples in `adata` to use.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
            A dictionary with keys "background" and "salient", with value as follows.
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and"
                    " return_mean is False, returning np.ndarray",
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "px_rate"
            scaling = 1
        else:
            generative_output_key = "px_scale"
            scaling = library_size

        background_exprs = []
        salient_exprs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            background_per_batch_exprs = []
            salient_per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_index = torch.ones_like(batch_index) * batch
                inference_outputs = self.module._generic_inference(
                    x=x, batch_index=batch_index, n_samples=n_samples
                )
                z = inference_outputs["z"]
                s = inference_outputs["s"]
                library = inference_outputs["library"]
                background_generative_outputs = self.module._generic_generative(
                    z=z, s=torch.zeros_like(s), library=library, batch_index=batch_index
                )
                salient_generative_outputs = self.module._generic_generative(
                    z=z, s=s, library=library, batch_index=batch_index
                )
                background_outputs = self._preprocess_normalized_expression(
                    background_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                background_per_batch_exprs.append(background_outputs)
                salient_outputs = self._preprocess_normalized_expression(
                    salient_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                salient_per_batch_exprs.append(salient_outputs)

            background_per_batch_exprs = np.stack(
                background_per_batch_exprs
            )  # Shape is (len(transform_batch) x batch_size x n_var).
            salient_per_batch_exprs = np.stack(salient_per_batch_exprs)
            background_exprs += [background_per_batch_exprs.mean(0)]
            salient_exprs += [salient_per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            background_exprs = np.concatenate(background_exprs, axis=-2)
            salient_exprs = np.concatenate(salient_exprs, axis=-2)
        else:
            background_exprs = np.concatenate(background_exprs, axis=0)
            salient_exprs = np.concatenate(salient_exprs, axis=0)
        if n_samples > 1 and return_mean:
            background_exprs = background_exprs.mean(0)
            salient_exprs = salient_exprs.mean(0)

        if return_numpy is None or return_numpy is False:
            genes = adata.var_names[gene_mask]
            samples = adata.obs_names[indices]
            background_exprs = pd.DataFrame(background_exprs, columns=genes, index=samples)
            salient_exprs = pd.DataFrame(salient_exprs, columns=genes, index=samples)
        return {"background": background_exprs, "salient": salient_exprs}

    @torch.inference_mode()
    def get_salient_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | str = 1.0,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
    ) -> np.ndarray | pd.DataFrame:
        """Returns the normalized (decoded) gene expression.

        Gene expressions are decoded from both the background and salient latent space.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size
            Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            The number of random samples in `adata` to use.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        exprs = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            transform_batch=transform_batch,
            gene_list=gene_list,
            library_size=library_size,
            n_samples=n_samples,
            n_samples_overall=n_samples_overall,
            batch_size=batch_size,
            return_mean=return_mean,
            return_numpy=return_numpy,
        )
        return exprs["salient"]

    @torch.inference_mode()
    def get_specific_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | str = 1,
        n_samples: int = 1,
        n_samples_overall: int | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        expression_type: str | None = None,
        indices_to_return_salient: Sequence[int] | None = None,
    ):
        """Returns the normalized (decoded) gene expression.

        Gene expressions are decoded from either the background or salient latent space.
        One of `expression_type` or `indices_to_return_salient` should have an input
        argument.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size
            Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            The number of random samples in `adata` to use.
        batch_size
            Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.
        expression_type
            One of {"salient", "background"} to specify the type of
            normalized expression to return.
        indices_to_return_salient
            If `indices` is a subset of `indices_to_return_salient`, normalized
            expressions derived from background and salient latent embeddings are
            returned. If `indices` is not `None` and is not a subset of
            `indices_to_return_salient`, normalized expressions derived only from
            background latent embeddings are returned.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        is_expression_type_none = expression_type is None
        is_indices_to_return_salient_none = indices_to_return_salient is None
        if is_expression_type_none and is_indices_to_return_salient_none:
            raise ValueError(
                "Both expression_type and indices_to_return_salient are None! "
                "Exactly one of them needs to be supplied with an input argument."
            )
        elif (not is_expression_type_none) and (not is_indices_to_return_salient_none):
            raise ValueError(
                "Both expression_type and indices_to_return_salient have an input "
                "argument! Exactly one of them needs to be supplied with an input "
                "argument."
            )
        else:
            exprs = self.get_normalized_expression(
                adata=adata,
                indices=indices,
                transform_batch=transform_batch,
                gene_list=gene_list,
                library_size=library_size,
                n_samples=n_samples,
                n_samples_overall=n_samples_overall,
                batch_size=batch_size,
                return_mean=return_mean,
                return_numpy=return_numpy,
            )
            if not is_expression_type_none:
                return exprs[expression_type]
            else:
                if indices is None:
                    indices = np.arange(adata.n_obs)
                if set(indices).issubset(set(indices_to_return_salient)):
                    return exprs["salient"]
                else:
                    return exprs["background"]

    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | (Sequence[bool] | str) | None = None,
        idx2: Sequence[int] | (Sequence[bool] | str) | None = None,
        mode: str = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        target_idx: Sequence[int] | None = None,
        n_samples: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        r"""Performs differential expression analysis.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        groupby
            The key of the observations grouping to consider.
        group1
            Subset of groups, e.g. ["g1", "g2", "g3"], to which comparison shall be
            restricted, or all groups in `groupby` (default).
        group2
            If `None`, compare each group in `group1` to the union of the rest of
            the groups in `groupby`. If a group identifier, compare with respect to this
            group.
        idx1
            `idx1` and `idx2` can be used as an alternative to the AnnData keys.
            Custom identifier for `group1` that can be of three sorts:
            (1) a boolean mask, (2) indices, or (3) a string. If it is a string, then
            it will query indices that verifies conditions on adata.obs, as described
            in `pandas.DataFrame.query()`. If `idx1` is not `None`, this option
            overrides `group1` and `group2`.
        idx2
            Custom identifier for `group2` that has the same properties as `idx1`.
            By default, includes all cells not specified in `idx1`.
        mode:
            Method for differential expression. See
            https://docs.scvi-tools.org/en/0.14.1/user_guide/background/differential_expression.html
            for more details.
        delta
            Specific case of region inducing differential expression. In this case,
            we suppose that R\[-delta, delta] does not induce differential expression
            (change model default case).
        batch_size
            Mini-batch size for data loading into model. Defaults to
            scvi.settings.batch_size.
        all_stats
            Concatenate count statistics (e.g., mean expression group 1) to DE
            results.
        batch_correction
            Whether to correct for batch effects in DE inference.
        batchid1
            Subset of categories from `batch_key` registered in `setup_anndata`,
            e.g. ["batch1", "batch2", "batch3"], for `group1`. Only used if
            `batch_correction` is `True`, and by default all categories are used.
        batchid2
            Same as `batchid1` for `group2`. `batchid2` must either have null
            intersection with `batchid1`, or be exactly equal to `batchid1`. When the
            two sets are exactly equal, cells are compared by decoding on the same
            batch. When sets have null intersection, cells from `group1` and `group2`
            are decoded on each group in `group1` and `group2`, respectively.
        fdr_target
            Tag features as DE based on posterior expected false discovery rate.
        silent
            If `True`, disables the progress bar. Default: `False`.
        target_idx
            If not `None`, a boolean or integer identifier should be used for
            cells in the contrastive target group. Normalized expression values derived
            from both salient and background latent embeddings are used when
            {group1, group2} is a subset of the target group, otherwise background
            normalized expression values are used.
        kwargs: Keyword args for
            `scvi.model.base.DifferentialComputation.get_bayes_factors`.

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)
        col_names = adata.var_names

        if target_idx is not None:
            target_idx = np.array(target_idx)
            if target_idx.dtype is np.dtype("bool"):
                assert (
                    len(target_idx) == adata.n_obs
                ), "target_idx mask must be the same length as adata!"
                target_idx = np.arange(adata.n_obs)[target_idx]
            model_fn = partial(
                self.get_specific_normalized_expression,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                expression_type=None,
                indices_to_return_salient=target_idx,
            )
        else:
            model_fn = partial(
                self.get_specific_normalized_expression,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                expression_type="salient",
                indices_to_return_salient=None,
            )

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=scrna_raw_counts_properties,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )
        return result

    @staticmethod
    @torch.inference_mode()
    def _preprocess_normalized_expression(
        generative_outputs: dict[str, torch.Tensor],
        generative_output_key: str,
        gene_mask: list | slice,
        scaling: float,
    ) -> np.ndarray:
        output = generative_outputs[generative_output_key]
        output = output[..., gene_mask]
        output *= scaling
        output = output.cpu().numpy()
        return output

    @torch.inference_mode()
    def get_latent_library_size(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        r"""Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        libraries = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            outputs = self.module._generic_inference(x=x, batch_index=batch_index)

            library = outputs["library"]
            if not give_mean:
                library = torch.exp(library)
            else:
                ql = (outputs["ql_m"], outputs["ql_v"])
                if ql is None:
                    raise RuntimeError(
                        "The module for this model does not compute the posterior"
                        "distribution for the library size. Set `give_mean` to False"
                        "to use the observed library size instead."
                    )
                library = torch.distributions.LogNormal(ql[0], ql[1]).mean
            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()
