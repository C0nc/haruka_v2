"""
Haruka for spatial multiomics: RNA + ATAC with PeakVI-style Poisson for peaks.
Produces salient and background embeddings per modality, then integrated multiomics embeddings
and three types of domain clustering (RNA salient, ATAC salient, multi salient).
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import squidpy as sq
import scipy.sparse
import torch
import scenvi
import scvi
import cellcharter as cc
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

from Haruka import Haruka


def ensure_spatial(adata, array_row_key="array_row", array_col_key="array_col", spatial_key="spatial"):
    """Set obsm['spatial'] from obs['array_row'] and obs['array_col'] if spatial is missing.
    Also accepts 'array_rol' as typo for 'array_col'."""
    if spatial_key in adata.obsm and adata.obsm[spatial_key] is not None:
        return adata
    row_key = array_row_key if array_row_key in adata.obs.columns else "array_row"
    col_key = (
        array_col_key
        if array_col_key in adata.obs.columns
        else ("array_col" if "array_col" in adata.obs.columns else "array_rol")
    )
    if row_key not in adata.obs.columns or col_key not in adata.obs.columns:
        raise ValueError(f"Need obs['{row_key}'] and obs['{col_key}'] or obsm['{spatial_key}']")
    adata.obsm[spatial_key] = adata.obs[[col_key, row_key]].values.astype(np.float32)
    return adata


class HarukaPeak(Haruka):
    """
    Haruka for ATAC/peak data with PeakVI-style Poisson likelihood.
    Same pipeline as Haruka but gene_likelihood='poisson' and COVET computed on peak matrix.
    """

    def __init__(self, adata, cov_peak_num=32, seed=42):
        super().__init__(adata, cov_gene_num=cov_peak_num, seed=seed)

    def setup_model(
        self,
        latent_dim=256,
        gene_likelihood="poisson",  # PeakVI-style Poisson for accessibility
        me_weight=0.3,
        batch_key="slice_id",
        wasserstein_penalty=0.2,
        condition_key="condition",
        condition_label="disease",
        seed=42,
        layer=None,
    ):
        if layer is None and "counts" in self.input_adata.layers:
            layer = "counts"
        scvi.external.ContrastiveVI.setup_anndata(self.input_adata, batch_key=batch_key, layer=layer)
        self.model = scvi.external.ContrastiveVI(
            self.input_adata,
            n_me=self.cov_gene_num * self.cov_gene_num,
            n_salient_latent=latent_dim,
            n_background_latent=latent_dim,
            use_observed_lib_size=True,
            wasserstein_penalty=wasserstein_penalty,
            gene_likelihood=gene_likelihood,
            me_weight=me_weight,
        )
        self.background_indices = np.where(
            self.input_adata.obs[condition_key] != condition_label
        )[0]
        self.target_indices = np.where(
            self.input_adata.obs[condition_key] == condition_label
        )[0]


class HarukaMultiomics:
    """
    Spatial multiomics Haruka: RNA (Poisson/NB) + ATAC (PeakVI-style Poisson).
    - RNA: COVET + ContrastiveVI -> salient + background -> CellCharter aggregate.
    - ATAC: COVET on peaks + ContrastiveVI (Poisson) -> salient + background -> CellCharter aggregate.
    - Concatenate RNA and ATAC embeddings -> integrated salient and background.
    - Three domain clusterings: RNA salient, ATAC salient, multi salient (and background variants).
    """

    def __init__(
        self,
        adata_rna,
        adata_atac,
        cov_gene_num=32,
        cov_peak_num=32,
        seed=42,
    ):
        self.adata_rna = adata_rna
        self.adata_atac = adata_atac
        self.cov_gene_num = cov_gene_num
        self.cov_peak_num = cov_peak_num
        self.seed = seed
        self.haruka_rna = None
        self.haruka_atac = None
        self.output_rna = None
        self.output_atac = None
        self.output_multi = None  # anndata with concatenated embeddings and cluster columns

    def run_rna(
        self,
        latent_dim=256,
        gene_likelihood="poisson",
        n_layer=3,
        salient_cluster_num=4,
        background_cluster_number=4,
        condition_key="condition",
        condition_label="disease",
        batch_key="slice_id",
        max_epochs=200,
        batch_size=512,
        me_weight_rna=0.3,
        cluster_method="cellcharter",
        **kwargs,
    ):
        """Run Haruka on RNA; use Poisson (PeakVI-style) or 'nb'."""
        seed_everything(self.seed)
        train_kwargs = {k: v for k, v in kwargs.items() if k in ("plan_kwargs", "datasplitter_kwargs")}
        setup_kwargs = {k: v for k, v in kwargs.items() if k not in ("plan_kwargs", "datasplitter_kwargs")}
        self.haruka_rna = Haruka(self.adata_rna, cov_gene_num=self.cov_gene_num, seed=self.seed)
        self.haruka_rna.construct_cov()
        self.haruka_rna.setup_model(
            latent_dim=latent_dim,
            gene_likelihood=gene_likelihood,
            batch_key=batch_key,
            condition_key=condition_key,
            condition_label=condition_label,
            me_weight=me_weight_rna,
            **setup_kwargs,
        )
        self.haruka_rna.train(max_epochs=max_epochs, batch_size=batch_size, **train_kwargs)
        self.haruka_rna.extract_rep(n_layer=n_layer)
        self.haruka_rna.cluster(
            salient_cluster_num=salient_cluster_num,
            background_cluster_number=background_cluster_number,
            seed=self.seed,
            cluster_method=cluster_method,
        )
        self.output_rna = self.haruka_rna.output_adata
        return self.output_rna

    def run_atac(
        self,
        latent_dim=256,
        n_layer=3,
        salient_cluster_num=4,
        background_cluster_number=4,
        condition_key="condition",
        condition_label="disease",
        batch_key="slice_id",
        max_epochs=200,
        batch_size=512,
        me_weight_atac=0.3,
        cluster_method="cellcharter",
        **kwargs,
    ):
        """Run Haruka on ATAC with PeakVI-style Poisson for peaks."""
        seed_everything(self.seed)
        train_kwargs = {k: v for k, v in kwargs.items() if k in ("plan_kwargs", "datasplitter_kwargs")}
        setup_kwargs = {k: v for k, v in kwargs.items() if k not in ("plan_kwargs", "datasplitter_kwargs")}
        self.haruka_atac = HarukaPeak(
            self.adata_atac, cov_peak_num=self.cov_peak_num, seed=self.seed
        )
        self.haruka_atac.construct_cov()
        self.haruka_atac.setup_model(
            latent_dim=latent_dim,
            gene_likelihood="poisson",
            batch_key=batch_key,
            condition_key=condition_key,
            condition_label=condition_label,
            me_weight=me_weight_atac,
            **setup_kwargs,
        )
        self.haruka_atac.train(max_epochs=max_epochs, batch_size=batch_size, **train_kwargs)
        self.haruka_atac.extract_rep(n_layer=n_layer)
        self.haruka_atac.cluster(
            salient_cluster_num=salient_cluster_num,
            background_cluster_number=background_cluster_number,
            seed=self.seed,
            cluster_method=cluster_method,
        )
        self.output_atac = self.haruka_atac.output_adata
        return self.output_atac

    def concat_embeddings_and_cluster(
        self,
        multi_salient_cluster_num=4,
        multi_background_cluster_number=4,
        sample_key="slice_id",
        cluster_method="cellcharter",
    ):
        """
        Align RNA and ATAC output by cell index, concatenate embeddings,
        and run domain clustering on the integrated multiomics embeddings.
        """
        if self.output_rna is None or self.output_atac is None:
            raise RuntimeError("Run run_rna() and run_atac() first.")

        # Align by common cells (assume same order if paired; otherwise inner join)
        common = self.output_rna.obs.index.intersection(self.output_atac.obs.index)
        if len(common) == 0:
            raise ValueError("No common cell IDs between RNA and ATAC output.")
        rna_idx = [self.output_rna.obs.index.get_loc(c) for c in common]
        atac_idx = [self.output_atac.obs.index.get_loc(c) for c in common]

        X_salient_rna = self.output_rna.obsm["X_cellcharter_salient"][rna_idx]
        X_background_rna = self.output_rna.obsm["X_cellcharter_background"][rna_idx]
        X_salient_atac = self.output_atac.obsm["X_cellcharter_salient"][atac_idx]
        X_background_atac = self.output_atac.obsm["X_cellcharter_background"][atac_idx]

        X_multi_salient = np.concatenate([X_salient_rna, X_salient_atac], axis=1)
        X_multi_background = np.concatenate([X_background_rna, X_background_atac], axis=1)

        # Build multi anndata (use RNA as base for obs)
        multi_obs = self.output_rna.obs.loc[common].copy()
        multi_obsm = {}
        multi_obsm["X_cellcharter_salient_rna"] = X_salient_rna
        multi_obsm["X_cellcharter_background_rna"] = X_background_rna
        multi_obsm["X_cellcharter_salient_atac"] = X_salient_atac
        multi_obsm["X_cellcharter_background_atac"] = X_background_atac
        multi_obsm["X_multi_salient"] = X_multi_salient
        multi_obsm["X_multi_background"] = X_multi_background
        if "spatial" in self.output_rna.obsm:
            multi_obsm["spatial"] = self.output_rna.obsm["spatial"][rna_idx]

        self.output_multi = ad.AnnData(X=X_multi_salient, obs=multi_obs)
        self.output_multi.obsm = multi_obsm

        # Copy cluster columns from RNA and ATAC
        self.output_multi.obs["cluster_haruka_salient_rna"] = self.output_rna.obs.loc[
            common, "cluster_haruka_salient"
        ].values
        self.output_multi.obs["cluster_haruka_background_rna"] = self.output_rna.obs.loc[
            common, "cluster_haruka_background"
        ].values
        self.output_multi.obs["cluster_haruka_salient_atac"] = self.output_atac.obs.loc[
            common, "cluster_haruka_salient"
        ].values
        self.output_multi.obs["cluster_haruka_background_atac"] = self.output_atac.obs.loc[
            common, "cluster_haruka_background"
        ].values

        # Domain clustering on integrated multiomics embeddings.
        # 'cellcharter' is the original path and stays the default; see Haruka.cluster.
        if cluster_method != "cellcharter":
            from haruka_cluster import FastCluster

            for kind, n_clusters in (
                ("salient", multi_salient_cluster_num),
                ("background", multi_background_cluster_number),
            ):
                model = FastCluster(
                    n_clusters=n_clusters, method=cluster_method, random_state=self.seed
                )
                self.output_multi.obs[f"cluster_haruka_{kind}_multi"] = model.fit_predict(
                    self.output_multi, use_rep=f"X_multi_{kind}"
                )
            return self._finalize_multi(sample_key)

        pl.seed_everything(self.seed)
        model_salient = cc.tl.Cluster(
            n_clusters=multi_salient_cluster_num,
            trainer_params=dict(accelerator="gpu", devices=1),
            random_state=self.seed,
        )
        model_salient.fit(self.output_multi, use_rep="X_multi_salient")
        self.output_multi.obs["cluster_haruka_salient_multi"] = model_salient.predict(
            self.output_multi, use_rep="X_multi_salient"
        )

        model_background = cc.tl.Cluster(
            n_clusters=multi_background_cluster_number,
            trainer_params=dict(accelerator="gpu", devices=1),
            random_state=self.seed,
        )
        model_background.fit(self.output_multi, use_rep="X_multi_background")
        self.output_multi.obs["cluster_haruka_background_multi"] = model_background.predict(
            self.output_multi, use_rep="X_multi_background"
        )

        return self._finalize_multi(sample_key)

    def _finalize_multi(self, sample_key):
        # Add spatial neighbors for visualization (squidpy)
        if "spatial" in self.output_multi.obsm and sample_key in self.output_multi.obs.columns:
            sq.gr.spatial_neighbors(
                self.output_multi,
                library_key=sample_key,
                spatial_key="spatial",
            )
            try:
                cc.gr.remove_long_links(self.output_multi)
            except Exception:
                pass

        return self.output_multi
