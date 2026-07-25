import os


import sys

import scanpy as sc 
import anndata  as ad 
import squidpy as sq
import pandas as pd 
import numpy as  np


import scenvi
import anndata as ad 
import numpy as np

import torch
import os 

import scvi


import cellcharter as cc
import anndata as ad 

import numpy as np

import scipy.sparse

from pytorch_lightning import seed_everything

import pytorch_lightning as pl

class Haruka():
    def __init__(self, adata, cov_gene_num=32, seed=42):
        self.adata = adata
        self.input_adata = None
        self.cov_gene_num = cov_gene_num
        self.background_indices = None 
        self.target_indices = None
        self.model = None
        self.output_adata = None

    def construct_cov(self):

        g = self.cov_gene_num
        adata = self.adata.copy()

        if scipy.sparse.issparse(adata.X):
            adata.X  = adata.X.toarray()

        if g <= 0:
            self.input_adata = adata
            self.cov_gene_num = 0
            return

        try:
            covet, covet_sqrt, gene = scenvi.utils.compute_covet(adata, g=g, batch_key='slice_id', batch_size=100000)
        except TypeError:
            try:
                covet, covet_sqrt, gene = scenvi.utils.compute_covet(adata, g=g, batch_key='slice_id')
            except Exception:
                self.input_adata = adata
                self.cov_gene_num = 0
                return
        except Exception:
            self.input_adata = adata
            self.cov_gene_num = 0
            return

        if g == -1 :
            g = adata.X.shape[1]

        new_adata = ad.AnnData(X=np.concatenate([adata.X, covet_sqrt.reshape(-1, g*g)], axis=1), obs=adata.obs)
        new_adata.obsm = adata.obsm
        self.input_adata = new_adata
    
    def setup_model(self, latent_dim=256, gene_likelihood='poisson', me_weight=0.3, batch_key='slice_id', wasserstein_penalty=0.2, condition_key='condition', condition_label='disease', seed=42, layer=None):
        # Use counts layer if present to satisfy ContrastiveVI count-data requirement
        if layer is None and "counts" in self.input_adata.layers:
            layer = "counts"
        scvi.external.ContrastiveVI.setup_anndata(self.input_adata, batch_key=batch_key, layer=layer)

        self.model = scvi.external.ContrastiveVI(
            self.input_adata, n_me=self.cov_gene_num * self.cov_gene_num,  n_salient_latent=latent_dim, n_background_latent=latent_dim, use_observed_lib_size=True, wasserstein_penalty=wasserstein_penalty,
            gene_likelihood=gene_likelihood, me_weight=me_weight
        )

        self.background_indices = np.where(self.input_adata.obs[condition_key] != condition_label)[0]
        self.target_indices = np.where(self.input_adata.obs[condition_key] == condition_label)[0]

    def train(self, max_epochs=200,  batch_size=512, **trainer_kwargs):

        #current_seed = torch.random.initial_seed()
        #print(current_seed) 

        self.model.train(
            background_indices=self.background_indices,
            target_indices=self.target_indices,
            early_stopping=False,
            train_size=1.0,
            max_epochs=max_epochs,
            batch_size=batch_size,
            gradient_clip_val=1.0,
            **trainer_kwargs,
        )

        self.output_adata = self.input_adata[self.target_indices].copy()

    def extract_rep(self, n_layer=3):
        if "slice_id" in self.output_adata.obs.columns:
            self.output_adata.obs["slice_id"] = self.output_adata.obs["slice_id"].astype("category")

        self.output_adata.obsm["salient_rep"] = self.model.get_latent_representation(
            self.output_adata, representation_kind="salient"
        ).astype(np.float32)


        self.output_adata.obsm["background_rep"] = self.model.get_latent_representation(
            self.output_adata, representation_kind="background"
        ).astype(np.float32)


        sq.gr.spatial_neighbors(self.output_adata, library_key='slice_id', spatial_key='spatial')\

        cc.gr.remove_long_links(self.output_adata)

        cc.gr.aggregate_neighbors(self.output_adata, n_layers=n_layer, use_rep='salient_rep', out_key='X_cellcharter_salient', sample_key='slice_id')

        cc.gr.aggregate_neighbors(self.output_adata, n_layers=n_layer, use_rep='background_rep', out_key='X_cellcharter_background', sample_key='slice_id')
    
    def cluster(self, salient_cluster_num=4, background_cluster_number=4, seed=42, cluster_method="cellcharter", **cluster_kwargs):
        # cluster_method='cellcharter' is the original path and stays the default.
        # 'kmeans' / 'minibatch_kmeans' / 'gmm' route to haruka_cluster.FastCluster,
        # which standardises and PCA-reduces first. That renumbers and re-partitions
        # domains, so it is opt-in rather than a new default.
        if cluster_method != "cellcharter":
            from haruka_cluster import FastCluster

            for kind, rep, n_clusters in (
                ("salient", "X_cellcharter_salient", salient_cluster_num),
                ("background", "X_cellcharter_background", background_cluster_number),
            ):
                model = FastCluster(
                    n_clusters=n_clusters,
                    method=cluster_method,
                    random_state=seed,
                    **cluster_kwargs,
                )
                self.output_adata.obs[f"cluster_haruka_{kind}"] = model.fit_predict(
                    self.output_adata, use_rep=rep
                )
            return

        # cluster_kwargs is only meaningful on the FastCluster path above. Dropping
        # leftovers silently here would turn a misspelled keyword into a wrong
        # result rather than an error: cluster(backgroud_cluster_number=7) would
        # bind the typo into **cluster_kwargs, discard it, and cluster 4 background
        # domains while the caller believes they asked for 7. Before this method
        # took **cluster_kwargs that raised TypeError, so restore that.
        if cluster_kwargs:
            raise TypeError(
                f"cluster() got unexpected keyword arguments {sorted(cluster_kwargs)}; "
                "these only apply when cluster_method != 'cellcharter'"
            )

        model = cc.tl.Cluster(n_clusters=salient_cluster_num, trainer_params=dict(accelerator='gpu', devices=1), random_state=seed)

        pl.seed_everything(seed)

        model.fit(self.output_adata, use_rep='X_cellcharter_salient')

        self.output_adata.obs['cluster_haruka_salient'] = model.predict(self.output_adata, use_rep='X_cellcharter_salient')

        model = cc.tl.Cluster(n_clusters= background_cluster_number, trainer_params=dict(accelerator='gpu', devices=1), random_state=seed)
        
        pl.seed_everything(seed)

        model.fit(self.output_adata, use_rep='X_cellcharter_background')

        self.output_adata.obs['cluster_haruka_background'] = model.predict(self.output_adata, use_rep='X_cellcharter_background')


