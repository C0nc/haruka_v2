#!/usr/bin/env python3
"""Denoised expression baseline for STARmap: scVI on raw counts of all 2,766 genes, all three
slices, batch = slice, 32 latent dims (same width as z and s), seed 0."""
import numpy as np, anndata as ad, scvi
scvi.settings.seed = 0
a = ad.read_h5ad('/home/yancui/haruka_v2/bench_real_discovery/data/ad.h5ad')
a.X = a.layers['counts']
scvi.model.SCVI.setup_anndata(a, batch_key='slice_id')
m = scvi.model.SCVI(a, n_latent=32, gene_likelihood='nb'); m.train(max_epochs=200, early_stopping=True)
np.save('scvi32.npy', m.get_latent_representation()); np.save('scvi32_obs_names.npy', a.obs_names.to_numpy())
print('saved', m.get_latent_representation().shape)
