#!/usr/bin/env python3
"""scVI baseline for the M2-1 lineage test: raw counts, the same cells Haruka trained on
(control + disease arms; halo excluded), one section so no batch key, 32 latent dims, seed 0."""
import numpy as np, anndata as ad, scvi
scvi.settings.seed = 0
a = ad.read_h5ad('petracer_M2-1_lineage.h5ad'); a = a[a.obs.condition.astype(str) != 'halo'].copy()
a.X = a.layers['counts']
scvi.model.SCVI.setup_anndata(a)
m = scvi.model.SCVI(a, n_latent=32, gene_likelihood='nb'); m.train(max_epochs=200, early_stopping=True)
np.save('scvi32.npy', m.get_latent_representation()); np.save('scvi32_obs_names.npy', a.obs_names.to_numpy())
print('saved', m.get_latent_representation().shape)
