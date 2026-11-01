# PEtracer Haruka: new asymmetric CC only

All figures use the current asymmetric readout on four independently trained
M1-T1 section pairs (M1-1 through M1-4). No original-CC figures are included.

Background z-CC uses the full-cell graph and three spatial hops. Salient s-CC
uses malignant cells, one spatial hop, and retains the focal-cell s latent.

## Main summary figures

- `01_current_default_ari_boxplot.png`: current asymmetric z-CC/s-CC benchmark.
- `02_condition_only_z_s_cc_umap.png`: z-CC and s-CC UMAPs using condition-arm
  cells only. Control cells are excluded from both UMAP fits and panels.
- `03_salient_spatial_enrichment.png`: condition-arm salient clusters, lineage
  families, tissue locations, programs, contacts, and tumor distance.
- `04_cell_state_distance_effect_sizes.png`: variance explained by salient
  clusters for lineage state and spatial variables.

## Per-section figures

- `M1-*_current_z_s_cc_umap.png`: current asymmetric z-CC/s-CC UMAPs. Both the
  background and salient UMAPs contain condition-arm cells only.
- `M1-*_salient_spatial_enrichment.png`: six-cluster condition-arm salient
  enrichment panels containing UMAP, physical tissue, tumor distance, paired
  programs, and neighboring cell-type contact enrichment.

The response programs are synthetic by construction. Spatial cell-type contact
enrichments reflect pre-existing tissue geometry because control and condition
have identical cell locations.
