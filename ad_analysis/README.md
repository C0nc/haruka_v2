# Latest AD figures

Consolidated on 2026-08-28 from the latest shared three-objective Haruka runs
(decoder-only batch injection, lambda 25, three-hop CellCharter).

## STARmap AD

- `starmap_four_slot_factor_matrix.png`
- `starmap_latest_overview.png`
- `starmap_background_salient_cellcharter_umap.png`
- `starmap_spatial_kmeans_disease_slices.png`

Source result root:
`bench_real_discovery/results_shared_full_niche_decoderonly_lambda25/ad`

## CODEX AD

- `codex_plaque_four_slot_umap_cc3.png`
- `codex_spatial_kmeans_ad_donors_cc3.png`

Source result root:
`bench_codex_cns_2026/results_shared_full_niche_decoderonly_lambda25/codex_cns_myeloid_primary_no_aml_no_combat/haruka_z_s_cellcharter_cc3_decoderbatch8_lambda25_seed0`

Older model ablations and the CODEX one-hop sensitivity analysis are not
included.

## Plaque-distance spatial maps

- `starmap_spatial_distance_to_abeta_ptau.png`: disease slices only; distances
  to the nearest Aβ-associated cell, p-tau-associated cell, and either type,
  computed within each slice in the dataset spatial-coordinate units.
- `codex_spatial_distance_to_dense_diffuse_abeta.png`: AD donors only; distances
  to official dense- and diffuse-Aβ plaque-object centroids plus the native
  all-plaque distance. The official 0.3774 µm/pixel conversion is used.

The CODEX protein panel does not contain a tau marker, so no CODEX tau distance
is shown. The corresponding per-cell distance tables are
`starmap_plaque_distance_cells.csv.gz` and
`codex_plaque_distance_cells.csv.gz`. The reproducible plotting code is
`plot_plaque_distance_spatial.py`.

## CellCharter PC1 versus plaque distance

- `scc_pc1_vs_plaque_distance.png`: disease-only within-slice/donor distance
  curves for s-CC PC1.
- `zcc_scc_pc1_plaque_distance_comparison.png`: matched z-CC/s-CC PC1
  comparison.
- `cc_pc1_plaque_distance_association.csv`: pooled and sample-controlled
  statistics.
- `cc_pc1_plaque_distance_by_sample.csv`: replicate-specific statistics.

PC1 was fitted separately on standardized condition-cell embeddings. Its sign
is arbitrary, so biological interpretation uses association magnitude and
cross-sample consistency rather than the raw sign alone.
