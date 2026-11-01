# CODEX-CNS Haruka result

## Primary design

- Source: official `figure7_mg_raw.h5ad` (48,146 IBA1+ cells x 10 protein
  intensities) and `figure7_clusters.csv` from the CODEX-CNS GitHub repository.
- Primary cells: 42,067 cells from AD-1/2/3/4 (26,275) versus
  Ctl-1/3/4 (15,792). Ctl-2/sample 3628 is excluded because the official
  analysis notebook identifies it as an AML control.
- Preprocessing: `log1p(raw intensity)`, then global per-protein z-score with
  `max_value=10`; **no donor ComBat**. Donor is nested within condition, so
  unconditional donor ComBat would risk removing the AD contrast.
- Haruka input: feature z-score clipped to [-5,5] and affine-shifted to [0,2].
  Reconstruction uses exact per-cell summed MSE for continuous protein. No
  RNA count likelihood is used.
- Fit: seed 0, reference 75 epochs, joint 150 epochs, 32-dimensional z and s,
  runtime 107.9 seconds on one H200 GPU. Plaque labels were not used in fitting.

## Spatial readout

CellCharter used one identical Delaunay graph/readout for z and s, constructed
within donor x condition, with long-link removal and self + three mean-hop
blocks. The graph has 249,510 directed edges, median degree 6 (range 2-13),
zero cross-donor/condition edges, and zero self edges. Control s was set to
zero before neighborhood aggregation.

## Donor-held-out evaluation

Plaque endpoints use 12,998 AD gray-matter cells across AD-1/2/3/4. The label
is distance to the nearest generic plaque annotation in the bundled H5AD; it
does not distinguish dense from diffuse. Results below are means across four
leave-one-AD-donor-out folds.

| Endpoint | z ROC-AUC / AP | s ROC-AUC / AP | z-CC ROC-AUC / AP | s-CC ROC-AUC / AP |
|---|---:|---:|---:|---:|
| plaque overlap (0 um) | 0.645 / 0.198 | 0.586 / 0.143 | 0.596 / 0.156 | 0.585 / 0.138 |
| within 5 um | 0.593 / 0.297 | 0.573 / 0.264 | 0.539 / 0.244 | 0.565 / 0.260 |
| within 30 um | 0.547 / 0.654 | 0.567 / 0.664 | 0.462 / 0.602 | 0.565 / 0.662 |

The plaque signal is modest and is not uniquely salient. Background z is best
for direct overlap and 5-um recovery, while s is slightly best at 30 um. This
does not support a strong claim that Haruka isolated a plaque-specific salient
factor in this 10-protein panel.

Annotation leave-one-donor-out balanced accuracy is 0.194 (z), 0.167 (s),
0.197 (z-CC), and 0.151 (s-CC) for the 11 publication-initial Leiden clusters;
GM/WM balanced accuracy is 0.579, 0.551, 0.590, and 0.469, respectively.

## Important artifact paths

- Corrected primary H5AD:
  `data/processed/codex_cns_myeloid_primary_no_aml_no_combat.h5ad`
- Full-eight-donor sensitivity H5AD:
  `data/processed/codex_cns_myeloid_sensitivity_all8_no_combat.h5ad`
- Haruka embeddings: `results/codex_cns_myeloid_primary_no_aml_no_combat/haruka_mender_mse_seed0/`
- CellCharter embeddings: `results/codex_cns_myeloid_primary_no_aml_no_combat/haruka_cellcharter_cc3_seed0/`
- Evaluation table: `.../evaluation/donor_heldout_readouts_summary.csv`
- Four-slot plaque UMAP: `.../evaluation/plaque_four_slot_umap.png`

The earlier `primary_no_aml.h5ad` and corresponding result directory used
unconditional donor ComBat and are retained only as a diagnostic. They must
not be reported as the primary result.

## Official URLs

- Paper: https://www.nature.com/articles/s41593-026-02267-3
- Code/cell tables: https://github.com/BaharehAjami/CODEX-CNS
- Main image data: https://zenodo.org/records/20725083,
  https://zenodo.org/records/20706336,
  https://zenodo.org/records/20726163
- Main segmentation and plaque masks:
  https://zenodo.org/records/21342581,
  https://zenodo.org/records/21347021,
  https://zenodo.org/records/21388042,
  https://zenodo.org/records/21417215,
  https://zenodo.org/records/21419095

The complete 139-GB QPTIFF + dense/diffuse-mask download is described by
`download_manifest.tsv` and can be resumed and checksum-verified with
`download_official_data.sh`. It was not started because the analysis-ready
protein/plaque H5AD was prioritized.
